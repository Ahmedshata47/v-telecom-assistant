"""Web API running the non-RAGAS agent graph from voda.ipynb.

The graph is a faithful port of the last cell of
C:\\Users\\ASUS\\Desktop\\3gpp_specs\\voda.ipynb:

    START -> route -> retrieve -> evaluate_retrieval
                                    |            |
                              generate      prepare_retry -> retrieve_retry
                                    |                                    |
                                   END                                 |
                                    ^                                  |
                                    +--- evaluate_retrieval <-----------+

RAGAS (notebook section 34) is deliberately excluded. Everything else is
preserved: the router, hybrid dense + BM25 + RRF retrieval with explicit
TS filtering, the Mistral retrieval critic, corrective multi-query retrieval
with a second query-level RRF, and local citation validation.

On top of the notebook this server adds developer telemetry only. It records
every retrieval round, the critic verdict for that round, the reason the graph
branched, and the raw router output. That is what lets the demo show whether RAG
was used, what was retrieved, and whether a corrective retry happened. No
prompt, threshold or routing decision is changed by the telemetry.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypedDict

import chromadb
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from peft import PeftModel
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Graph node key -> the name the node is called in voda.ipynb. The browser shows
# these verbatim so the transcript maps one-to-one onto the notebook's nodes.
NODE_LABELS = {
    "route": "Router",
    "retrieve": "Hybrid retrieval",
    "evaluate_retrieval": "Mistral critic",
    "prepare_retry": "Corrective retrieval",
    "retrieve_retry": "Multi-query RRF",
    "generate": "Generation",
}

ROOT = Path(os.getenv("TELECOM_ROOT", r"C:\Users\ASUS\Desktop\3gpp_specs"))
BASE_MODEL = Path(os.getenv("BASE_MODEL", r"C:\Users\ASUS\Desktop\telecom_project\models\qwen2.5-1.5b"))
ADAPTER = ROOT / "Tuned_model"
COLLECTION = "3gpp_specs"

TOP_K = 5
CANDIDATE_K = 40
BM25_CANDIDATE_K = 40
RRF_K = 60
MAX_RETRIEVAL_ATTEMPTS = 2

CITATION_PATTERN = re.compile(r"\[\s*(?:TS\s*)?(\d{2}\.\d{3})[,\s]+(?:Section|Clause)\s+([\d.]+)\s*\]", re.I)
SPEC_PATTERN = re.compile(r"\bTS\s*(\d{2}\.\d{3})\b", re.I)

# The embedding model must be the exact one the Chroma collection was built
# with in voda.ipynb cell "CREATE CHROMADB CLIENT". The collection stores
# 768-dimensional vectors, so querying it with MiniLM (384-d) raises a
# dimension mismatch and the retrieval branch can never run.
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# voda.ipynb's route_query() router prompt, verbatim. ask() wraps it as
# "Question:\n<text>", so the prefix/suffix here reproduce the notebook's
#     f"""
#         return 'NO' if you know the answer to the question: {question} and 'YES' otherwise.
#
#         return ONLY 'YES' or 'NO'.
#         Decision:
#         """
# including its leading newline and four-space indentation.
ROUTER_PROMPT_PREFIX = "\n    return 'NO' if you know the answer to the question: "
ROUTER_PROMPT_SUFFIX = " and 'YES' otherwise.\n\n    return ONLY 'YES' or 'NO'.\n    Decision:\n    "


def router_prompt(question: str) -> str:
    """The exact string voda.ipynb passes to ask() for the routing decision."""
    return ROUTER_PROMPT_PREFIX + question + ROUTER_PROMPT_SUFFIX

# ============================================================
# 2. CITATION HELPERS
# ============================================================


def citation(item: Dict[str, Any]) -> Optional[str]:
    """Canonical citation string from retrieved metadata."""
    spec = str(item.get("spec", "")).removeprefix("TS ").strip()
    section = str(item.get("section", "")).strip()
    if not spec or not section:
        return None
    return f"[TS {spec}, Section {section}]"


def validated_citation(answer: str, context: List[Dict[str, Any]]) -> Optional[str]:
    """Accept only a citation that belongs to one of the retrieved chunks."""
    allowed = {citation(item) for item in context}
    allowed.discard(None)
    for spec, section in CITATION_PATTERN.findall(answer):
        text = f"[TS {spec}, Section {section}]"
        if text in allowed:
            return text
    return None


def ensure_retrieved_citation(answer: str, context: List[Dict[str, Any]]) -> Tuple[str, bool]:
    """Keep a verified model citation, otherwise attach the top retrieved source.

    The fallback citation is selected locally from retrieval metadata rather than
    generated by another model, so it can never be hallucinated.
    """
    verified = validated_citation(answer, context)
    if verified:
        return answer, True

    cleaned = CITATION_PATTERN.sub("", answer).strip()
    if "insufficient" in cleaned.lower():
        return cleaned, False

    fallback = next((citation(item) for item in context if citation(item)), None)
    if fallback:
        return f"{cleaned}\n{fallback}", False
    return cleaned, False


# ============================================================
# 3. STRUCTURED CRITIC OUTPUT
# ============================================================


class RetrievalEvaluation(BaseModel):
    relevant: bool = Field(description="True only if the retrieved chunks contain enough evidence to answer the question.")
    score: float = Field(description="Context sufficiency score from 0 to 1.")
    missing_information: str = Field(description="Information needed but missing from the context.")
    search_queries: List[str] = Field(description="Up to three better search queries. Return [] if the context is sufficient.")


RETRIEVAL_EVALUATION_PROMPT = """
You are the retrieval critic for a 3GPP telecom RAG system.

DO NOT answer the user's question.

Your ONLY job is to determine whether the retrieved context
contains enough evidence to answer the question accurately.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Rules:

1. relevant=true ONLY when the retrieved chunks contain enough
   evidence to answer the question reliably.

2. relevant=false when:
   - the chunks are only loosely related,
   - the required procedure is missing,
   - the required message is missing,
   - the required requirement is missing,
   - the context is incomplete,
   - the answer would require guessing,
   - an exact TS/section requested by the user is not represented.

3. Do NOT use your own knowledge to fill missing information.

4. Explain what is missing.

5. If relevant=false, generate up to THREE improved search queries.

6. Use exact 3GPP terminology whenever possible.

7. Preserve an explicitly mentioned TS number.

8. If the context is sufficient, return an empty search_queries list.

9. The score represents CONTEXT SUFFICIENCY,
   not general topic similarity.

Return the requested structured evaluation.
"""


def format_context_for_judge(context: List[Dict[str, Any]]) -> str:
    if not context:
        return "NO CONTEXT WAS RETRIEVED."
    return "\n\n".join(
        f"""
[Chunk {i}]

Specification:
{item.get('spec', '')}

Section:
{item.get('section', '')}

Title:
{item.get('title', '')}

Content:
{item.get('text', '')}
"""
        for i, item in enumerate(context, 1)
    )


# ============================================================
# 4. AGENT STATE
# ============================================================


class AgentState(TypedDict, total=False):
    # Notebook fields
    question: str
    options: list
    use_rag: bool
    route_reason: str
    context: list
    answer: str
    cited_by_local_model: bool
    retrieval_attempt: int
    retrieval_query: str
    retrieval_queries: list
    retrieval_relevant: bool
    retrieval_score: float
    missing_information: str
    expanded_queries: list
    retry_count: int
    # Developer telemetry (added for the demo; carries no decisions)
    router_output: str
    retry_decision: str
    rounds: list
    trace: list
    timings: dict


# ============================================================
# 5. AGENT
# ============================================================


class TelecomAgent:
    def __init__(self) -> None:
        missing = [str(p) for p in (ROOT / "chroma_db", BASE_MODEL, ADAPTER) if not p.exists()]
        if missing:
            raise RuntimeError("Missing local model/corpus path(s): " + ", ".join(missing))
        if not os.getenv("MISTRAL_API_KEY"):
            raise RuntimeError("MISTRAL_API_KEY is required for the Mistral retrieval critic.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required, matching voda.ipynb's QLoRA loading configuration.")

        self.lock = threading.Lock()
        self.loaded_at = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER)
        self.model.eval()

        self.embedding = SentenceTransformer(EMBEDDING_MODEL)

        self.collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection(COLLECTION)
        all_docs = self.collection.get(include=["documents", "metadatas"])
        self.docs = all_docs["documents"]
        self.metadata = all_docs["metadatas"]
        self.bm25 = BM25Okapi([re.findall(r"\b\w+\b", doc.lower()) for doc in self.docs])

        judge = ChatOpenAI(
            model="codestral-2508",
            base_url="https://api.mistral.ai/v1",
            api_key=os.environ["MISTRAL_API_KEY"],
            temperature=0,
        )
        self.critic = judge.with_structured_output(RetrievalEvaluation)
        self.graph = self.build_graph()

    # ------------------------------------------------------------------
    # Notebook: ask()
    # ------------------------------------------------------------------

    def ask_model(self, question: str, max_new_tokens: int = 256) -> str:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Question:\n{question}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in self.tokenizer(prompt, return_tensors="pt").items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Notebook: hybrid_retrieve()
    # ------------------------------------------------------------------

    def hybrid_retrieve(self, question: str) -> List[Dict[str, Any]]:
        """Dense + BM25 + RRF. Restricted to one specification when the query names a TS number."""
        spec = SPEC_PATTERN.search(question)
        embedding = self.embedding.encode(question, normalize_embeddings=True).tolist()
        args: Dict[str, Any] = {"query_embeddings": [embedding], "n_results": CANDIDATE_K}

        allowed = list(range(len(self.docs)))
        if spec:
            target = f"TS {spec.group(1)}"
            args["where"] = {"spec": target}
            allowed = [i for i, meta in enumerate(self.metadata) if meta.get("spec", "") == target]

        dense = self.collection.query(**args)
        scores = self.bm25.get_scores(re.findall(r"\b\w+\b", question.lower()))
        candidates: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        def blank(meta: Dict[str, Any], text: str) -> Dict[str, Any]:
            return {
                "text": text,
                "spec": meta.get("spec", ""),
                "section": meta.get("section", ""),
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "distance": None,
                "dense_rank": None,
                "bm25_rank": None,
                "bm25_score": None,
            }

        dense_documents = dense["documents"][0]
        dense_metadatas = dense["metadatas"][0]
        dense_distances = dense.get("distances", [[]])[0]
        for rank, index in enumerate(range(len(dense_documents)), 1):
            meta = dense_metadatas[index]
            item = blank(meta, dense_documents[index])
            item["distance"] = float(dense_distances[index]) if index < len(dense_distances) else None
            item["dense_rank"] = rank
            candidates[(item["spec"], str(item["section"]), item["text"])] = item

        for rank, index in enumerate(
            sorted(allowed, key=lambda i: scores[i], reverse=True)[:BM25_CANDIDATE_K], 1
        ):
            meta = self.metadata[index]
            key = (meta.get("spec", ""), str(meta.get("section", "")), self.docs[index])
            item = candidates.setdefault(key, blank(meta, self.docs[index]))
            item["bm25_rank"] = rank
            item["bm25_score"] = float(scores[index])

        for item in candidates.values():
            item["hybrid_score"] = sum(
                1 / (RRF_K + item[key]) for key in ("dense_rank", "bm25_rank") if item[key] is not None
            )

        return sorted(candidates.values(), key=lambda item: item["hybrid_score"], reverse=True)[:TOP_K]

    # ------------------------------------------------------------------
    # Notebook: merge_retrieval_results()
    # ------------------------------------------------------------------

    def merge_retrieval_results(self, result_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Second RRF across the per-query rankings from corrective retrieval."""
        merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for results in result_lists:
            for rank, item in enumerate(results, 1):
                key = (item.get("spec", ""), str(item.get("section", "")), item.get("text", ""))
                if key not in merged:
                    merged[key] = {**item, "query_rrf": 0.0, "query_hits": 0}
                merged[key]["query_rrf"] += 1 / (RRF_K + rank)
                merged[key]["query_hits"] += 1
        for item in merged.values():
            item["hybrid_score"] = item["query_rrf"]
        return sorted(merged.values(), key=lambda item: item["hybrid_score"], reverse=True)[:TOP_K]

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def event(state: AgentState, node: str, detail: str, status: str = "complete") -> List[Dict[str, str]]:
        return [*state.get("trace", []), {"node": node, "detail": detail, "status": status}]

    @staticmethod
    def summarise_chunks(context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "spec": item.get("spec", ""),
                "section": str(item.get("section", "")),
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "citation": citation(item),
                "hybrid_score": round(float(item.get("hybrid_score") or 0.0), 6),
                "dense_rank": item.get("dense_rank"),
                "bm25_rank": item.get("bm25_rank"),
                "bm25_score": item.get("bm25_score"),
                "distance": item.get("distance"),
                "query_hits": item.get("query_hits"),
            }
            for item in context
        ]

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def route(self, state: AgentState) -> Dict[str, Any]:
        decision = self.ask_model(router_prompt(state["question"]), 16).upper()
        use_rag = bool(re.search(r"\bYES\b", decision))
        reason = (
            "Authoritative 3GPP retrieval is needed."
            if use_rag
            else "External 3GPP retrieval is not required."
        )
        return {
            "use_rag": use_rag,
            "route_reason": reason,
            "router_output": decision.strip(),
            "retrieval_attempt": 0,
            "retrieval_query": state["question"],
            "retrieval_queries": [],
            "retrieval_relevant": False,
            "retrieval_score": 0.0,
            "missing_information": "",
            "expanded_queries": [],
            "retry_count": 0,
            "rounds": [],
            "trace": self.event(
                state,
                "Router",
                f"{'RAG selected' if use_rag else 'Direct answer selected'}. {reason} "
                f"Router said: \"{decision.strip()}\"",
            ),
        }

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        if not state["use_rag"]:
            return {"context": []}

        query = state.get("retrieval_query", state["question"])
        attempt = state.get("retrieval_attempt", 0)
        context = self.hybrid_retrieve(query)

        queries = list(state.get("retrieval_queries", []))
        if query not in queries:
            queries.append(query)

        rounds = [
            *state.get("rounds", []),
            {
                "round": attempt + 1,
                "kind": "hybrid",
                "queries": [query],
                "chunk_count": len(context),
                "chunks": self.summarise_chunks(context),
                "critic": None,
                "decision": "pending",
            },
        ]

        return {
            "context": context,
            "retrieval_query": query,
            "retrieval_queries": queries,
            "retrieval_attempt": attempt + 1,
            "rounds": rounds,
            "trace": self.event(
                state,
                "Hybrid retrieval",
                f"Round {attempt + 1}: dense({CANDIDATE_K}) + BM25({BM25_CANDIDATE_K}) + RRF returned {len(context)} chunks.",
            ),
        }

    def evaluate_retrieved_context(self, state: AgentState) -> Dict[str, Any]:
        prompt = RETRIEVAL_EVALUATION_PROMPT.format(
            question=state["question"], context=format_context_for_judge(state.get("context", []))
        )
        result = self.critic.invoke(prompt)

        score = max(0.0, min(1.0, float(result.score)))
        queries = [q.strip() for q in result.search_queries if q and q.strip()][:3]
        decision = self.retry_decision(state, bool(result.relevant), queries)

        rounds = list(state.get("rounds", []))
        if rounds:
            rounds[-1] = {
                **rounds[-1],
                "critic": {
                    "relevant": bool(result.relevant),
                    "score": score,
                    "missing_information": result.missing_information,
                    "search_queries": queries,
                },
                "decision": decision,
            }

        verdict = (
            "context sufficient"
            if result.relevant
            else f"context insufficient, {len(queries)} corrective quer{'y' if len(queries) == 1 else 'ies'} proposed"
        )
        return {
            "retrieval_relevant": bool(result.relevant),
            "retrieval_score": score,
            "missing_information": result.missing_information,
            "expanded_queries": queries,
            "retry_decision": decision,
            "rounds": rounds,
            "trace": self.event(
                state,
                "Mistral critic",
                f"Sufficiency {score:.0%}: {verdict}. Decision: {decision}.",
                "accepted" if result.relevant else "retry",
            ),
        }

    @staticmethod
    def retry_decision(state: AgentState, relevant: bool, queries: List[str]) -> str:
        """Notebook: after_retrieval_evaluation. Recorded during the node so the demo can show why it branched."""
        if relevant:
            return "generate"
        if state.get("retrieval_attempt", 0) >= MAX_RETRIEVAL_ATTEMPTS:
            return "generate_max_attempts"
        if not queries:
            return "generate_no_new_query"
        return "prepare_retry"

    @staticmethod
    def after_retrieval_evaluation(state: AgentState) -> str:
        decision = state.get("retry_decision", "generate")
        return "prepare_retry" if decision == "prepare_retry" else "generate"

    def prepare_retry(self, state: AgentState) -> Dict[str, Any]:
        queries = state["expanded_queries"]
        attempt = state["retry_count"] + 1
        return {
            "retrieval_query": queries[0],
            "retry_count": attempt,
            "trace": self.event(
                state,
                "Corrective retrieval",
                f"Retry {attempt}: retrieving again with {len(queries)} critic-generated quer"
                f"{'y' if len(queries) == 1 else 'ies'} plus the original question.",
                "retry",
            ),
        }

    def retrieve_retry_queries(self, state: AgentState) -> Dict[str, Any]:
        queries = list(
            dict.fromkeys(
                q.strip() for q in [state["question"], *state.get("expanded_queries", [])] if q and q.strip()
            )
        )
        context = self.merge_retrieval_results([self.hybrid_retrieve(query) for query in queries])

        history = list(state.get("retrieval_queries", []))
        for query in queries:
            if query not in history:
                history.append(query)

        attempt = state.get("retrieval_attempt", 0)
        previous = state.get("context", [])
        previous_keys = {(i.get("spec", ""), str(i.get("section", "")), i.get("text", "")) for i in previous}
        added = sum(
            1
            for item in context
            if (item.get("spec", ""), str(item.get("section", "")), item.get("text", "")) not in previous_keys
        )

        rounds = [
            *state.get("rounds", []),
            {
                "round": attempt + 1,
                "kind": "multi_query_rrf",
                "queries": queries,
                "chunk_count": len(context),
                "new_chunks_vs_previous": added,
                "chunks": self.summarise_chunks(context),
                "critic": None,
                "decision": "pending",
            },
        ]

        return {
            "context": context,
            "retrieval_query": queries[0],
            "retrieval_queries": history,
            "retrieval_attempt": attempt + 1,
            "rounds": rounds,
            "trace": self.event(
                state,
                "Multi-query RRF",
                f"Round {attempt + 1}: merged {len(queries)} queries with a second RRF into {len(context)} chunks "
                f"({added} not seen in the previous round).",
            ),
        }

    def generate(self, state: AgentState) -> Dict[str, Any]:
        options = state.get("options", [])
        if not state["use_rag"]:
            prompt = f"""
You are a telecom and AI assistant.

Answer the user's question directly using your learned knowledge.

No 3GPP retrieval was performed.

Do not pretend that you retrieved information from the
3GPP knowledge base.

Do not invent 3GPP citations.

If the user requires an exact 3GPP clause or standards-specific
answer, say that the 3GPP knowledge base should be consulted
rather than inventing a clause.

QUESTION:

{state['question']}

OPTIONS:
{options}

ANSWER:
"""
            return {
                "answer": self.ask_model(prompt),
                "cited_by_local_model": False,
                "trace": self.event(
                    state, "Generation", "Direct model answer generated without RAG. No 3GPP chunks were retrieved."
                ),
            }

        context = state.get("context", [])
        context_text = "\n\n".join(
            f"""
3GPP SPECIFICATION:
{item['spec']}

SECTION:
{item['section']} \u2014 {item['title']}

CONTENT:
{item['text']}
"""
            for item in context
        ) or "NO 3GPP CONTEXT WAS RETRIEVED."

        if state.get("retrieval_relevant"):
            evidence_instruction = "The retrieval critic judged the context sufficient."
        else:
            evidence_instruction = (
                "The retrieval critic could NOT confirm that the context is sufficient after the allowed "
                "retrieval attempts. Do NOT fill missing information from model memory. If the evidence is "
                'insufficient, explicitly say: "The retrieved 3GPP context is insufficient to answer reliably."'
            )

        prompt = f"""
You are a 3GPP telecom assistant.

{evidence_instruction}

Answer using ONLY the context below.

Question:
{state['question']}

Context:
{context_text}

Rules:
- Do not use outside knowledge or guess.
- If the context does not contain the answer, say:
"The retrieved 3GPP context is insufficient to answer reliably."
- Answer in 1-2 sentences.
- Give a direct answer, not an option letter.
- If supported, end with the exact citation from the context:
[TS XX.XXX, Section X.X.X]
- Never invent a citation.

Answer:
"""
        answer, cited = ensure_retrieved_citation(self.ask_model(prompt), context)
        return {
            "answer": answer,
            "cited_by_local_model": cited,
            "trace": self.event(
                state,
                "Generation",
                f"Grounded answer generated after {state.get('retrieval_attempt', 0)} retrieval round(s); "
                f"citation {'validated against a retrieved chunk' if cited else 'attached from the top retrieved chunk'}.",
            ),
        }

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("route", self.route)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("evaluate_retrieval", self.evaluate_retrieved_context)
        graph.add_node("prepare_retry", self.prepare_retry)
        graph.add_node("retrieve_retry", self.retrieve_retry_queries)
        graph.add_node("generate", self.generate)

        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            lambda s: "retrieve" if s["use_rag"] else "generate",
            {"retrieve": "retrieve", "generate": "generate"},
        )
        graph.add_edge("retrieve", "evaluate_retrieval")
        graph.add_conditional_edges(
            "evaluate_retrieval",
            self.after_retrieval_evaluation,
            {"generate": "generate", "prepare_retry": "prepare_retry"},
        )
        graph.add_edge("prepare_retry", "retrieve_retry")
        graph.add_edge("retrieve_retry", "evaluate_retrieval")
        graph.add_edge("generate", END)
        return graph.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str, options: Optional[List[str]] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        with self.lock:
            result = self.graph.invoke(self._initial_state(question, options))
        return self._finalise(question, options, result, started)

    @staticmethod
    def _initial_state(question: str, options: Optional[List[str]]) -> AgentState:
        return {
            "question": question,
            "options": list(options or []),
            "context": [],
            "answer": "",
            "cited_by_local_model": False,
            "rounds": [],
            "trace": [],
        }

    def _finalise(
        self, question: str, options: Optional[List[str]], result: AgentState, started: float
    ) -> Dict[str, Any]:
        context = result.get("context", [])
        answer = result.get("answer", "")
        citations = [f"[TS {s}, Section {sec}]" for s, sec in CITATION_PATTERN.findall(answer)]
        cited_by_model = bool(result.get("cited_by_local_model", False))
        accepted = bool(result.get("retrieval_relevant", False))
        use_rag = bool(result.get("use_rag", False))
        return {
            "question": question,
            "options": list(options or []),
            "answer": answer,
            "use_rag": use_rag,
            "route": "RAG" if use_rag else "DIRECT",
            "route_reason": result.get("route_reason", ""),
            "router_output": result.get("router_output", ""),
            "retrieval_attempts": result.get("retrieval_attempt", 0),
            "retry_count": result.get("retry_count", 0),
            "retrieval_relevant": accepted,
            "retrieval_score": round(float(result.get("retrieval_score", 0.0)), 4),
            "retrieval_queries": result.get("retrieval_queries", []),
            "expanded_queries": result.get("expanded_queries", []),
            "missing_information": result.get("missing_information", ""),
            "retry_decision": result.get("retry_decision", "skipped"),
            "cited_by_model": cited_by_model,
            "citation_source": "model" if cited_by_model else ("fallback" if citations else "none"),
            "citations": citations,
            "sources": self.summarise_chunks(context),
            "rounds": result.get("rounds", []),
            "trace": result.get("trace", []),
            "elapsed_sec": round(time.perf_counter() - started, 2),
            "quality_flags": self.quality_flags(answer, use_rag, accepted, citations, cited_by_model),
        }

    # ------------------------------------------------------------------
    # Streaming (same graph, same decisions, one event per completed node)
    # ------------------------------------------------------------------

    def stream_answer(self, question: str, options: Optional[List[str]] = None) -> Iterator[str]:
        """Yield SSE frames as each graph node finishes, then the final payload.

        This runs the identical compiled graph through stream() instead of
        invoke(). No prompt, threshold or branch differs, so a streamed run and
        a non-streamed run of the same question take the same path.
        """
        started = time.perf_counter()
        initial = self._initial_state(question, options)
        final = initial
        yield self._sse({"type": "start", "question": question})
        try:
            with self.lock:
                for chunk in self.graph.stream(initial, stream_mode=["updates", "values"]):
                    mode, data = self._split_stream_chunk(chunk)
                    if mode == "values":
                        final = data
                        continue
                    for node, delta in data.items():
                        yield self._sse(self._node_event(node, delta))
            result = final
        except Exception as exc:  # keep the browser transcript alive on failure
            yield self._sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            return
        yield self._sse({"type": "result", "result": self._finalise(question, options, result, started)})

    @staticmethod
    def _split_stream_chunk(chunk: Any) -> Tuple[str, Any]:
        """LangGraph yields (mode, payload) tuples; older builds yield the payload."""
        if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] in ("updates", "values"):
            return chunk[0], chunk[1]
        return "updates", chunk

    @staticmethod
    def _sse(payload: Dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, default=str)}\n\n"

    def _node_event(self, node: str, delta: Dict[str, Any]) -> Dict[str, Any]:
        """One finished node: its notebook name, its own words, and a status pill."""
        label = NODE_LABELS.get(node, node)
        entries = delta.get("trace") or []
        detail = ""
        if entries and entries[-1].get("node") == label:
            detail = entries[-1].get("detail", "")

        pill, tone = "", "mute"
        if node == "route":
            on = bool(delta.get("use_rag"))
            pill, tone = ("RAG ON", "ok") if on else ("RAG OFF", "mute")
        elif node in ("retrieve", "retrieve_retry"):
            pill, tone = f"{len(delta.get('context') or [])} CHUNKS", "mute"
        elif node == "evaluate_retrieval":
            ok = bool(delta.get("retrieval_relevant"))
            pill = f"{'SUFFICIENT' if ok else 'INSUFFICIENT'} {round(float(delta.get('retrieval_score') or 0.0) * 100)}%"
            tone = "ok" if ok else "warn"
        elif node == "prepare_retry":
            pill, tone = "RETRY", "warn"
        elif node == "generate":
            pill, tone = ("CITED" if delta.get("cited_by_local_model") else "ANSWERED"), "accent"

        return {
            "type": "node",
            "node": node,
            "label": label,
            "detail": detail,
            "pill": pill,
            "tone": tone,
            "use_rag": bool(delta.get("use_rag", False)),
            "retry_decision": delta.get("retry_decision", ""),
        }

    @staticmethod
    def quality_flags(answer: str, use_rag: bool, accepted: bool, citations: List[str], cited_by_model: bool) -> List[str]:
        """Observable facts about the answer that a developer should see, not hide.

        None of these change the agent's behaviour. They report what the notebook's
        own generation rules asked for but the fine-tuned model did not deliver.
        """
        flags = []
        if re.match(r"\s*option\s*[\da-zA-Z]+\s*[:.\)]", answer, re.I):
            flags.append("multiple_choice_prefix")
        if use_rag and not accepted and citations and not cited_by_model:
            flags.append("citation_attached_after_rejection")
        if use_rag and not accepted and not re.search(r"\binsufficient\b", answer, re.I):
            flags.append("answered_after_rejection_without_abstaining")
        return flags

    def dev_run(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Notebook sections 31/32/35/36 over a question set. RAGAS is excluded."""
        rows = []
        for case in cases:
            question = str(case.get("question", "")).strip()
            if not question:
                continue
            outcome = self.answer(question, case.get("options"))
            rows.append(
                {
                    "question": question,
                    "options": list(case.get("options") or []),
                    "ground_truth": case.get("ground_truth", ""),
                    "answer": outcome["answer"],
                    "use_rag": outcome["use_rag"],
                    "route": outcome["route"],
                    "route_reason": outcome["route_reason"],
                    "router_output": outcome["router_output"],
                    "retrieval_attempts": outcome["retrieval_attempts"],
                    "retry_count": outcome["retry_count"],
                    "retry_decision": outcome["retry_decision"],
                    "retrieval_relevant": outcome["retrieval_relevant"],
                    "retrieval_score": outcome["retrieval_score"],
                    "retrieval_queries": outcome["retrieval_queries"],
                    "expanded_queries": outcome["expanded_queries"],
                    "missing_information": outcome["missing_information"],
                    "cited_by_model": outcome["cited_by_model"],
                    "citation_source": outcome["citation_source"],
                    "citations": outcome["citations"],
                    "quality_flags": outcome["quality_flags"],
                    "retrieved_specs": [f"{s['spec']} section {s['section']}" for s in outcome["sources"]],
                    "elapsed_sec": outcome["elapsed_sec"],
                }
            )
        return {"results": rows, "summary": summarise_run(rows), "ragas": None}


def summarise_run(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Notebook sections 35 and 36."""
    total = len(rows)
    rag = [r for r in rows if r["use_rag"]]
    accepted_first = sum(1 for r in rag if r["retrieval_attempts"] == 1 and r["retrieval_relevant"])
    retried = sum(1 for r in rag if r["retry_count"] > 0)
    accepted_final = sum(1 for r in rag if r["retrieval_relevant"])
    cited = sum(1 for r in rag if r["cited_by_model"])
    pct = lambda n: round(n / len(rag), 4) if rag else 0.0
    return {
        "total_questions": total,
        "rag_questions": len(rag),
        "no_rag_questions": total - len(rag),
        "accepted_after_initial_retrieval": accepted_first,
        "accepted_after_initial_retrieval_rate": pct(accepted_first),
        "required_corrective_retrieval": retried,
        "required_corrective_retrieval_rate": pct(retried),
        "accepted_after_correction": accepted_final,
        "accepted_after_correction_rate": pct(accepted_final),
        "still_insufficient": len(rag) - accepted_final,
        "still_insufficient_rate": pct(len(rag) - accepted_final),
        "average_retrieval_score": round(sum(r["retrieval_score"] for r in rag) / len(rag), 4) if rag else 0.0,
        "verified_citations": cited,
        "verified_citations_rate": pct(cited),
    }


# ============================================================
# 6. THE NOTEBOOK'S TEST QUESTIONS
# ============================================================

TEST_QUESTIONS: List[Dict[str, Any]] = [
    {"question": "Which network function provides Network Data Analytics services in the 5G System?", "ground_truth": "NWDAF"},
    {"question": "Which interface is used by NWDAF to request subscription to data delivery from a source?", "ground_truth": "Nnf interface"},
    {"question": "Which network function is responsible for establishing and managing PDU sessions?", "ground_truth": "SMF"},
    {"question": "Which interface connects the UE and AMF for NAS signalling?", "ground_truth": "N1"},
    {"question": "Which interface is used between the (R)AN and UPF for user-plane traffic?", "ground_truth": "N3"},
    {"question": "Which procedure is used by a UE to register with the 5G System?", "ground_truth": "Registration procedure"},
    {"question": "What message does the UE send to initiate the 5GS registration procedure?", "ground_truth": "Registration Request"},
    {"question": "What is the purpose of the Service Request procedure in 5GS?", "ground_truth": "To request establishment or resumption of signalling or user-plane connectivity for a UE in relevant states"},
    {"question": "What procedure is used when a UE releases its registration from the 5G System?", "ground_truth": "Deregistration procedure"},
    {"question": "What protocol is specified by 3GPP TS 24.501?", "ground_truth": "Non-Access-Stratum (NAS) protocol for 5GS"},
    {"question": "What is the purpose of the SUCI in 5G security?", "ground_truth": "To conceal the permanent subscription identifier (SUPI)"},
    {"question": "Which authentication methods are specified for 5G System authentication?", "ground_truth": "5G-AKA and EAP-based authentication methods"},
    {"question": "What is the role of the SEAF during 5G authentication?", "ground_truth": "It serves as the security anchor function in the serving network"},
    {"question": "What does 3GPP TS 38.300 specify?", "ground_truth": "NR and NG-RAN overall description and Stage-2 aspects"},
    {"question": "Which NG-RAN node provides NR radio access to the UE?", "ground_truth": "gNB"},
    {"question": "Which interface connects a gNB to another gNB for inter-node communication?", "ground_truth": "Xn"},
    {"question": "What protocol is specified by 3GPP TS 38.331?", "ground_truth": "NR Radio Resource Control (RRC)"},
    {"question": "What are the main RRC states defined for NR?", "ground_truth": "RRC_IDLE, RRC_INACTIVE and RRC_CONNECTED"},
    {"question": "Which RRC message is used to establish an RRC connection?", "ground_truth": "RRCSetup"},
    {"question": "What is the purpose of RRC measurement reporting?", "ground_truth": "To provide the network with radio measurements used for mobility and radio-resource decisions"},
]

# ============================================================
# 7. HTTP API
# ============================================================


class Question(BaseModel):
    question: str
    options: Optional[List[str]] = None


class DevRun(BaseModel):
    questions: Optional[List[Question]] = None
    limit: int = 10


app = FastAPI(title="V Telecom Assistant")
agent: Optional[TelecomAgent] = None
load_error: Optional[str] = None


@app.on_event("startup")
def startup() -> None:
    global agent, load_error
    try:
        agent = TelecomAgent()
    except Exception as exc:  # surfaced to the browser instead of killing the process
        load_error = str(exc)
        raise


def require_agent() -> TelecomAgent:
    if agent is None:
        raise HTTPException(503, load_error or "The telecom agent is still loading.")
    return agent


@app.get("/")
def home() -> FileResponse:
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/api/health")
def health() -> dict:
    return {
        "ready": agent is not None,
        "error": load_error,
        "corpus": str(ROOT),
        "base_model": str(BASE_MODEL),
        "adapter": str(ADAPTER),
        "collection": COLLECTION,
        "loaded_at": agent.loaded_at if agent else None,
        "ragas": "excluded",
    }


@app.post("/api/ask")
def ask(payload: Question) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(422, "A question is required.")
    return require_agent().answer(question, payload.options)


@app.post("/api/ask/stream")
def ask_stream(payload: Question) -> StreamingResponse:
    """Same graph as /api/ask, emitted as SSE so each node is reported as it finishes."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(422, "A question is required.")
    running = require_agent()
    return StreamingResponse(
        running.stream_answer(question, payload.options),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/api/dev_run")
def dev_run(payload: DevRun) -> dict:
    """Notebook sections 31/32/35/36 over a question set. RAGAS is excluded."""
    running = require_agent()
    if payload.questions:
        cases = [{"question": q.question, "options": q.options, "ground_truth": ""} for q in payload.questions]
    else:
        cases = TEST_QUESTIONS[: max(1, min(payload.limit, len(TEST_QUESTIONS)))]
    return running.dev_run(cases)
