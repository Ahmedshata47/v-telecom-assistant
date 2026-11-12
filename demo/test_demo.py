"""Checks for the demo port of the voda.ipynb agent graph.

Offline checks (no API key) always run:

  * RAGAS is not imported anywhere in the server
  * the graph topology matches the notebook
  * citation validation matches the notebook
  * the critic branch conditions match after_retrieval_evaluation
  * real hybrid retrieval returns real 3GPP chunks from the local ChromaDB

End-to-end checks only run when MISTRAL_API_KEY is set, because they call the
Mistral retrieval critic and the local Qwen model.

    C:\\Users\\ASUS\\miniconda3\\envs\\tf\\python.exe test_demo.py
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import agent_server as srv  # noqa: E402

PASS, FAIL = [], []


def check(name):
    def wrap(fn):
        try:
            detail = fn() or ""
            PASS.append(name)
            print(f"  PASS  {name}" + (f" -- {detail}" if detail else ""))
        except Exception as exc:
            FAIL.append((name, exc))
            print(f"  FAIL  {name} -- {type(exc).__name__}: {exc}")
            traceback.print_exc()
        return fn
    return wrap


# ============================================================
# Offline checks
# ============================================================


@check("RAGAS is not imported or referenced by the server")
def _():
    source = (Path(srv.__file__)).read_text(encoding="utf-8")
    assert "from ragas" not in source and "import ragas" not in source, "ragas import found"
    assert "evaluate(" not in source, "ragas evaluate() call found"
    return "no ragas import, no ragas evaluate()"


@check("graph topology matches the notebook")
def _():
    agent = object.__new__(srv.TelecomAgent)  # no model load
    graph = agent.build_graph()
    nodes = set(graph.get_graph().nodes)
    expected_nodes = {
        "__start__", "route", "retrieve", "evaluate_retrieval",
        "prepare_retry", "retrieve_retry", "generate", "__end__",
    }
    assert nodes == expected_nodes, f"nodes differ: {nodes ^ expected_nodes}"
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    expected_edges = {
        ("__start__", "route"),
        ("route", "retrieve"), ("route", "generate"),
        ("retrieve", "evaluate_retrieval"),
        ("evaluate_retrieval", "generate"), ("evaluate_retrieval", "prepare_retry"),
        ("prepare_retry", "retrieve_retry"),
        ("retrieve_retry", "evaluate_retrieval"),
        ("generate", "__end__"),
    }
    assert edges == expected_edges, f"edges differ: {edges ^ expected_edges}"
    return f"{len(nodes)} nodes, {len(edges)} edges"


@check("retrieval config matches the notebook")
def _():
    assert (srv.TOP_K, srv.CANDIDATE_K, srv.BM25_CANDIDATE_K, srv.RRF_K) == (5, 40, 40, 60)
    assert srv.MAX_RETRIEVAL_ATTEMPTS == 2
    return "TOP_K=5 CANDIDATE_K=40 BM25=40 RRF_K=60 MAX_ATTEMPTS=2"


@check("embedding model and router prompt match the notebook")
def _():
    # The Chroma collection was built with this 768-d model in the notebook's
    # index cell; querying it with any other model raises a dimension mismatch.
    assert srv.EMBEDDING_MODEL == "sentence-transformers/multi-qa-mpnet-base-dot-v1", srv.EMBEDDING_MODEL
    # route_query() router prompt, verbatim: ask() wraps it as "Question:\n<text>".
    assert srv.router_prompt("Q") == (
        "\n    return 'NO' if you know the answer to the question: Q and 'YES' otherwise."
        "\n\n    return ONLY 'YES' or 'NO'.\n    Decision:\n    "
    ), repr(srv.router_prompt("Q"))
    import re as _re
    assert _re.search(r"\bYES\b", "YES") and not _re.search(r"\bYES\b", "NO")
    return "multi-qa-mpnet-base-dot-v1, notebook YES/NO router prompt"


@check("citation helpers match the notebook")
def _():
    context = [
        {"spec": "TS 23.501", "section": "4.3.1", "title": "T", "text": "a"},
        {"spec": "TS 38.331", "section": "5.3.1", "title": "T", "text": "b"},
    ]
    assert srv.citation(context[0]) == "[TS 23.501, Section 4.3.1]"
    assert srv.citation({"spec": "", "section": "1"}) is None

    kept, cited = srv.ensure_retrieved_citation("Yes. [TS 23.501, Section 4.3.1]", context)
    assert cited and kept.endswith("[TS 23.501, Section 4.3.1]"), kept

    hallucinated, cited = srv.ensure_retrieved_citation("Yes. [TS 99.999, Section 1.1]", context)
    assert not cited, "hallucinated citation must be rejected"
    assert "99.999" not in hallucinated, hallucinated
    assert hallucinated.endswith("[TS 23.501, Section 4.3.1]"), hallucinated

    # An abstention that also carries an unsupported citation is cleaned, and no
    # citation is attached to it. A *valid* citation wins first, so the abstention
    # branch is only reachable with an unsupported citation present.
    abstained, cited = srv.ensure_retrieved_citation(
        "The retrieved 3GPP context is insufficient to answer reliably. [TS 99.999, Section 1.1]", context
    )
    assert not cited and "Section" not in abstained, abstained
    assert "insufficient" in abstained.lower()

    # A valid citation always wins, even on an abstention, exactly as in the notebook.
    kept, cited = srv.ensure_retrieved_citation(
        "Insufficient context. [TS 38.331, Section 5.3.1]", context
    )
    assert cited and kept.endswith("[TS 38.331, Section 5.3.1]"), (cited, kept)
    return "validated / hallucination-rejected / abstention-clean"


@check("critic branch conditions match after_retrieval_evaluation")
def _():
    decide = srv.TelecomAgent.retry_decision
    assert decide({"retrieval_attempt": 1}, True, []) == "generate"
    assert decide({"retrieval_attempt": 1}, False, ["q"]) == "prepare_retry"
    assert decide({"retrieval_attempt": 2}, False, ["q"]) == "generate_max_attempts"
    assert decide({"retrieval_attempt": 1}, False, []) == "generate_no_new_query"
    edge = srv.TelecomAgent.after_retrieval_evaluation
    assert edge({"retry_decision": "prepare_retry"}) == "prepare_retry"
    assert edge({"retry_decision": "generate_max_attempts"}) == "generate"
    assert edge({}) == "generate"
    return "accept / retry / max-rounds / no-new-query all branch correctly"


@check("second RRF merge is stable and dedupes")
def _():
    agent = object.__new__(srv.TelecomAgent)
    a = {"spec": "TS 23.501", "section": "4.3.1", "text": "a", "hybrid_score": 0.1}
    b = {"spec": "TS 38.331", "section": "5.3.1", "text": "b", "hybrid_score": 0.1}
    c = {"spec": "TS 38.300", "section": "6.1.1", "text": "c", "hybrid_score": 0.1}
    merged = agent.merge_retrieval_results([[a, b], [b, c], [a]])
    by_section = {m["section"]: m for m in merged}
    assert by_section["4.3.1"]["query_hits"] == 2, by_section["4.3.1"]
    assert by_section["5.3.1"]["query_hits"] == 2, by_section["5.3.1"]
    assert by_section["6.1.1"]["query_hits"] == 1, by_section["6.1.1"]
    assert set(by_section) == {"4.3.1", "5.3.1", "6.1.1"}, by_section
    # a: ranks 1 and 1  -> 2/61 = 0.032787
    # b: ranks 2 and 1  -> 1/62 + 1/61 = 0.032522
    # c: rank 1         -> 1/61
    assert merged[0]["section"] == "4.3.1", merged[0]["section"]
    assert merged[-1]["section"] == "6.1.1", merged[-1]["section"]
    return "3 lists, 3 unique chunks, tie-break matches the notebook RRF"


@check("run summary counters match the notebook")
def _():
    rows = [
        {"use_rag": True, "retrieval_attempts": 1, "retrieval_relevant": True, "retry_count": 0, "retrieval_score": 0.9, "cited_by_model": True},
        {"use_rag": True, "retrieval_attempts": 2, "retrieval_relevant": True, "retry_count": 1, "retrieval_score": 0.7, "cited_by_model": False},
        {"use_rag": True, "retrieval_attempts": 2, "retrieval_relevant": False, "retry_count": 1, "retrieval_score": 0.2, "cited_by_model": False},
        {"use_rag": False, "retrieval_attempts": 0, "retrieval_relevant": False, "retry_count": 0, "retrieval_score": 0.0, "cited_by_model": False},
    ]
    s = srv.summarise_run(rows)
    assert s["total_questions"] == 4 and s["rag_questions"] == 3 and s["no_rag_questions"] == 1, s
    assert s["accepted_after_initial_retrieval"] == 1, s
    assert s["required_corrective_retrieval"] == 2, s
    assert s["accepted_after_correction"] == 2, s
    assert s["still_insufficient"] == 1, s
    assert s["verified_citations"] == 1, s
    assert abs(s["average_retrieval_score"] - 0.6) < 1e-6, s
    return "4 questions -> 1 ok, 2 retried, 1 still insufficient, 1 direct"


@check("real hybrid retrieval against the local ChromaDB")
def _():
    import chromadb
    from sentence_transformers import SentenceTransformer

    agent = object.__new__(srv.TelecomAgent)
    agent.collection = chromadb.PersistentClient(path=str(srv.ROOT / "chroma_db")).get_collection(srv.COLLECTION)
    all_docs = agent.collection.get(include=["documents", "metadatas"])
    agent.docs, agent.metadata = all_docs["documents"], all_docs["metadatas"]
    import re
    from rank_bm25 import BM25Okapi

    agent.bm25 = BM25Okapi([re.findall(r"\b\w+\b", d.lower()) for d in agent.docs])
    agent.embedding = SentenceTransformer(srv.EMBEDDING_MODEL)

    open_ended = agent.hybrid_retrieve("Which network function is responsible for establishing and managing PDU sessions?")
    assert len(open_ended) == srv.TOP_K, len(open_ended)
    assert any("SMF" in (c["text"] or "") or "SMF" in (c["title"] or "") for c in open_ended), open_ended
    assert all(c["hybrid_score"] > 0 for c in open_ended)

    ts_filtered = agent.hybrid_retrieve("What does 3GPP TS 38.331 specify?")
    assert ts_filtered, "TS-filtered query returned nothing"
    assert {c["spec"] for c in ts_filtered} == {"TS 38.331"}, {c["spec"] for c in ts_filtered}
    return f"{len(agent.docs)} chunks indexed, SMF question found SMF, TS 38.331 filter held"


# ============================================================
# End-to-end checks (need MISTRAL_API_KEY)
# ============================================================


def online_checks():
    if not os.getenv("MISTRAL_API_KEY"):
        print("\n  SKIP  end-to-end checks: MISTRAL_API_KEY is not set")
        return

    print("\nLoading the real agent (model + corpus + critic)...")

    @check("agent loads with the local QLoRA adapter and Chroma corpus")
    def _():
        srv.agent = srv.TelecomAgent()
        assert srv.agent.collection.count() > 0
        return f"{srv.agent.collection.count()} chunks, {os.path.basename(str(srv.BASE_MODEL))} + QLoRA"

    if srv.agent is None:
        return

    # Chosen because the router demonstrably sends them down the RAG branch. The
    # router's low recall is measured separately in the "router discriminates" check.
    rag_question = "What is the purpose of the Service Request procedure in 5GS?"
    ts_question = "What protocol is specified by 3GPP TS 24.501?"
    direct_question = "What is a full-mesh network?"
    # Verified to route to RAG and to make the critic reject twice, so the
    # corrective multi-query retry actually runs.
    retry_question = "What are the main RRC states defined for NR?"

    @check("end-to-end: RAG route reports rounds, chunks, critic verdict and citation")
    def _():
        result = srv.agent.answer(rag_question)
        assert result["route"] == "RAG", f"router sent this to {result['route']}: {result['router_output']!r}"
        assert result["retrieval_attempts"] == 1, result["retrieval_attempts"]
        assert len(result["rounds"]) == 1, len(result["rounds"])
        round_one = result["rounds"][0]
        assert round_one["kind"] == "hybrid" and round_one["chunk_count"] == srv.TOP_K
        assert round_one["critic"] is not None and "relevant" in round_one["critic"]
        assert round_one["decision"] in {"generate", "generate_max_attempts", "generate_no_new_query"}, round_one["decision"]
        assert result["sources"], "no sources returned"
        assert result["citations"], f"no citation on the answer: {result['answer']!r}"
        nodes = [t["node"] for t in result["trace"]]
        assert nodes[0] == "Router" and nodes[1] == "Hybrid retrieval" and nodes[2] == "Mistral critic"
        assert nodes[-1] == "Generation", nodes
        return (f"rounds={result['retrieval_attempts']} retry={result['retry_count']} "
                f"sufficiency={result['retrieval_score']:.2f} accepted={result['retrieval_relevant']} "
                f"cite={result['citations']} in {result['elapsed_sec']}s")

    @check("end-to-end: TS filter narrows retrieval to the named specification")
    def _():
        result = srv.agent.answer(ts_question)
        assert result["route"] == "RAG", f"router sent this to {result['route']}: {result['router_output']!r}"
        specs = {s["spec"] for s in result["sources"]}
        assert specs == {"TS 24.501"}, specs
        return f"{len(result['sources'])} chunks, all from TS 24.501"

    @check("end-to-end: DIRECT route returns no retrieval and no citation")
    def _():
        result = srv.agent.answer(direct_question)
        assert result["route"] == "DIRECT", result["route"]
        assert result["retrieval_attempts"] == 0 and result["retry_count"] == 0
        assert result["sources"] == [] and result["rounds"] == []
        assert result["cited_by_model"] is False and result["citations"] == []
        assert result["retry_decision"] == "skipped"
        nodes = [t["node"] for t in result["trace"]]
        assert nodes == ["Router", "Generation"], nodes
        return f"no retrieval, answer generated directly in {result['elapsed_sec']}s"

    @check("end-to-end: critic rejection triggers a corrective multi-query retry")
    def _():
        result = srv.agent.answer(retry_question)
        assert result["route"] == "RAG", f"router sent this to {result['route']}: {result['router_output']!r}"
        assert result["retry_count"] == 1, f"retry_count={result['retry_count']}"
        assert result["retrieval_attempts"] == srv.MAX_RETRIEVAL_ATTEMPTS, result["retrieval_attempts"]
        assert len(result["rounds"]) == 2, len(result["rounds"])

        first, second = result["rounds"]
        assert first["kind"] == "hybrid" and first["decision"] == "prepare_retry", first["decision"]
        assert first["critic"]["relevant"] is False and first["critic"]["search_queries"], first["critic"]
        assert first["critic"]["missing_information"], "critic gave no missing_information"

        assert second["kind"] == "multi_query_rrf", second["kind"]
        assert len(second["queries"]) == 1 + len(first["critic"]["search_queries"]), second["queries"]
        assert second["queries"][0] == retry_question, second["queries"][0]
        assert "new_chunks_vs_previous" in second, "retry did not report what changed"

        # Round 2 re-enters the critic; if it is still unhappy the graph must
        # stop at MAX_RETRIEVAL_ATTEMPTS rather than loop.
        assert second["decision"] in {"generate", "generate_max_attempts", "generate_no_new_query"}, second["decision"]
        nodes = [t["node"] for t in result["trace"]]
        assert nodes == [
            "Router", "Hybrid retrieval", "Mistral critic",
            "Corrective retrieval", "Multi-query RRF", "Mistral critic", "Generation",
        ], nodes
        return (f"round1 rejected ({first['critic']['score']:.0%}) -> retry with "
                f"{len(second['queries'])} queries -> {second['decision']}, "
                f"{second['new_chunks_vs_previous']} new chunks, flags={result['quality_flags']}")

    @check("end-to-end: quality flags expose what the model did not follow")
    def _():
        result = srv.agent.answer(retry_question)
        assert isinstance(result["quality_flags"], list), result["quality_flags"]
        assert result["citation_source"] in {"model", "fallback", "none"}
        if result["citations"] and not result["cited_by_model"]:
            assert result["citation_source"] == "fallback"
            if not result["retrieval_relevant"]:
                assert "citation_attached_after_rejection" in result["quality_flags"], result["quality_flags"]
        elif result["cited_by_model"]:
            assert result["citation_source"] == "model"
        return f"flags={result['quality_flags']} citation_source={result['citation_source']}"

    @check("end-to-end: router discriminates RAG from non-RAG questions")
    def _():
        import re

        should_rag = [
            "Which interface connects the UE and AMF for NAS signalling?",
            "What is the purpose of the Service Request procedure?",
            "What protocol is specified by 3GPP TS 24.501?",
            "What are the main RRC states defined for NR?",
            "Which RRC message is used to establish an RRC connection?",
            "Which network function is responsible for establishing and managing PDU sessions?",
            "What is the role of the AUSF in 5G authentication?",
            "Which interface is used between the (R)AN and UPF for user-plane traffic?",
        ]
        should_not = [
            "What is a full-mesh network?",
            "How does analog over-the-air computing improve privacy?",
            "Explain gradient descent in simple terms.",
            "What is the difference between TCP and UDP?",
            "Write a Python function that reverses a string.",
            "What is a bloom filter?",
            "Who wrote the novel Dune?",
            "What is the capital of Japan?",
        ]
        hit = lambda q: bool(re.search(r"\bYES\b", srv.agent.ask_model(srv.router_prompt(q), 16).upper()))
        tp = sum(1 for q in should_rag if hit(q))
        fn = len(should_rag) - tp
        fp = sum(1 for q in should_not if hit(q))
        tn = len(should_not) - fp
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        # The fine-tuned 1.5B router is imprecise in one direction only: it
        # under-uses RAG rather than over-using it. Guard that asymmetry so a
        # regression that floods every question into retrieval is caught.
        assert fp == 0, f"router sent {fp} non-RAG question(s) to retrieval"
        assert tp >= 3, f"router only found {tp}/8 RAG questions, retrieval is effectively dead"
        assert accuracy >= 0.6, f"router accuracy {accuracy:.0%} is too low to demo"
        return (f"accuracy {accuracy:.0%}, precision {(tp / (tp + fp) if tp + fp else 1):.0%}, "
                f"recall {tp / len(should_rag):.0%} -- {tp}/{len(should_rag)} RAG, {fp} false positives")

    @check("end-to-end: HTTP API exposes the same payload")
    def _():
        from fastapi.testclient import TestClient

        with TestClient(srv.app) as client:
            health = client.get("/api/health").json()
            assert health["ready"] is True and health["ragas"] == "excluded", health
            assert client.get("/").status_code == 200
            assert client.post("/api/ask", json={"question": "   "}).status_code == 422
            response = client.post("/api/ask", json={"question": rag_question})
            assert response.status_code == 200, response.text
            payload = response.json()
            for key in ("use_rag", "route", "rounds", "sources", "trace", "retry_count",
                        "retrieval_relevant", "missing_information", "expanded_queries", "citations",
                        "router_output", "retry_decision", "quality_flags", "citation_source"):
                assert key in payload, f"missing {key}"
            return f"health ok, ask ok, {len(payload)} keys, {len(payload['rounds'])} round(s)"


# ============================================================


print("=" * 68)
print("OFFLINE CHECKS")
print("=" * 68)
online_checks()
print("\n" + "=" * 68)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
print("=" * 68)
sys.exit(1 if FAIL else 0)
