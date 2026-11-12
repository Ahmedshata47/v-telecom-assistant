param(
  [Parameter(Mandatory=$true)] [string]$MistralApiKey,
  [int]$Port = 8000
)

$env:MISTRAL_API_KEY = $MistralApiKey
$tfPython = "C:\Users\ASUS\miniconda3\envs\tf\python.exe"
if (-not (Test-Path $tfPython)) {
  throw "The tf Conda environment was not found at $tfPython"
}
Set-Location $PSScriptRoot
& $tfPython -m uvicorn agent_server:app --host 127.0.0.1 --port $Port
