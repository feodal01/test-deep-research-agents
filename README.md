## test-deep-research-agents

A sandbox to evaluate open‑source deep‑research agents on open benchmarks. The first target benchmark is SealQA (seal-0). The focus is web research: searching the internet, gathering evidence, and producing grounded answers. This repo stores run rules, configs, artifacts, and reports.

### Why
- **Comparability**: single run protocol and unified report format across agents
- **Reproducibility**: pinned env, seeds, and artifacts
- **Simplicity**: quick start via `uv` and minimal run steps

### Important
Do NOT clone the upstream `langchain-ai/open_deep_research` separately. The agent is vendored in this repo at `src/agents/open_deep_research` and includes modifications that enable per‑node model overrides (one place to configure model/base_url/api_key for all nodes: research, summarization, compression, final_report). Upstream had different models hard‑coded per node; our version lets you set them uniformly via `.env` or runtime config.

There is an `env.example` at the repo root. Copy it to `.env` and fill in your keys.

### Requirements
- macOS/Linux (arm64/x86_64)
- Python 3.11+
- `uv` for env and dependency management

### Install uv and root deps
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
uv venv .venv
source .venv/bin/activate
uv sync
```

### Download SealQA dataset
Dataset: https://huggingface.co/datasets/vtllms/sealqa

This downloads all subsets (`seal_0`, `seal_hard`, `longseal`) and available splits into `data/`:
```bash
uv run python src/benchmarks/download_sealqa.py --out data --format parquet
```

### Environment variables (.env)
We auto‑load repo_root/.env in all scripts. See `env.example` and copy it to `.env`.

Key settings:
```text
SEARCH_API=tavily                     # tavily|openai|anthropic|none
TAVILY_API_KEY=...
# optional: TAVILY_API_BASE_URL=https://api.tavily.com

# Per-node overrides (model / base_url / api_key)
RESEARCH_MODEL=openai:gpt-4o-mini
RESEARCH_BASE_URL=
RESEARCH_API_KEY=...
SUMMARIZATION_MODEL=...
SUMMARIZATION_BASE_URL=...
SUMMARIZATION_API_KEY=...
COMPRESSION_MODEL=...
COMPRESSION_BASE_URL=...
COMPRESSION_API_KEY=...
FINAL_REPORT_MODEL=...
FINAL_REPORT_BASE_URL=...
FINAL_REPORT_API_KEY=...

# Behavior
ALLOW_CLARIFICATION=false

# Optional: read from runtime config instead of .env
GET_MODEL_OVERRIDES_FROM_CONFIG=false
GET_API_KEYS_FROM_CONFIG=false

# Provider fallbacks (used if per-node keys are not set)
# OPENAI_API_KEY=...
# ANTHROPIC_API_KEY=...
# GOOGLE_API_KEY=...

# Small Search Agent (Ollama) — only for src/agents/small_search_agent
SSA_OLLAMA_BASE_URL=http://localhost:11434
SSA_OLLAMA_MODEL=llama3.1:8b-instruct-q4_0
```

### Quick agent run (vendored agent)
Use the vendored agent and its environment (it has its own deps):
```bash
cd src/agents/open_deep_research
uv sync
uv run python ../open_deep_research_cli.py "Who authored SealQA and what is it?"
```
This uses model/search settings from repo_root/.env and writes nothing by default.

### Run the benchmark (SealQA)
We recommend running the benchmark using the agent’s environment, while the script lives in the root:
```bash
# from the agent env
cd src/agents/open_deep_research
uv run python ../../evals/run_benchmark.py --subset seal_0 --limit 5
```
Reports are saved under repo_root:
```
reports/open_deep_research/sealqa/<subset>/<split>/<timestamp>/run.json
```

### Run the benchmark with the Small Search Agent
Small agent has its own uv environment and writes identically structured results (raw.messages and raw.final_report):
```bash
# from the small agent env
cd src/agents/small_search_agent
uv sync   # first time only
uv run python ../../evals/run_benchmark.py --agent small --subset seal_0 --limit 5
```
Outputs for the small agent are written to:
```
reports/small_search_agent/sealqa/<subset>/<split>/<timestamp>/{run.json,results.jsonl,progress.json}
```

Optional anti-loop tuning for small agent (env):
```bash
# number of extra search rounds after the first forced one
SSA_EXTRA_SEARCH_ROUNDS=1
# hard wall-clock limit (seconds) for source gathering
SSA_MAX_SECONDS=120
# cap on total Tavily queries across all rounds
SSA_MAX_TOTAL_QUERIES=12
# stop if next-round queries are all duplicates of previous ones (true|false)
SSA_BREAK_ON_DUPLICATE=true
```

### Tavily connectivity test
```bash
cd src/agents/open_deep_research
uv run python ../../src/agents/scripts/test_tavily.py --query "SealQA benchmark" --max-results 3
```
This verifies your `TAVILY_API_KEY` (and `TAVILY_API_BASE_URL` if set). Our code supports older `tavily-python` versions by falling back when `api_base_url` is not available.

### Contributing
PRs welcome. Please include a short description of your settings and versions with results.

### License
MIT — see `LICENSE`.
