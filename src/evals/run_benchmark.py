from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List
import sys

from dotenv import load_dotenv
from datasets import load_dataset
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

# Подключение локального клона агента: src/agents/open_deep_research/src
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ODR_SRC = os.path.join(REPO_ROOT, "src", "agents", "open_deep_research", "src")
if os.path.isdir(ODR_SRC) and ODR_SRC not in sys.path:
    sys.path.insert(0, ODR_SRC)

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.deep_researcher import deep_researcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Open Deep Research agent against a selected benchmark (SealQA)")
    parser.add_argument("--benchmark", type=str, default="sealqa", choices=["sealqa"], help="Название бенчмарка")
    parser.add_argument("--subset", type=str, default="seal_0", choices=["seal_0", "seal_hard", "longseal"], help="Поднабор SealQA")
    parser.add_argument("--split", type=str, default="test", help="Сплит датасета (по умолчанию test)")
    parser.add_argument("--limit", type=int, default=10, help="Ограничение числа задач для быстрого прогона")
    parser.add_argument("--out", type=str, default="reports/open_deep_research/sealqa", help="Каталог для сохранения отчётов")
    return parser.parse_args()


def build_config() -> Configuration:
    api_map = {
        "tavily": SearchAPI.TAVILY,
        "openai": SearchAPI.OPENAI,
        "anthropic": SearchAPI.ANTHROPIC,
        "none": SearchAPI.NONE,
    }
    search_api_str = os.environ.get("SEARCH_API", "tavily").lower()
    cfg = Configuration(search_api=api_map.get(search_api_str, SearchAPI.TAVILY))
    return cfg


async def run_one(question: str, cfg: Configuration) -> Dict[str, Any]:
    messages = [
        SystemMessage(content="You are a helpful research assistant that performs web research and cites sources."),
        HumanMessage(content=question),
    ]
    model_name = os.environ.get("RESEARCH_MODEL", "openai:gpt-4.1-mini")
    config = RunnableConfig(configurable={**cfg.model_dump(), "research_model": model_name, "allow_clarification": False})
    result = await deep_researcher.ainvoke({"messages": messages}, config=config)
    return result if isinstance(result, dict) else {"result": result}


async def amain() -> int:
    # Явная загрузка .env из корня репозитория
    env_path = os.path.join(REPO_ROOT, ".env")
    load_dotenv(env_path)
    args = parse_args()
    cfg = build_config()

    if args.benchmark == "sealqa":
        ds = load_dataset("vtllms/sealqa", name=args.subset, split=args.split)
    else:
        raise ValueError("Unsupported benchmark")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Пишем отчёты строго в корневой reports/, независимо от текущего CWD
    base_out = os.path.join(REPO_ROOT, args.out)
    out_dir = os.path.join(base_out, args.subset, args.split, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    limit = min(args.limit, len(ds)) if args.limit else len(ds)
    for i in range(limit):
        row = ds[i]
        question = row.get("question") or row.get("prompt") or str(row)
        res = await run_one(question=question, cfg=cfg)
        entry = {
            "id": i,
            "question": question,
            "raw": res,
        }
        results.append(entry)

    search_api_value = getattr(cfg.search_api, "value", cfg.search_api)

    model_name = os.environ.get("RESEARCH_MODEL", "openai:gpt-4.1-mini")
    report = {
        "benchmark": args.benchmark,
        "subset": args.subset,
        "split": args.split,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "search_api": search_api_value,
        "results": results,
    }
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"Saved results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))


