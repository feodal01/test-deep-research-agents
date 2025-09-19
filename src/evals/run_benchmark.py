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

# Подключение локального клона агента: src/agents/open_deep_research/src
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ODR_SRC = os.path.join(REPO_ROOT, "src", "agents", "open_deep_research", "src")
if os.path.isdir(ODR_SRC) and ODR_SRC not in sys.path:
    sys.path.insert(0, ODR_SRC)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Open Deep Research agent against a selected benchmark (SealQA)")
    parser.add_argument("--benchmark", type=str, default="sealqa", choices=["sealqa"], help="Название бенчмарка")
    parser.add_argument("--subset", type=str, default="seal_0", choices=["seal_0", "seal_hard", "longseal"], help="Поднабор SealQA")
    parser.add_argument("--split", type=str, default="test", help="Сплит датасета (по умолчанию test)")
    parser.add_argument("--limit", type=int, default=10, help="Ограничение числа задач для быстрого прогона")
    parser.add_argument("--out", type=str, default=None, help="Каталог для сохранения отчётов (по умолчанию зависит от агента)")
    parser.add_argument("--agent", type=str, default="odr", choices=["odr", "small"], help="Какого агента запускать: odr (vendored) или small")
    return parser.parse_args()


def build_config():
    # Lazy import to avoid requiring ODR deps when using --agent small
    from open_deep_research.configuration import Configuration, SearchAPI
    api_map = {
        "tavily": SearchAPI.TAVILY,
        "openai": SearchAPI.OPENAI,
        "anthropic": SearchAPI.ANTHROPIC,
        "none": SearchAPI.NONE,
    }
    search_api_str = os.environ.get("SEARCH_API", "tavily").lower()
    cfg = Configuration(search_api=api_map.get(search_api_str, SearchAPI.TAVILY))
    return cfg


async def run_one(question: str, cfg) -> Dict[str, Any]:
    from open_deep_research.deep_researcher import deep_researcher
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.runnables import RunnableConfig
    messages = [
        SystemMessage(content="You are a helpful research assistant that performs web research and cites sources."),
        HumanMessage(content=question),
    ]
    model_name = os.environ.get("RESEARCH_MODEL", "openai:gpt-4.1-mini")
    config = RunnableConfig(configurable={**cfg.model_dump(), "research_model": model_name, "allow_clarification": False})
    result = await deep_researcher.ainvoke({"messages": messages}, config=config)
    return result if isinstance(result, dict) else {"result": result}


def run_one_small(question: str) -> Dict[str, Any]:
    """Синхронный прогон small-агента с возвратом {messages, final_report}.

    Без изменения логики small-агента: импортируем функции и временно
    перехватываем его call_ollama_chat, чтобы собрать все сообщения в/из LLM.
    """
    # Ленивая загрузка small-агента
    # Попробуем прямой файловый импорт по абсолютному пути репозитория
    repo_root = REPO_ROOT
    sys.path.insert(0, repo_root)
    try:
        from src.agents.small_search_agent import main as sa  # type: ignore
    except Exception:
        import importlib.util
        module_path = os.path.join(repo_root, "src", "agents", "small_search_agent", "main.py")
        spec = importlib.util.spec_from_file_location("small_search_agent_main", module_path)
        if not spec or not spec.loader:
            raise
        sa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sa)  # type: ignore

    # Используем встроенный адаптер small-агента, который уже собирает messages и final_report
    return sa.run_small_agent_for_benchmark(question)


async def amain() -> int:
    # Явная загрузка .env из корня репозитория
    env_path = os.path.join(REPO_ROOT, ".env")
    load_dotenv(env_path)
    args = parse_args()
    cfg = build_config() if args.agent == "odr" else None

    if args.benchmark == "sealqa":
        ds = load_dataset("vtllms/sealqa", name=args.subset, split=args.split)
    else:
        raise ValueError("Unsupported benchmark")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Пишем отчёты строго в корневой reports/, независимо от текущего CWD
    default_out = "reports/open_deep_research/sealqa" if args.agent == "odr" else "reports/small_search_agent/sealqa"
    base_out = os.path.join(REPO_ROOT, args.out or default_out)
    out_dir = os.path.join(base_out, args.subset, args.split, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    partial_jsonl_path = os.path.join(out_dir, "results.jsonl")
    progress_path = os.path.join(out_dir, "progress.json")

    results: List[Dict[str, Any]] = []
    total = len(ds)
    limit = min(args.limit, total) if args.limit else total
    print(f"Running SealQA subset={args.subset} split={args.split}: total={total}, limit={limit}", flush=True)
    for i in range(limit):
        row = ds[i]
        question = row.get("question") or row.get("prompt") or str(row)
        if args.agent == "odr":
            assert cfg is not None
            res = await run_one(question=question, cfg=cfg)
        else:
            res = run_one_small(question=question)
        entry = {
            "id": i,
            "question": question,
            "raw": res,
        }
        results.append(entry)
        # progress line
        print(f"[{i+1}/{limit}] done id={i}", flush=True)
        # промежуточное сохранение (append JSONL)
        try:
            with open(partial_jsonl_path, "a", encoding="utf-8") as pf:
                pf.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                pf.flush()
                try:
                    os.fsync(pf.fileno())
                except Exception:
                    pass
        except Exception:
            pass
        # короткий прогресс
        try:
            with open(progress_path, "w", encoding="utf-8") as pf:
                pf.write(json.dumps({
                    "subset": args.subset,
                    "split": args.split,
                    "total": total,
                    "completed": i+1,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False, indent=2))
        except Exception:
            pass

    if args.agent == "odr" and cfg is not None:
        search_api_value = getattr(cfg.search_api, "value", cfg.search_api)
        model_name = os.environ.get("RESEARCH_MODEL", "openai:gpt-4.1-mini")
    else:
        # small agent
        search_api_value = os.environ.get("SEARCH_API", "tavily")
        model_name = os.environ.get("SSA_OLLAMA_MODEL", os.environ.get("OLLAMA_MODEL", "gemma3:1b"))
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
