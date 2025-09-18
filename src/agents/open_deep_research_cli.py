from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
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
    parser = argparse.ArgumentParser(description="Run Open Deep Research agent on a single question")
    parser.add_argument("question", type=str, help="Вопрос для исследования")
    parser.add_argument("--timeout", type=int, default=600, help="Таймаут в секундах")
    parser.add_argument("--max-steps", type=int, default=16, help="Ограничение шагов агента (если поддерживается)")
    parser.add_argument("--output", type=str, default=None, help="Путь для сохранения JSON результата")
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


async def amain() -> int:
    # Явная загрузка .env из корня репозитория
    env_path = os.path.join(REPO_ROOT, ".env")
    load_dotenv(env_path)
    args = parse_args()
    cfg = build_config()

    messages = [
        SystemMessage(content="You are a helpful research assistant that performs web research and cites sources."),
        HumanMessage(content=args.question),
    ]

    # Модель берём из .env (пер-нодовый оверрайд RESEARCH_MODEL)
    model_name = os.environ.get("RESEARCH_MODEL", "openai:gpt-4.1-mini")
    # Передаём выбранную модель через configurable, т.к. в Configuration нет поля model
    config = RunnableConfig(configurable={**cfg.model_dump(), "research_model": model_name, "allow_clarification": False})
    result = await deep_researcher.ainvoke({"messages": messages}, config=config)

    # result — состояние графа; постараемся извлечь последний полезный ответ
    search_api_value = getattr(cfg.search_api, "value", cfg.search_api)

    output: Dict[str, Any] = {
        "question": args.question,
        "result": result,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "search_api": search_api_value,
    }
    text = None
    try:
        # многие реализации возвращают список сообщений; попробуем нащупать текст
        maybe_msgs = result.get("messages") if isinstance(result, dict) else None
        if isinstance(maybe_msgs, list) and maybe_msgs:
            for msg in reversed(maybe_msgs):
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    text = msg.content
                    break
    except Exception:
        pass

    if text:
        print(text)
    else:
        # запасной вывод
        print(json.dumps(output, default=str, ensure_ascii=False, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))


