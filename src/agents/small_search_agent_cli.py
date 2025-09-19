from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv

# Local import of vendored small agent
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SSA_DIR = os.path.join(REPO_ROOT, "src", "agents", "small_search_agent")
if SSA_DIR not in sys.path:
    sys.path.insert(0, SSA_DIR)

from main import run_query  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run small-search-agent on a single question")
    p.add_argument("question", type=str)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main() -> int:
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    args = parse_args()
    answer = run_query(args.question)
    out: Dict[str, Any] = {
        "question": args.question,
        "answer": answer,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


