from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from tavily import AsyncTavilyClient


async def amain() -> int:
    parser = argparse.ArgumentParser(description="Tavily connectivity test")
    parser.add_argument("--query", default="What is SealQA?", help="Search query")
    parser.add_argument("--max-results", type=int, default=3, help="Max results to fetch")
    args = parser.parse_args()

    # Load .env from repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    load_dotenv(os.path.join(repo_root, ".env"))

    api_key = os.getenv("TAVILY_API_KEY")
    base_url = os.getenv("TAVILY_API_BASE_URL") or os.getenv("TAVILY_BASE_URL")
    if not api_key:
        print("TAVILY_API_KEY is not set")
        return 2

    try:
        client = AsyncTavilyClient(api_key=api_key, api_base_url=base_url)
    except TypeError:
        client = AsyncTavilyClient(api_key=api_key)
    try:
        res = await client.search(
            args.query,
            max_results=args.max_results,
            include_raw_content=False,
            topic="general",
        )
    except Exception as e:
        print(f"Tavily request failed: {e}")
        return 1

    items = res.get("results", []) if isinstance(res, dict) else res
    print(f"OK: got {len(items)} result(s)")
    for i, it in enumerate(items[: args.max_results], 1):
        title = (it.get("title") or "").strip()
        url = it.get("url") or ""
        print(f"{i}. {title} — {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))


