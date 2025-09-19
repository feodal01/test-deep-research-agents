#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic demo: schema-guided decision (search | answer) + Tavily search + grounded answer

Usage:
  export TAVILY_API_KEY="..."
  python agent_tavily_demo.py "Tell me three quirky facts about wolves."
  # Или без аргументов — будет интерактивный режим.

Requires:
  - ollama
  - pydantic>=2
  - tavily-python
  - python-dotenv (optional, for .env)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import List, Literal, Union

from pydantic import BaseModel, Field, ValidationError
from tavily import TavilyClient

import ollama


# =========================
# 1) Pydantic схемы
# =========================

class SearchRequest(BaseModel):
    reasoning: str = Field(..., max_length=100, description="Why search this (1 short sentence)")
    request: str = Field(..., max_length=100, description="Full search query for the web")

class Search(BaseModel):
    name: Literal['search']
    reasoning: str = Field(..., max_length=100, description="Why search this (1 short sentence)")
    request: List[SearchRequest] = Field(..., min_items=1, max_items=5, description="List of search queries to be executed")

class Answer(BaseModel):
    name: Literal['answer']
    reasoning: str = Field(..., max_length=100, description="Brief decision rationale based strictly on facts above")
    answer: str = Field(..., max_length=1000, description="Answer to the user's question")

class ModelResponse(BaseModel):
    model_response: Union[Search, Answer]


class ForcedSearchResponse(BaseModel):
    """Minimal shape to force a search-only plan."""
    model_response: Search


# =========================
# 2) System-промпты
# =========================

DECISION_SYS_TEMPLATE = """
You are a router. Your only task: decide between SEARCH or ANSWER for the user's question.

Tools:
- web_search: the only available tool.

Current date (UTC): {today}

Decision rule (pick exactly one):
- If the question needs up-to-date or verifiable info → SEARCH.
- If it is stable general knowledge → ANSWER.
- When uncertain → SEARCH.

Output requirements:
- OUTPUT ONLY VALID JSON per schema. No prose, no markdown.
- "reasoning": one short sentence.
- If name=="search": produce exactly 5 SearchRequest items. Queries must be distinct, concise, and general-purpose (no site: filters).

Schema (return EXACTLY one of the following via the top-level object):
{schema}

Valid examples (STRICT SHAPE):

EXAMPLE A (answer):
{{
  "model_response": {{
    "name": "answer",
    "reasoning": "This can be answered without external facts.",
    "answer": "Domestic cats are crepuscular and have a strong righting reflex..."
  }}
}}

EXAMPLE B (search):
{{
  "model_response": {{
    "name": "search",
    "request": [
      {{
        "reasoning": "Find reputable recent articles and summaries.",
        "request": "surprising facts about wolves 2023..2025"
      }},
      {{
        "reasoning": "Add behavioral research and scientific sources.",
        "request": "wolf pack dynamics peer-reviewed 2020..2025"
      }}
    ]
  }}
}}
""".strip()

ANSWER_SYS_TEMPLATE = """
Role:
You are a careful, grounded answerer. You MUST synthesize an answer to the user's question
STRICTLY from the provided search results. If the results are insufficient, say so explicitly.
Keep the reasoning concise and the final answer clear and specific.

Hard requirements:
- Base your answer ONLY on "SOURCES" content below.
- Be concise and concrete. Avoid filler text.
- If helpful, include a short bullet list.
- Optionally include a "Sources:" section with the most relevant URLs (3–6).
- OUTPUT ONLY VALID JSON matching the provided schema.

Time sensitivity:
- Consider today's date: {today}.
- If the question depends on the latest status (e.g., values that change, ongoing events), prioritize newer sources, compare publication dates, and resolve conflicts by recency and credibility.
- If recency is uncertain or sources may be outdated, clearly state this and qualify the answer with "As of {today}".
- When listing sources, prefer including source dates if available.
- If the fact/value depends on a reference date (e.g., age, duration since an event, seniority), explicitly compute or state the value relative to the correct date: use the user's specified reference date if provided, otherwise use {today}. When helpful, include both: the value as of the source's publication date and the updated value as of {today}.

Schema:
{schema}

Current date (UTC): {today}
""".strip()


SEARCH_PLAN_SYS_TEMPLATE = """
You MUST output a search plan JSON for the user's question. Answering directly is NOT allowed in this iteration.

Current date (UTC): {today}

Requirements (strict):
- Return ONLY the JSON matching the schema below.
- Top-level: `model_response.name` MUST be "search".
- Provide 1-sentence plan `reasoning`.
- Provide EXACTLY 5 SearchRequest items, each with short `reasoning` and a distinct concrete query.
- Queries must be general-purpose (no domain restrictions), and reflect freshness if relevant (include the year or {today}).

Schema (STRICT):
{schema}
""".strip()


# =========================
# 3) Инициализация клиентов
# =========================

def get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[ERROR] TAVILY_API_KEY is not set. Export it or put into .env", file=sys.stderr)
        sys.exit(1)
    return TavilyClient(api_key=api_key)

def call_ollama_chat(messages, model: str, format_schema: dict, temperature: float = 0.2, *, messages_collector: list[str] | None = None, step_name: str | None = None) -> str:
    # Логируем входящие сообщения
    if messages_collector is not None:
        prefix = f"[{step_name}] " if step_name else ""
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            messages_collector.append(f"{prefix}role={role} content={json.dumps(content, ensure_ascii=False)}")

    resp = ollama.chat(
        messages=messages,
        model=model,
        options={
            "temperature": temperature,
            "repeat_penalty": 1.7,
        },
        format=format_schema,
    )
    content = resp["message"]["content"]

    # Логируем ответ модели
    if messages_collector is not None:
        prefix = f"[{step_name}] " if step_name else ""
        messages_collector.append(f"{prefix}assistant content={json.dumps(content, ensure_ascii=False)}")

    return content


# =========================
# 4) Логика решения + поиск + финальный ответ
# =========================

def decide_search_or_answer(question: str, model: str = "gemma3:1b", *, messages_collector: list[str] | None = None) -> ModelResponse:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sys_prompt = (
        DECISION_SYS_TEMPLATE
        .replace("{schema}", json.dumps(ModelResponse.model_json_schema(), ensure_ascii=False))
        .replace("{today}", today)
    )
    content = call_ollama_chat(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ],
        model=model,
        format_schema=ModelResponse.model_json_schema(),
        temperature=0.1,
        messages_collector=messages_collector,
        step_name="decide_search_or_answer",
    )
    try:
        return ModelResponse.model_validate_json(content)
    except ValidationError as e:
        print("\n[ERROR] Decision JSON failed validation. Raw content:\n", content, file=sys.stderr)
        raise e


def generate_search_plan(question: str, model: str = "gemma3:1b", *, messages_collector: list[str] | None = None) -> ForcedSearchResponse:
    """Generate a forced search-only plan with exactly 5 queries."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sys_prompt = (
        SEARCH_PLAN_SYS_TEMPLATE
        .replace("{schema}", json.dumps(ForcedSearchResponse.model_json_schema(), ensure_ascii=False))
        .replace("{today}", today)
    )
    content = call_ollama_chat(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ],
        model=model,
        format_schema=ForcedSearchResponse.model_json_schema(),
        temperature=0.5,
        messages_collector=messages_collector,
        step_name="generate_search_plan",
    )
    try:
        return ForcedSearchResponse.model_validate_json(content)
    except ValidationError as e:
        print("\n[ERROR] Forced search plan JSON failed validation. Raw content:\n", content, file=sys.stderr)
        raise e


def tavily_search_many(queries: List[str], max_results_per_query: int = 3, search_depth: str = "advanced") -> List[dict]:
    """
    Выполняем Tavily по каждому запросу и агрегируем результаты.
    Возвращаем список объектов вида:
    { "query": str, "results": [ {"title":..., "url":..., "content":...}, ... ] }
    """
    client = get_tavily_client()
    aggregated = []
    for q in queries:
        print(f"[TAVILY] {q}")
        try:
            res = client.search(
                query=q,
                search_depth=search_depth,
                max_results=max_results_per_query,
                include_domains=None,
                exclude_domains=None,
                include_answer=False,
                include_images=False,
                include_image_descriptions=False,
            )
            # API возвращает: {"results": [{"title","url","content",...}, ...], ...}
            results = res.get("results", [])
        except Exception as e:
            print(f"[WARN] Tavily error for query: {q} → {e}", file=sys.stderr)
            results = []
        aggregated.append({"query": q, "results": results})
    
    print(f"[TAVILY] Done {len(aggregated)} queries")
    print(aggregated)
    return aggregated


def build_sources_context(aggregated: List[dict], top_k: int = 8) -> tuple[str, list[dict]]:
    """
    Готовим компактный контекст «источников» для LLM.
    Дедуп по URL, отбор top_k по порядку появления.
    Возвращает:
      - text_context: нумерованный список источников (для system+user)
      - sources_list: список словарей {index, title, url, content}
    """
    seen = set()
    flat = []
    for bucket in aggregated:
        for item in bucket["results"]:
            url = item.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            flat.append({
                "title": item.get("title") or "",
                "url": url,
                "content": item.get("content") or "",
            })
            if len(flat) >= top_k:
                break
        if len(flat) >= top_k:
            break

    # Формируем нумерованный блок
    lines = []
    sources_list = []
    for i, it in enumerate(flat, start=1):
        title = it["title"].strip()[:160]
        url = it["url"]
        content = (it["content"] or "").strip().replace("\n", " ")
        # Подрежем контент, чтобы не перегружать микро-модель
        snippet = content[:800]
        lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}\n")
        sources_list.append({"index": i, **it})

    text_context = "\n".join(lines)
    return text_context, sources_list


def answer_with_sources(question: str, sources_block: str, model: str = "gemma3:1b", *, messages_collector: list[str] | None = None) -> Answer:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sys_prompt = (
        ANSWER_SYS_TEMPLATE
        .replace("{schema}", json.dumps(Answer.model_json_schema(), ensure_ascii=False))
        .replace("{today}", today)
    )
    user_prompt = f"""USER_QUESTION:
    {question}

    SOURCES:
    {sources_block}

    Return valid JSON only
    """
    content = call_ollama_chat(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        format_schema=Answer.model_json_schema(),
        temperature=0.1,
        messages_collector=messages_collector,
        step_name="final_answer",
    )
    try:
        return Answer.model_validate_json(content)
    except ValidationError as e:
        print("\n[ERROR] Final answer JSON failed validation. Raw content:\n", content, file=sys.stderr)
        raise e


# =========================
# 5) Benchmark adapter (non-printing)
# =========================

def gather_sources(
    question: str,
    model: str,
    *,
    messages_collector: list[str] | None = None,
    extra_rounds: int = 1,
) -> tuple[str, list[dict], dict]:
    """Формируем список источников с возможными доп.раундами поиска.

    Возвращает (sources_block, sources_list, debug), где debug содержит
    исходный принудительный план и решения доп.раундов.
    """
    search_plan = generate_search_plan(question, model=model, messages_collector=messages_collector)

    # Первый раунд поиска
    queries = [r.request for r in search_plan.model_response.request]
    aggregated_all: list[dict] = tavily_search_many(queries, max_results_per_query=5, search_depth="advanced")

    extra_decisions: list[ModelResponse] = []
    for _ in range(max(0, extra_rounds)):
        decision = decide_search_or_answer(question, model=model, messages_collector=messages_collector)
        mr = decision.model_response
        # Если модель готова отвечать — прекращаем дополнительные поиски
        if isinstance(mr, Answer) or getattr(mr, "name", None) == "answer":
            extra_decisions.append(decision)
            break
        # Иначе выполняем дополнительный поиск
        extra_decisions.append(decision)
        more_queries = [r.request for r in mr.request]
        more_agg = tavily_search_many(more_queries, max_results_per_query=5, search_depth="advanced")
        aggregated_all.extend(more_agg)

    sources_block, sources_list = build_sources_context(aggregated_all, top_k=8)
    debug = {"search_plan": search_plan, "decisions": extra_decisions}
    return sources_block, sources_list, debug


# =========================
# 6) Benchmark adapter (non-printing)
# =========================

def run_small_agent_for_benchmark(question: str, model: str | None = None) -> dict:
    """Запуск логики small-агента для использования в бенчмарке.

    Возвращает словарь совместимый с форматом vendored агента:
      { "messages": [..все промпты и ответы..], "final_report": "..." }

    Никаких print и побочных эффектов.
    """
    model_name = model or os.getenv("OLLAMA_MODEL", "gemma3:1b")
    raw_messages: list[str] = []

    extra_rounds = int(os.getenv("SSA_EXTRA_SEARCH_ROUNDS", "1"))
    sources_block, _sources_list, _debug = gather_sources(
        question, model_name, messages_collector=raw_messages, extra_rounds=extra_rounds
    )
    final = answer_with_sources(question, sources_block, model=model_name, messages_collector=raw_messages)

    return {
        "messages": raw_messages,
        "final_report": final.answer,
    }


# =========================
# 7) CLI
# =========================

def run_once(question: str, model: str = "gemma3:1b") -> None:
    print(f"\n=== USER: {question}")

    raw_messages: list[str] = []
    extra_rounds = int(os.getenv("SSA_EXTRA_SEARCH_ROUNDS", "1"))
    sources_block, sources_list, debug = gather_sources(
        question, model, messages_collector=raw_messages, extra_rounds=extra_rounds
    )

    # Печать плана и решений
    print("\n--- FORCED SEARCH PLAN ---")
    print(debug["search_plan"].model_dump_json(indent=2))
    for d in debug["decisions"]:
        mr = d.model_response
        if isinstance(mr, Answer) or getattr(mr, "name", None) == "answer":
            print("\n--- SECOND DECISION: ANSWER ---")
        else:
            print("\n--- SECOND DECISION: SEARCH ---")
            for i, req in enumerate(mr.request, 1):
                print(f"  [+{i}] {req.reasoning} -> {req.request}")
    print("\n--- SOURCES (compact) ---")
    print(sources_block)

    # Финальный ответ
    final = answer_with_sources(question, sources_block, model=model, messages_collector=raw_messages)

    print("\n--- GROUNDED ANSWER (JSON) ---")
    print(final.model_dump_json(indent=2))

    print("\nFinal answer:\n")
    print(final.answer)

    # Совместимый с ODR формат результата для последующей агрегации в results.jsonl
    result_entry = {
        "id": 0,
        "question": question,
        "raw": {
            "messages": raw_messages,
            "final_report": final.answer,
        },
    }
    print("\n--- RESULT JSONL LINE ---")
    print(json.dumps(result_entry, ensure_ascii=False))


def main():
    try:
        from dotenv import load_dotenv  # optional
        load_dotenv()
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Schema-guided (search|answer) with Tavily grounding")
    parser.add_argument("question", nargs="*", help="User question")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma3:1b"), help="Ollama model name")
    args = parser.parse_args()

    if args.question:
        question = " ".join(args.question).strip()
        run_once(question, model=args.model)
        return

    # Interactive
    print("Interactive mode. Ctrl+C to exit.")
    while True:
        try:
            q = input("\n> ") 
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q.strip():
            continue
        run_once(q.strip(), model=args.model)


if __name__ == "__main__":
    main()