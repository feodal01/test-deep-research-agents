## test-deep-research-agents

Песочница для тестирования и сравнения open‑source deep‑research агентов на открытых бенчмарках. Первый целевой бенчмарк — `seal-0`. Фокус — агентный поиск и сбор фактов в интернете (web research). Код писать не требуется: репозиторий хранит правила запусков, конфигурации, артефакты результатов и отчёты.

### Зачем это нужно
- **Сравнимость**: единый протокол прогонов и формат отчётов для разных агентов.
- **Воспроизводимость**: фиксированные версии окружения, сиды и артефакты.
- **Простота**: быстрый старт через `uv` и понятные шаги запуска.

### Первый бенчмарк: sealqa
`sealqa` — открытый бенчмарк для оценки способности агента находить и агрегировать релевантные сведения из интернета. Мы используем его как стартовую точку для валидного сравнения open‑source deep‑research агентов. Детали набора, метрик и правил фиксируются в документации и конфигурациях в этом репозитории.

### Область проекта
- В фокусе: интернет‑поиск, навигация по источникам, сбор и верификация фактов.
- Агентов мы запускаем как внешние инструменты по их документации; в репозитории храним конфиги, протоколы и результаты.

### Требования
- macOS/Linux (arm64/x86_64)
- Python 3.11+ (предложение)
- `uv` — менеджер окружения и зависимостей

### Быстрый старт (uv)
1) Установите `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

2) Создайте изолированное окружение и активируйте его.

```bash
uv venv .venv
source .venv/bin/activate  # macOS/Linux
python -V
```

3) Зависимости. На старте код в репозитории не требуется; при добавлении вспомогательных утилит будет опубликован `pyproject.toml`/`uv.lock`, и вы сможете выполнить:

```bash
uv sync  # установит из pyproject.toml и uv.lock
```

### Загрузка датасета SealQA
Датасет: [vtllms/sealqa на Hugging Face](https://huggingface.co/datasets/vtllms/sealqa).

Скрипт загрузит все подсеты (`seal_0`, `seal_hard`, `longseal`) и все доступные сплиты в `data/`.

```bash
uv run python src/benchmarks/download_sealqa.py --out data --format parquet
```

Структура вывода:

```text
data/sealqa/
  ├── seal_0/
  │   ├── test.parquet
  │   └── metadata.json
  ├── seal_hard/
  │   ├── test.parquet
  │   └── metadata.json
  └── longseal/
      ├── test.parquet
      └── metadata.json
```

### Запуск агента (Open Deep Research)
Агент: [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research).

Клонирование в локальную папку (для редактирования конфигурации агента):

```bash
uv run python scripts/clone_open_deep_research.py
```

При необходимости отредактируйте конфигурацию агента в его репозитории (например, файл `open_deep_research/configuration.py`), затем используйте наш CLI.

Пример одиночного запроса:

```bash
uv run python src/agents/open_deep_research_cli.py "Кто автор SealQA и что это такое?" --search-api tavily --model openai:gpt-4.1-mini
```

Требуются валидные ключи API для выбранного провайдера (например, `OPENAI_API_KEY`, `TAVILY_API_KEY`).

Переменные окружения:

```text
# Эти ключи относятся к deep‑research агенту Open Deep Research.
# Другие агенты могут потребовать свои ключи — добавляйте их аналогично.
OPENAI_API_KEY=...       # если используете openai:* модели
ANTHROPIC_API_KEY=...    # если используете anthropic:* модели
OPENROUTER_API_KEY=...   # опционально, если используете OpenRouter
TAVILY_API_KEY=...       # провайдер поиска Tavily (рекомендуется по умолчанию)
GOOGLE_API_KEY=...       # опционально, если используете Google CSE
GOOGLE_CSE_ID=...        # опционально, если используете Google CSE
```

Можно создать файл `env.example`, скопировать его в `.env` и заполнить значениями.

### Прогон на бенчмарке SealQA
Быстрый прогон на первых N примеров:

```bash
uv run python src/evals/run_benchmark.py --benchmark sealqa --subset seal_0 --split test --limit 5
```

Результаты сохраняются в `reports/open_deep_research/sealqa/<subset>/<split>/<timestamp>/run.json`.

### Вклад
PR и предложения приветствуются. Просьба сопровождать результаты кратким описанием настроек и версий.

### Лицензия
MIT — см. `LICENSE`.
