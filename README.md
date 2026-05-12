# BTC Price Direction Prediction — Multiagent System

LangGraph DAG of LLM-powered agents (Twitter, tech indicators, news, on-chain, economic calendar) that vote on LONG/SHORT for a requested date. Served by a FastAPI app on port **8080**.

> **Port: `8080`** everywhere — `Dockerfile`, `docker-compose.yml`, local dev. All examples and curl snippets in this README hit `http://localhost:8080`.

---

## Table of Contents

1. [Quickstart (Docker)](#1-quickstart-docker)
2. [Local dev install](#2-local-dev-install)
3. [Environment variables](#3-environment-variables)
4. [API reference](#4-api-reference)
5. [Module CLI cheatsheet](#5-module-cli-cheatsheet)
6. [Multiagent config](#6-multiagent-config)
7. [Multiagent system internals](#7-multiagent-system-internals)
8. [Project structure](#8-project-structure)
9. [Troubleshooting](#9-troubleshooting)
10. [Multiagent tests](#10-multiagent-tests)

---

## 1. Quickstart (Docker)

The image (`danilacrazy1337/multiagent_system:v1.0`) is built from this repo's [Dockerfile](Dockerfile) and pinned in [docker-compose.yml](docker-compose.yml). The fastest path to a running API — from a folder containing `dev.env` and the four mounted SQLite/profile files:

```bash
docker compose up -d
```

The container exposes the API on **`http://localhost:8080`**. Verify:

```bash
# Health
curl http://localhost:8080/api/health

# Swagger UI
open http://localhost:8080/docs
```

Required side files (mounted into the container by `docker-compose.yml`):

| Path on host | Mounted to | Used by |
|---|---|---|
| `dev.env` | `/app/dev.env` | API key + secrets |
| `news_archive.db` | `…/news_analyser/news_archive.db` | News agent |
| `calendar_archive.db` | `…/economic_calendar_analyser/calendar_archive.db` | Calendar agent |
| `twitter_archive.db` | `…/twitter_scrapper/twitter_archive.db` | Twitter agent |
| `chrome_profile/` | `…/twitter_scrapper/chrome_profile` | Twitter login session |

> **`twitter_cookies.json` is baked into the image** (see [.dockerignore](.dockerignore) — the file is intentionally **not** ignored). When the session expires, refresh cookies at runtime via `POST /api/agents/twitter-upload-cookies` (needs `TWITTER_UPLOAD_KEY`) without rebuilding.

---

## 2. Local dev install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

Run the API on port **8080**:

```bash
# Production-shaped (matches Dockerfile)
uvicorn api.main:app --host 0.0.0.0 --port 8080

# Dev (auto-reload on code change)
uvicorn api.main:app --reload --port 8080
```

> Always pass `--port 8080` explicitly — uvicorn's bare default (8000) is **not** what this project uses.

After launch:

- Swagger UI: <http://localhost:8080/docs>
- ReDoc: <http://localhost:8080/redoc>

---

## 3. Environment variables

Loaded from `dev.env` at the project root via `python-dotenv` ([api/main.py:33](api/main.py#L33)).

| Variable | Required | Where used | What breaks if missing |
|---|:---:|---|---|
| `COINGLASS_API_KEY` | yes | All CoinGlass data fetches (used by tech/onchain agents through the shared base cache) | API starts but tech/onchain agents abstain (warning at startup) |
| `OPENAI_API_KEY` | yes (if any OpenAI agent active) | `MultiagentSystem/llm_factory.py` for `gpt-*` model ids | LLM agents return abstain stub (`429 insufficient_quota` or `RuntimeError`) |
| `CLAUDE_KEY` | optional | Same factory, for `claude-*` model ids | RuntimeError only when an agent is configured with a `claude-*` `llm_model` |
| `TWITTER_EMAIL`, `TWITTER_PASSWORD` | required for headed re-login | `chrome_login_before_scrapping.py` | Re-login flow fails (existing cookies still work) |
| `TWITTER_USERNAME` | optional | Same — fallback for "unusual activity" challenge | Verification challenge cannot be solved |
| `TWITTER_UPLOAD_KEY` | required for `/api/agents/twitter-upload-cookies` | [`api/routers/multiagent_predictions.py:249`](api/routers/multiagent_predictions.py#L249) | Endpoint always returns 401 |

Example `dev.env`:

```env
COINGLASS_API_KEY=...
OPENAI_API_KEY=sk-...
CLAUDE_KEY=sk-ant-...
TWITTER_EMAIL=you@example.com
TWITTER_PASSWORD=...
TWITTER_USERNAME=your_handle
TWITTER_UPLOAD_KEY=<exactly 100 random chars>
```

---

## 4. API reference

All endpoints are mounted under the prefix `/api` in [`api/main.py`](api/main.py). Full request/response schemas live in [`api/schemas.py`](api/schemas.py); Swagger at `/docs` is the live source of truth.

### 4.1 Endpoint table

| Method | Path | Purpose | Concurrency |
|---|---|---|---|
| GET | `/api/health` | Server status | — |
| POST | `/api/multiagent_predictions` | Run LangGraph DAG for last N eligible dates | `_prediction_lock` → 409 if already running |
| POST | `/api/system/collect_agent_data` | Incremental fetch into news / calendar / twitter SQLite archives | per-agent `_collection_locks` → 409 per agent |
| GET | `/api/agents/data-status` | MAX(date) per agent's SQLite archive | — |
| GET | `/api/agents/twitter-auth-status` | Twitter session + cookie health check | — |
| POST | `/api/agents/twitter-upload-cookies` | Replace `twitter_cookies.json` (re-login without restart) | requires `TWITTER_UPLOAD_KEY` (401 otherwise) |

### 4.2 Concurrency model

Long-running endpoints are guarded by `asyncio.Lock` instances so that **only one** instance of each operation can run at a time. The currently-running request is **never interrupted** — it runs to completion. Any **new** request that arrives while the lock is held is **rejected immediately with HTTP 409 Conflict** (no queuing, no waiting). The client decides whether to retry later.

Example: a multiagent run for 100 dates may take ~1 hour. While it runs, a second `POST /api/multiagent_predictions` returns 409 instantly. The first run keeps going untouched. As soon as it finishes, the next call is accepted.

Locks:

- `_prediction_lock` — held for the duration of a multiagent run
- `_collection_locks[agent]` — one per agent (`news_analyser`, `economic_calendar_analyser`, `twitter_analyser`)

### 4.3 Examples

The full step-by-step walkthrough (data-status → collect → predict) lives in [HOW_TO_USE_MULTIAGENT_ENDPOINTS_GUIDE.md](HOW_TO_USE_MULTIAGENT_ENDPOINTS_GUIDE.md). Quick references below.

**Run a prediction (config-driven):**

```bash
curl -X POST "http://localhost:8080/api/multiagent_predictions" \
  -H "Content-Type: application/json" \
  -d @configs/multiagent_config.json
# or send a body with the same shape plus n_last_dates
```

The body schema mirrors `configs/multiagent_config.json` plus `n_last_dates: int` (1–365). See [`api/schemas.py:MultiagentPredictionsRequest`](api/schemas.py) for the full Pydantic example.

**Collect news + calendar + twitter:**

```bash
curl -X POST "http://localhost:8080/api/system/collect_agent_data" \
  -H "Content-Type: application/json" \
  -d '{
    "agents": ["news_analyser", "economic_calendar_analyser", "twitter_analyser"],
    "twitter_since_date": "2026-04-28",
    "twitter_until_date": "2026-05-11"
  }'
```

`twitter_since_date` / `twitter_until_date` apply only to `twitter_analyser` (news/calendar always increment from `MAX(date)` to today). When both dates are passed, the just-fetched tweets are also LLM-classified in the same call.

**Check data freshness:**

```bash
curl http://localhost:8080/api/agents/data-status
```

Returns `MAX(date)` per archive — compare against `forecast_start_date - window + 1` for each active agent to decide whether you need to top up the archives.

**Twitter session health and re-login:**

```bash
# Check if cookies are still good
curl http://localhost:8080/api/agents/twitter-auth-status

# Upload fresh cookies (when relogin_required=true and you can't open a GUI on the host)
curl -X POST "http://localhost:8080/api/agents/twitter-upload-cookies" \
  -H "Content-Type: application/json" \
  -d '{"upload_key": "<TWITTER_UPLOAD_KEY value>",
       "cookies":    [{"name": "auth_token", "value": "...", "domain": ".x.com", "path": "/"},
                      {"name": "ct0",        "value": "...", "domain": ".x.com", "path": "/"}]}'
```

### 4.4 Python client snippet

```python
import requests, json

BASE = "http://localhost:8080"

with open("configs/multiagent_config.json") as f:
    body = json.load(f)
body["n_last_dates"] = 1

resp = requests.post(f"{BASE}/api/multiagent_predictions", json=body)
data = resp.json()

for row in data["predictions"]:
    pred = row["y_prediction"]
    direction = "LONG" if pred == 1 else "SHORT" if pred == 0 else "NEUTRAL"
    print(f"{row['date']}: {direction} (score={row['confidence_score']:.2f}, base={row['base_price']})")
```

---

## 5. Module CLI cheatsheet

All commands are run from the project root with the venv activated.

| Task | Command |
|---|---|
| Run multiagent predictions for last N days (config-driven) | `python -m MultiagentSystem.multiagent_system_main` |
| Tune Twitter agent hyperparameters via Optuna | `python -m MultiagentSystem.agents_tuners.twitter_tuner.tuner_main` |
| Re-login to Twitter (headed Chrome, writes `twitter_cookies.json`) | `python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.chrome_login_before_scrapping --login` |
| Start the API locally (canonical port) | `uvicorn api.main:app --host 0.0.0.0 --port 8080` |
| Start the API for development | `uvicorn api.main:app --reload --port 8080` |

---

## 6. Multiagent config

Canonical config at [configs/multiagent_config.json](configs/multiagent_config.json) — currently activates **two agents**: `agent_for_twitter_analysis` and `agent_for_analysing_tech_indicators`. Other agents' settings live in `agent_settings` "in reserve" — they are ignored unless added to `agent_envolved_in_prediction`.

```json
{
  "forecast_start_date": "2026-05-11",
  "horizon": 1,
  "agent_envolved_in_prediction": [
    "agent_for_twitter_analysis",
    "agent_for_analysing_tech_indicators"
  ],
  "neutral_threshold": 0.0,
  "agent_settings": {
    "agent_for_analysing_tech_indicators": {
      "system_prompt_file": "agents/tech_indicators/system_prompt_general.md",
      "llm_model": "gpt-4.1",
      "window_to_analysis": 21,
      "base_feats": ["spot_price_history__close", "..."]
    },
    "agent_for_twitter_analysis": {
      "authors": ["CarpeNoctom", "rektcapital", "..."],
      "window_to_analysis": 14,
      "decay_rate": 0.05,
      "decay_start_day": 1,
      "initial_weight": 1.0
    },
    "verdicts_validator": { "llm_model": "gpt-4.1" }
  }
}
```

| Top-level field | Meaning |
|---|---|
| `forecast_start_date` | **Decision day** — last date with known data. Prediction is made for `forecast_start_date + horizon`. Must be ≤ yesterday. |
| `horizon` | Forecast horizon in days (1–30). |
| `agent_envolved_in_prediction` | Only agents listed here vote. Others run the node but return `{}`. |
| `neutral_threshold` | Score in `[-3, +3]`; `\|score\| ≤ neutral_threshold` → verdict `None`. Default `0.0` — no neutral band. |
| `agent_settings` | Per-agent block, schema-free. Common keys: `llm_model`, `system_prompt_file`, `window_to_analysis`, `base_feats`, `decay_rate`, `decay_start_day`, `initial_weight`, `authors`. |

---

## 7. Multiagent system internals

A LangGraph DAG of LLM agents that vote on LONG/SHORT for a given `forecast_start_date`. Source under `MultiagentSystem/`.

### 7.1 Graph

```
START → supervisor → [tech, onchain, news, twitter, economic_calendar]   (parallel)
                   → validator                                           (fan-in)
                   → _should_retry?
                       ├─ retry → supervisor   (any agent has requirements & retry budget left)
                       └─ done  → reports_analyser → END
```

Active voting agents are decided **at runtime** from `agent_envolved_in_prediction`. Agents not listed there still execute but return `{}`. `MAX_RETRIES = 2` per retry-eligible agent.

### 7.2 Agent inventory

| Agent | What it does |
|---|---|
| `agent_for_analysing_tech_indicators` | LLM reads windowed TA/OHLCV slice from the cached base df; returns LONG/SHORT + confidence |
| `agent_for_analysing_onchain_indicators` | LLM analyses LTH/STH supply, active addresses, reserve risk |
| `agent_for_news_analysis` | Classifies news from `news_archive.db` and decays by age |
| `agent_for_twitter_analysis` | Aggregates pre-classified tweets from `twitter_archive.db`; **no LLM at predict-time** — pure formula with exponential age decay |
| `agent_for_economic_calendar_analysis` | LLM analyses major + medium US calendar events |
| `agent_for_verdicts_validation` | Quality-checks tech + onchain outputs; can request a retry |
| `agent_reports_analyser` | Aggregates validated signals: `score = mean(sign × weight)`; verdict by `neutral_threshold` |

### 7.3 Final verdict math

```
weight = {"low": 1, "medium": 2, "high": 3}[confidence]
sign   = +1 if prediction is True (LONG/HIGHER) else -1
vote   = sign * weight                               # ∈ {-3..-1, +1..+3}
score  = arithmetic mean of votes over voting agents # ∈ [-3, +3]

direction = LONG  if score >  neutral_threshold
            SHORT if score < -neutral_threshold
            None  otherwise
```

Agents with `prediction is None` or `confidence is None` abstain and are **excluded from the mean** (the divisor counts only voting agents).

### 7.4 `predictions_results.csv` schema

Written by `make_one_prediction` / `make_prediction_for_last_N_days` ([`MultiagentSystem/multiagent_predictions_module.py`](MultiagentSystem/multiagent_predictions_module.py)).

| Column | Meaning |
|---|---|
| `forecast_start_date` | Anchor date (YYYY-MM-DD) |
| `y_predict` | `"LONG"` / `"SHORT"` / `None` |
| `y_predict_confidence` | Aggregate score, float in `[-3, +3]` |
| `summary`, `reasoning`, `risks` | Human-readable verdict text |
| `{agent_short}__prediction` | Per-agent True (LONG) / False (SHORT) / None |
| `{agent_short}__confidence` | Per-agent `"high"`/`"medium"`/`"low"` / None |
| `start_date_price`, `btc_bybit_close_price`, `btc_bybit_high_price`, `btc_bybit_low_price` | Filled by `add_y_true()` |
| `y_true` | `"LONG"` / `"SHORT"` / `None` (None for future dates) |

`agent_short` = agent name with `agent_for_` and `agent_for_analysing_` stripped (e.g. `tech_indicators__prediction`).

---

## 8. Project structure

```
.
├── api/                                 # FastAPI service
│   ├── main.py                          # App entry, CORS, lifespan
│   ├── schemas.py                       # Pydantic request/response models
│   └── routers/
│       └── multiagent_predictions.py    # /api/multiagent_predictions, /api/system/collect_agent_data,
│                                        # /api/agents/{data-status,twitter-auth-status,twitter-upload-cookies}
│
├── MultiagentSystem/
│   ├── multiagent_graph.py              # build_multiagent_graph() — DAG wiring
│   ├── multiagent_system_main.py        # __main__ runner; re-exports compiled `app`
│   ├── multiagent_predictions_module.py # make_one_prediction / make_prediction_for_last_N_days
│   ├── multiagent_types.py              # AgentState, AgentSignal, reducers
│   ├── llm_factory.py                   # gpt-* → ChatOpenAI, claude-* → ChatAnthropic
│   ├── agents/                          # tech_indicators, twitter_analyser, news_analyser,
│   │                                    # onchain_indicators, economic_calendar_analyser,
│   │                                    # verdicts_validator, reports_analyser, unbias_agent
│   └── agents_tuners/twitter_tuner/     # Optuna hyperparameter search
│
├── configs/
│   └── multiagent_config.json           # Multiagent runtime config
│
├── Logs/
│   └── LoggingSystem/                   # stdout → logs.log helper (training/tuning only)
│
├── Dockerfile                           # Builds the prod image (port 8080)
├── docker-compose.yml                   # One-shot local deploy
├── dev.env                              # API keys + secrets (NOT committed in production)
├── HOW_TO_USE_MULTIAGENT_ENDPOINTS_GUIDE.md  # Step-by-step API walkthrough
└── requirements.txt
```

---

## 9. Troubleshooting

**`409 Conflict` on prediction / collection.** Another request is already holding the lock for that operation. Wait for it to finish; the API does not queue.

**Startup warning: "Failed to fetch CoinGlass dataset".** `COINGLASS_API_KEY` is missing or the API is unreachable. Tech/onchain agents will abstain because the windowed feature slice is empty.

**Tech-indicators agent abstains every day with `no vote (skipped)`.** The LLM call failed (commonly `429 insufficient_quota` for OpenAI). Either top up the OpenAI account or change `agent_settings.agent_for_analysing_tech_indicators.llm_model` to a `claude-*` id (requires `CLAUDE_KEY`).

**Twitter agent always says "too weak — abstaining".** Window has too few actionable tweets. Causes: (1) the `twitter_archive.db` does not cover `forecast_start_date - window + 1 … forecast_start_date` — fix via `POST /api/system/collect_agent_data` with explicit `twitter_since_date` / `twitter_until_date`; (2) the classifier dropped most tweets to `NO_CORRELATION_TO_BTC`; (3) BULL/BEAR signals from different authors cancel each other within a date.

**`relogin_required: true` from `/api/agents/twitter-auth-status`.** The Twitter session expired. Two options:
1. Run the headed login on a machine with a display:
   ```bash
   python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.chrome_login_before_scrapping --login
   ```
2. Or upload fresh cookies via `POST /api/agents/twitter-upload-cookies` (needs `TWITTER_UPLOAD_KEY`).

**Chrome volume mount issues in Docker.** The container needs `chrome_profile/` mounted at the path shown in `docker-compose.yml`. Without it, every Twitter scrape starts from a fresh profile and fails authentication.

---

## 10. Multiagent tests

Tests live in [`MultiagentSystem/tests/`](MultiagentSystem/tests) and are written with the stdlib **`unittest`** framework — no extra dependency required, just the venv from §2.

| File | What it covers | Type |
|---|---|---|
| [`test_reports_analyser.py`](MultiagentSystem/tests/test_reports_analyser.py) | `compute_confidence_score`: weight table, abstain handling, neutral threshold, breakdown text | Pure unit |
| [`test_twitter_aggregation.py`](MultiagentSystem/tests/test_twitter_aggregation.py) | The four pure aggregation helpers in `agent_for_twitter_analysis` (window dates, group-by-date, per-author averaging, age-decay verdict) | Pure unit |
| [`test_make_one_prediction.py`](MultiagentSystem/tests/test_make_one_prediction.py) | End-to-end run of `make_one_prediction` through the full LangGraph DAG with only the Twitter agent enabled (no LLM call, no network) | Integration |

### Run all multiagent tests

From the project root, with the venv activated:

```bash
# Windows
.venv\Scripts\python.exe -m unittest discover -s MultiagentSystem/tests -v

# macOS / Linux
.venv/bin/python -m unittest discover -s MultiagentSystem/tests -v
```

### Run a single test file

```bash
python -m unittest MultiagentSystem.tests.test_reports_analyser -v
python -m unittest MultiagentSystem.tests.test_twitter_aggregation -v
python -m unittest MultiagentSystem.tests.test_make_one_prediction -v
```

### Run a single test class or method

```bash
# Class
python -m unittest MultiagentSystem.tests.test_reports_analyser.TestComputeConfidenceScore -v

# Method
python -m unittest MultiagentSystem.tests.test_reports_analyser.TestComputeConfidenceScore.test_no_signals_returns_zero_neutral -v
```

### Notes

- The tests make no network calls and never invoke an LLM. `test_make_one_prediction.py` patches the SQLite tweet reader via `unittest.mock`, and the validator short-circuits on `claude-*` models when `CLAUDE_KEY` is unset. They are safe to run in CI without `OPENAI_API_KEY` / `CLAUDE_KEY` / `COINGLASS_API_KEY`.
- Whenever you add a new agent or change the aggregation formula in `compute_confidence_score` / `agent_for_twitter_analysis`, run the corresponding file — it covers the edge cases (abstain, zero mean, decay zone).
