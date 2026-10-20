# PocketQuant2

A personal quantitative trading research repo: fetch market data, build technical/statistical features, backtest strategies with [vectorbt](https://github.com/polakowo/vectorbt), and train ML models (XGBoost) on top of it. Includes a standing research track on VIX mean-reversion and a scheduled options-data fetch pipeline.


## Layout

```
PocketQuant2/
├── scraper/            # API clients (yfinance, AlphaVantage) + feature math + SQLite build/read
│   ├── api_clients/     # AlphaVantage.py, YFinance.py
│   ├── features/        # RSI, MACD, Bollinger, ATR, moving averages, returns, volatility
│   ├── utils/            # build_db.py, feature_builder.py, filter.py
│   └── setup.py         # BaseSetup: fetch -> cache -> build_features pipeline
├── backtester/          # Backtester(BaseSetup): signal alignment, run_backtest, stats, alpha/beta
├── models/               # ML pipelines (XGBoost daily-direction classifier)
├── research/             # Feature evaluation tools + the vix_mean_reversion research track
├── execution/            # Scheduled fetch scripts (SQLite + DynamoDB) and their logs
├── experiments/          # One-off strategy sandboxes and scratch analyses
├── tests/                # Main pytest suite
├── configs/paths.py      # Central path constants (DATA_RAW, DATABASE_PATH, ...)
├── data/                 # Local cache: raw/, processed/, features/, database/ (gitignored)
├── instructions/         # Deep-dive docs for agents/contributors — start at instructions/index.md
└── .claude/skills/backtest/  # Step-by-step skill for building a new backtest correctly
```

For the full architecture (class hierarchy, data flow, canonical data shapes), read **[instructions/index.md](instructions/index.md)** — it's the operational manual and links out to every other doc in `instructions/`.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Always run through `.venv` (see `.agent/rules/venv.md`) — either activate it or call `.\.venv\Scripts\python.exe` directly.

Create a `.env` in the repo root (already gitignored) with:

```
ALPHAVANTAGE_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

AlphaVantage is used by `scraper/api_clients/AlphaVantage.py`; the AWS keys are used by the DynamoDB options-fetch path (`execution/data_fetch/fetch_data_dynamo.py`, `delete_data_dynamo.py`) and by the scheduled `.github/workflows/fetch-options.yml` job (as repo secrets there, not from `.env`).

## Running things

```bash
.\.venv\Scripts\python.exe -m pytest      # full test suite
```

To run a backtest, follow the **[backtest skill](.claude/skills/backtest/SKILL.md)** (preflight -> setup -> signals -> execution -> evaluation -> robustness -> checklist) rather than freehanding it — it encodes lookahead-bias and cost rules that are easy to get wrong. `experiments/test_aapl_backtest.py` and `instructions/examples.md` are good starting templates.

## Known issues

None currently tracked — see `instructions/gotchas.md` for pitfalls and past fixes.

## Data & caching

`data/` (raw prices, processed prices, computed features, SQLite databases) and `execution/logs/` are gitignored — they're local caches/outputs, rebuilt by `BaseSetup.run_pipeline()` and the execution scripts, not source of truth. Nothing under `data/` needs to be committed.
