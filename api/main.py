"""FastAPI application entry point."""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # Set working directory for config/model paths

from api.routers import multiagent_predictions
from api.routers import scheduler as scheduler_router
from api.scheduler import JOB_ID, create_scheduler, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting BTC Price Prediction API...")
    load_dotenv(PROJECT_ROOT / "dev.env")
    if not os.getenv("COINGLASS_API_KEY"):
        print("Warning: COINGLASS_API_KEY not found. Predictions will fail.")

    scheduler = create_scheduler()
    scheduler.start()
    if not get_settings().enabled:
        scheduler.pause_job(JOB_ID)
    app.state.scheduler = scheduler
    job = scheduler.get_job(JOB_ID)
    print(f"Daily collection scheduler started; next run (UTC): {job.next_run_time if job else None}")

    yield

    scheduler.shutdown(wait=False)
    print("Shutting down API...")


app = FastAPI(
    title="BTC Price Direction Prediction API",
    description="""
API for predicting Bitcoin price direction using a multiagent system.

## Multiagent Predictions

- **POST /api/multiagent_predictions** — run LangGraph multiagent system for N dates
- **POST /api/system/collect_agent_data** — collect news / calendar / twitter data
- **GET /api/agents/data-status** — last fetched date per agent archive
- **GET /api/agents/twitter-auth-status** — Twitter session health check
- **POST /api/agents/twitter-upload-cookies** — upload Twitter cookies

## Daily Collection Scheduler

- **GET /api/system/collect_scheduler_settings** — read scheduler settings + next run (UTC)
- **POST /api/system/change_collect_scheduler_settings** — change schedule live (no restart)
""",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(multiagent_predictions.router)
app.include_router(scheduler_router.router)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "BTC Price Prediction API v2.0", "docs": "/docs"}
