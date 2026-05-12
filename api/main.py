"""FastAPI application entry point."""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Ensure project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # Set working directory for config/model paths

from api.routers import multiagent_predictions
from MultiagentSystem.dataset_pipeline.SharedDataCache.SharedBaseDataCache import SharedBaseDataCache

# Module-level reference for access from routers
shared_data_cache: SharedBaseDataCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global shared_data_cache

    print("Starting BTC Price Prediction API...")

    # Load API key and create shared data cache
    load_dotenv(PROJECT_ROOT / "dev.env")
    api_key = os.getenv("COINGLASS_API_KEY")

    if api_key:
        shared_data_cache = SharedBaseDataCache(api_key=api_key)
        print("Shared data cache initialized. Fetching dataset...")
        try:
            shared_data_cache.refresh()
        except Exception as exc:
            print(f"Warning: Failed to fetch CoinGlass dataset: {exc}")
            print("API will start without cached dataset. Multiagent predictions may fail.")
    else:
        print("Warning: COINGLASS_API_KEY not found. Predictions will fail.")

    yield

    print("Shutting down API...")
    if shared_data_cache is not None:
        shared_data_cache.clear()


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

## Service

- **GET /api/health** — server status and loaded models
""",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(multiagent_predictions.router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to documentation."""
    return {"message": "BTC Price Prediction API v2.0", "docs": "/docs"}
