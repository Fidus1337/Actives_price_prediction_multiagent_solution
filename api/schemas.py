"""Pydantic models for API request/response validation."""

import re
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class ClassicML_PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""

    models: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of model names from Models folder",
        examples=[["base_model_1d", "range_model_3d"]]
    )
    dates: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of dates in YYYY-MM-DD format",
        examples=[["2026-04-01", "2026-04-02"]]
    )
    refresh_dataset: bool = Field(
        default=False,
        description="Force refresh dataset from API before predicting. "
                    "If False, uses cached data (auto-loads on first request)."
    )

    @field_validator("dates")
    @classmethod
    def validate_date_format(cls, v: list[str]) -> list[str]:
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for d in v:
            if not pattern.match(d):
                raise ValueError(f"Invalid date format: {d}. Expected YYYY-MM-DD")
        return v


class ClassicML_SinglePrediction(BaseModel):
    """Prediction result for a single date."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction (0=down, 1=up)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of price increase")
    spot_price_close: float | None = Field(None, description="BTC spot close price on prediction date")
    range_sma: float | None = Field(None, description="SMA of close price over ma_window days. Populated for range models only.")
    sma_window: int | None = Field(None, exclude=True)

    @model_serializer(mode="wrap")
    def _serialize(self, handler) -> dict:
        data = handler(self)
        if "range_sma" in data:
            # Hide nullable field entirely for base models / missing values
            if data["range_sma"] is None:
                data.pop("range_sma", None)
            elif self.sma_window is not None:
                data[f"range_sma_{self.sma_window}"] = data.pop("range_sma")
        return data


class ClassicML_ModelPredictionResult(BaseModel):
    """Prediction results for a single model."""

    model_name: str = Field(..., description="Model used for predictions")
    model_type: str = Field(..., description="Model type (base or range)")
    horizon_days: int = Field(..., description="Prediction horizon in days")
    found_dates: list[str] = Field(..., description="Dates found in data")
    missing_dates: list[str] = Field(..., description="Dates not found in data")
    predictions: list[ClassicML_SinglePrediction] = Field(..., description="List of predictions")
    error: str | None = Field(None, description="Error message if model failed")


class ClassicML_PredictionResponse(BaseModel):
    """Response schema for batch predictions endpoint."""

    requested_models: list[str] = Field(..., description="Models requested")
    requested_dates: list[str] = Field(..., description="Dates requested")
    results: list[ClassicML_ModelPredictionResult] = Field(..., description="Predictions per model")


class ModelMetrics(BaseModel):
    """Quality metrics for a model."""

    auc: float = Field(..., description="Area Under ROC Curve")
    accuracy: float = Field(..., description="Accuracy score")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1: float = Field(..., description="F1 score")
    threshold: float = Field(default=0.5, description="Classification threshold")


class ModelInfo(BaseModel):
    """Information about a single model."""

    name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type (base or range)")
    horizon_days: int = Field(..., description="Prediction horizon in days")
    feature_count: int = Field(..., description="Number of features used")
    metrics: ModelMetrics | None = Field(None, description="Model quality metrics")


class ModelsResponse(BaseModel):
    """Response schema for models endpoint."""

    available_models: list[ModelInfo]


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = "healthy"
    models_loaded: dict[str, bool] = Field(
        default_factory=dict,
        description="Status of loaded models"
    )


class DatasetStatusResponse(BaseModel):
    """Response schema for dataset status endpoint."""

    is_loaded: bool = Field(..., description="Whether dataset is currently loaded in memory")
    last_refreshed_at: str | None = Field(None, description="ISO datetime of last refresh, null if never loaded")
    shape: list[int] | None = Field(None, description="[rows, columns], null if not loaded")

# Схема ожидаемого JSON. 
# Мы повторяем структуру config.json, где есть ключ "runs", содержащий список конфигов.
class TrainConfigRequest(BaseModel):
    runs: List[Dict[str, Any]]

    # Добавляем конфигурацию с примером
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "runs": [
                        {
                            "name": "base_model_1d",
                            "N_DAYS": 1,
                            "threshold": 0.5,
                            "base_feats": [
                                "spot_price_history__close__pct1",
                                "spot_price_history__close__diff1",
                                "futures_open_interest_aggregated_history__close__pct1",
                                "futures_liquidation_aggregated_history__aggregated_short_liquidation_usd__diff1",
                                "futures_global_long_short_account_ratio_history__global_account_long_percent__pct1",
                                "futures_top_long_short_account_ratio_history__top_account_long_short_ratio__pct1",
                                "premium__diff1",
                                "cb_premium_abs"
                            ]
                        },
                        {
                            "name": "range_model_1d",
                            "N_DAYS": 1,
                            "threshold": 0.5,
                            "ma_window": 14,
                            "range_feats": [
                                "range_pct",
                                "range_pct_ma14"
                            ],
                            "base_feats": [
                                "spot_price_history__close__pct1",
                                "spot_price_history__close__diff1",
                                "futures_open_interest_aggregated_history__close__pct1",
                                "futures_liquidation_aggregated_history__aggregated_short_liquidation_usd__diff1",
                                "futures_global_long_short_account_ratio_history__global_account_long_percent__pct1",
                                "futures_top_long_short_account_ratio_history__top_account_long_short_ratio__pct1",
                                "premium__diff1",
                                "cb_premium_abs"
                            ]
                        },
                        {
                            "name": "base_model_3d",
                            "N_DAYS": 3,
                            "threshold": 0.5,
                            "base_feats": [
                                "spot_price_history__close__pct1",
                                "spot_price_history__close__diff1",
                                "feat__orderbook_imbalance_usd",
                                "futures_liquidation_aggregated_history__aggregated_short_liquidation_usd__diff1",
                                "futures_open_interest_aggregated_stablecoin_history__close",
                                "futures_open_interest_aggregated_stablecoin_history__low",
                                "cgdi_log_level",
                                "cgdi_dev_softsign",
                                "cgdi_dev_from_base",
                                "cgdi_index_value"
                            ]
                        },
                        {
                            "name": "range_model_3d",
                            "N_DAYS": 3,
                            "threshold": 0.5,
                            "ma_window": 14,
                            "range_feats": [
                                "range_pct",
                                "range_pct_ma14"
                            ],
                            "base_feats": [
                                "spot_price_history__close__pct1",
                                "spot_price_history__close__diff1",
                                "feat__orderbook_imbalance_usd",
                                "futures_liquidation_aggregated_history__aggregated_short_liquidation_usd__diff1",
                                "futures_open_interest_aggregated_stablecoin_history__close",
                                "futures_open_interest_aggregated_stablecoin_history__low",
                                "cgdi_log_level",
                                "cgdi_dev_softsign",
                                "cgdi_dev_from_base",
                                "cgdi_index_value"
                            ]
                        },
                        {
                            "name": "base_model_5d",
                            "N_DAYS": 5,
                            "threshold": 0.5,
                            "base_feats": [
                                "cgdi_dev_from_base",
                                "cgdi_index_value",
                                "cgdi_log_level",
                                "cgdi_dev_softsign",
                                "futures_open_interest_aggregated_stablecoin_history__close",
                                "futures_open_interest_aggregated_stablecoin_history__low",
                                "futures_open_interest_aggregated_stablecoin_history__high",
                                "futures_open_interest_aggregated_stablecoin_history__open",
                                "futures_orderbook_aggregated_ask_bids_history__aggregated_asks_usd__pct1",
                                "futures_orderbook_ask_bids_history__asks_usd__pct1"
                            ]
                        },
                        {
                            "name": "range_model_5d",
                            "N_DAYS": 5,
                            "threshold": 0.6,
                            "ma_window": 14,
                            "range_feats": [
                                "range_pct",
                                "range_pct_ma14"
                            ],
                            "base_feats": [
                                "cgdi_dev_from_base",
                                "cgdi_index_value",
                                "cgdi_log_level",
                                "cgdi_dev_softsign",
                                "futures_open_interest_aggregated_stablecoin_history__close",
                                "futures_open_interest_aggregated_stablecoin_history__low",
                                "futures_open_interest_aggregated_stablecoin_history__high",
                                "futures_open_interest_aggregated_stablecoin_history__open",
                                "futures_orderbook_aggregated_ask_bids_history__aggregated_asks_usd__pct1",
                                "futures_orderbook_ask_bids_history__asks_usd__pct1"
                            ]
                        },
                        {
                            "name": "base_model_7d",
                            "N_DAYS": 7,
                            "threshold": 0.5,
                            "base_feats": [
                                "cgdi_dev_from_base",
                                "cgdi_index_value",
                                "cgdi_log_level",
                                "cgdi_dev_softsign",
                                "futures_open_interest_aggregated_stablecoin_history__close",
                                "futures_open_interest_aggregated_stablecoin_history__low",
                                "futures_open_interest_aggregated_stablecoin_history__high",
                                "futures_open_interest_aggregated_stablecoin_history__open",
                                "futures_orderbook_aggregated_ask_bids_history__aggregated_asks_usd__pct1",
                                "futures_orderbook_ask_bids_history__asks_usd__pct1"
                            ]
                        },
                        {
                            "name": "range_model_7d",
                            "N_DAYS": 7,
                            "threshold": 0.5,
                            "ma_window": 14,
                            "range_feats": [
                                "range_pct",
                                "range_pct_ma14"
                            ],
                            "base_feats": [
                                "long_quantity",
                                "index_btc_reserve_risk__reserve_risk_index",
                                "index_btc_sth_supply__supply_z180",
                                "index_btc_reserve_risk__hodl_bank__pct1",
                                "index_btc_lth_supply__supply_z180__pct1",
                                "gold__volume",
                                "sp500__low__diff1__lag3"
                            ]
                        }
                    ]
                }
            ]
        }
    }
    
class TrainConfigResponse(BaseModel):
    status: str
    message: str
    source: str

    # Опционально: пример, который будет показан в документации
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "message": "Training executed successfully.",
                    "source": "custom_json"
                }
            ]
        }
    }


class MultiagentPredictionsRequest(BaseModel):
    """Request body for multiagent predictions endpoint."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "forecast_start_date": "2026-04-20",
            "horizon": 1,
            "n_last_dates": 1,
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
                    "base_feats": [
                        "spot_price_history__close",
                        "spot_price_history__intraday_range_pct",
                        "spot_price_history__volume_usd__pct1",
                        "spot_price_history__realized_vol_3d",
                        "spot_price_history__realized_vol_7d",
                        "spot_price_history__close__sma7_rel",
                        "spot_price_history__close__sma14_rel",
                        "spot_price_history__ta_rsi",
                        "spot_price_history__ta_adx",
                        "spot_price_history__ta_bbw",
                        "futures_open_interest_aggregated_history__close__pct1",
                        "futures_open_interest_aggregated_stablecoin_history__close__pct1",
                        "futures_open_interest_aggregated_coin_margin_history__close__pct1",
                        "futures_funding_rate_oi_weight_history__close",
                        "feat__funding_minus_oi_weight",
                        "feat__taker_imbalance_agg",
                        "feat__liq_imbalance_short_minus_long",
                        "feat__liq_total_pct1",
                        "futures_global_long_short_account_ratio_history__global_account_long_short_ratio",
                        "cb_premium_rate_bps",
                        "cb_premium_abs"
                    ]
                },
                "agent_for_news_analysis": {
                    "system_prompt_file": "agents/news_analyser/system_prompt.md",
                    "window_to_analysis": 1,
                    "decay_rate": 0.20,
                    "decay_start_day": 1,
                    "initial_weight": 1.0
                },
                "agent_for_economic_calendar_analysis": {
                    "llm_model": "gpt-4.1",
                    "window_to_analysis": 1
                },
                "agent_for_twitter_analysis": {
                    "authors": [
                        "CarpeNoctom",
                        "JSeyff",
                        "AltcoinPsycho",
                        "DavidDuong",
                        "TraderMercury",
                        "_Checkmatey_",
                        "CryptoHayes",
                        "rektcapital"
                    ],
                    "window_to_analysis": 14,
                    "decay_rate": 0.05,
                    "decay_start_day": 1,
                    "initial_weight": 1.0
                },
                "verdicts_validator": {
                    "llm_model": "gpt-4.1"
                },
                "agent_for_analysing_onchain_indicators": {
                    "system_prompt_file": "agents/onchain_indicators/system_prompt_1d.md",
                    "llm_model": "claude-sonnet-4-5",
                    "window_to_analysis": 21,
                    "base_feats": [
                        "spot_price_history__open",
                        "spot_price_history__high",
                        "spot_price_history__low",
                        "spot_price_history__close",
                        "index_btc_lth_supply__lth_supply",
                        "index_btc_lth_supply__supply_slope14",
                        "index_btc_lth_supply__supply_z180",
                        "index_btc_lth_supply__lth_supply__lag3",
                        "index_btc_lth_supply__lth_supply__lag5",
                        "index_btc_lth_supply__lth_supply__lag7",
                        "index_btc_sth_supply__sth_supply",
                        "index_btc_sth_supply__supply_slope14",
                        "index_btc_sth_supply__supply_z180",
                        "index_btc_sth_supply__sth_supply__lag1",
                        "index_btc_sth_supply__sth_supply__lag3",
                        "index_btc_sth_supply__sth_supply__lag5",
                        "index_btc_sth_supply__sth_supply__lag7",
                        "index_btc_active_addresses__active_address_count",
                        "index_btc_active_addresses__aa_z180",
                        "index_btc_active_addresses__aa_slope14",
                        "index_btc_reserve_risk__reserve_risk_index",
                        "index_btc_reserve_risk__log_rr",
                        "index_btc_reserve_risk__rr_z180",
                        "index_btc_reserve_risk__rr_slope14"
                    ]
                }
            }
        }
    })

    forecast_start_date: str = Field(..., description="Anchor date in YYYY-MM-DD format")
    horizon: int = Field(..., ge=1, le=30, description="Prediction horizon in days")
    agent_envolved_in_prediction: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "List of active agent names for this run. Supported: "
            "agent_for_analysing_tech_indicators, agent_for_twitter_analysis, "
            "agent_for_news_analysis, agent_for_economic_calendar_analysis, "
            "agent_for_analysing_onchain_indicators"
        ),
    )
    neutral_threshold: float = Field(
        default=0.0,
        description="Threshold for neutral verdict in reports analyser",
    )
    agent_settings: dict[str, dict[str, Any]] = Field(
        ...,
        description=(
            "Per-agent settings map, same shape as multiagent_config.json. "
            "Each agent block may include: system_prompt_file, llm_model "
            "(e.g. 'gpt-4.1' or 'claude-sonnet-4-5'), window_to_analysis, "
            "base_feats, decay_rate, decay_start_day, initial_weight, authors. "
            "Reserved keys: 'verdicts_validator' (llm_model override for the validator node)."
        ),
    )
    n_last_dates: int = Field(
        default=10,
        ge=1,
        le=365,
        description="Number of last eligible dates to evaluate",
    )

    @field_validator("forecast_start_date")
    @classmethod
    def validate_forecast_date(cls, value: str) -> str:
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if not pattern.match(value):
            raise ValueError("forecast_start_date must be in YYYY-MM-DD format")
        return value


class AgentPredictionDetail(BaseModel):
    prediction: int | None = Field(
        None, ge=0, le=1,
        description="Per-agent binary direction (1=HIGHER/BULL, 0=LOWER/BEAR, None=no vote)",
    )
    confidence: str | None = Field(
        None, description="Per-agent confidence: 'high' | 'medium' | 'low' | None"
    )
    summary: str | None = Field(None, description="Short verdict text from the agent")
    reasoning: str | None = Field(None, description="Detailed reasoning text from the agent")
    risks: str | None = Field(None, description="Counter-arguments / risks noted by the agent")


class MultiagentSinglePrediction(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    base_price: float | None = Field(
        None,
        description="BTC spot close price (Bybit) on forecast_start_date — the price the prediction is anchored to",
    )
    y_true: int | None = Field(..., description="Observed binary outcome (None if future date)")
    y_prediction: int | None = Field(None, ge=0, le=1, description="Predicted binary direction")
    confidence_score: float | int | None = Field(None, description="Aggregated multiagent confidence score")
    agents: dict[str, AgentPredictionDetail] = Field(
        default_factory=dict,
        description=(
            "Per-agent breakdown keyed by full agent name "
            "(e.g. 'agent_for_analysing_tech_indicators'). "
            "Only agents present in agent_envolved_in_prediction appear here."
        ),
    )


class MultiagentPredictionsResponse(BaseModel):
    requested_forecast_start_date: str
    requested_horizon: int
    requested_n_last_dates: int
    rows_returned: int
    predictions: list[MultiagentSinglePrediction]


class CollectAgentDataRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "agents": ["news_analyser", "economic_calendar_analyser", "twitter_analyser"],
            "twitter_authors": [
                "CarpeNoctom", "caprioleio", "JSeyff", "DonAlt", "krugermacro",
                "DavidDuong", "TraderMercury", "Yodaskk", "_Checkmatey_", "CrypNuevo",
            ],
            "twitter_since_date": "2026-03-01",
            "twitter_until_date": "2026-04-03",
        }
    })

    agents: list[str] = Field(
        default=["news_analyser", "economic_calendar_analyser", "twitter_analyser"],
        description="Agents to collect data for. Valid values: 'news_analyser', 'economic_calendar_analyser', 'twitter_analyser'",
    )
    twitter_authors: list[str] | None = Field(
        default=None,
        description="Optional list of Twitter usernames to scrape. If omitted, uses all enabled accounts from twitter_collector_settings.json",
    )
    twitter_since_date: str | None = Field(
        default=None,
        description="Start date for Twitter scraping (YYYY-MM-DD). If omitted, uses latest date in DB (incremental).",
    )
    twitter_until_date: str | None = Field(
        default=None,
        description="End date for Twitter scraping (YYYY-MM-DD). If omitted, uses today.",
    )


class CollectAgentDataResult(BaseModel):
    agent: str
    before: int
    fetched: int
    new: int
    after: int
    date_range: str | None = None


class CollectAgentDataResponse(BaseModel):
    results: list[CollectAgentDataResult]


class AgentsDataStatusResponse(BaseModel):
    news_analyser: str | None = Field(None, description="Last fetched date in news archive (YYYY-MM-DD)")
    economic_calendar_analyser: str | None = Field(None, description="Last fetched date in calendar archive (YYYY-MM-DD)")
    twitter_analyser: str | None = Field(None, description="Last fetched date in twitter archive (YYYY-MM-DD)")


class TwitterAuthStatusResponse(BaseModel):
    cookies_exist: bool = Field(..., description="twitter_cookies.json exists and is non-empty")
    session_cookies_ok: bool = Field(..., description="auth_token and ct0 session cookies are present")
    credentials_configured: bool = Field(..., description="TWITTER_EMAIL and TWITTER_PASSWORD are set in dev.env")
    cookies_path: str = Field(..., description="Absolute path to twitter_cookies.json")
    cookies_count: int = Field(..., description="Number of cookies in the file")
    relogin_required: bool = Field(..., description="True when session cookies are missing or credentials are not configured")


class TwitterCookiesUploadRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "upload_key": "<100-char key from TWITTER_UPLOAD_KEY in dev.env>",
            "cookies": [
                {"name": "auth_token", "value": "...", "domain": ".x.com", "path": "/"},
                {"name": "ct0", "value": "...", "domain": ".x.com", "path": "/"},
            ]
        }
    })

    upload_key: str = Field(
        ...,
        min_length=100,
        max_length=100,
        description="Secret key from TWITTER_UPLOAD_KEY in dev.env",
    )
    cookies: list[dict] = Field(
        ...,
        min_length=1,
        description="Cookie list exported from browser (DevTools → Application → Cookies, or EditThisCookie extension)",
    )


class TwitterCookiesUploadResponse(BaseModel):
    saved: int = Field(..., description="Number of cookies written to twitter_cookies.json")
    session_cookies_ok: bool = Field(..., description="auth_token and ct0 are present in the uploaded cookies")
    cookies_path: str