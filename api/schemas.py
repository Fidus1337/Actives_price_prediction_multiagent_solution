"""Pydantic models for API request/response validation."""

import re
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any


class MultiagentPredictionsRequest(BaseModel):
    """Request body for multiagent predictions endpoint."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "forecast_start_date": "2026-05-16",
            "horizon": 1,
            "n_last_dates": 1,
            "agent_envolved_in_prediction": [
                "agent_for_analysing_tech_indicators",
                "agent_for_twitter_analysis",
                "agent_for_analysing_onchain_indicators"
            ],
            "neutral_threshold": 0.0,
            "agent_settings": {
                "agent_for_analysing_tech_indicators": {
                    "system_prompt_file": "agents/tech_indicators/system_prompt_general.md",
                    "llm_model": "gpt-5-mini",
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
                    "window_to_analysis": 3
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
                    "llm_model": "gpt-5-mini",
                    "reasoning_effort": "medium",
                    "window_to_analysis": 21,
                    "base_feats": [
                        "spot_price_history__open",
                        "spot_price_history__high",
                        "spot_price_history__low",
                        "spot_price_history__close",
                        "index_btc_mvrv__mvrv_z180",
                        "index_btc_mvrv__mvrv_slope14",
                        "index_btc_sth_sopr__sopr_z30",
                        "index_btc_sth_sopr__sopr_slope14",
                        "index_btc_lth_sopr__sopr_z180",
                        "index_btc_lth_sopr__sopr_slope14",
                        "index_btc_nupl__nupl_z180",
                        "index_btc_nupl__nupl_slope14",
                        "index_puell_multiple__puell_z180",
                        "index_puell_multiple__puell_slope14"
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
    avg_score: float | None = Field(
        None,
        description=(
            "Per-agent weighted average signal score over its analysis window. "
            "Currently populated by agent_for_twitter_analysis (positive=BULL bias, "
            "negative=BEAR bias). If abs(avg_score) < 0.5 the agent abstains from "
            "voting in the general forecast (prediction/confidence=null), but "
            "avg_score is still reported here so weak signals stay visible."
        ),
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


class PredictionMetrics(BaseModel):
    """Binary classification metrics over the returned predictions.

    Computed only across rows where both y_true and y_prediction are defined.
    Positive class = 1 (HIGHER/LONG); negative class = 0 (LOWER/SHORT).
    """

    evaluable_dates: int = Field(..., description="Rows used to compute metrics (both y_true and y_prediction defined)")
    skipped_dates: int = Field(..., description="Rows excluded (future date with y_true=null, or model abstained with y_prediction=null)")
    tp: int = Field(..., description="True positives: predicted=1, actual=1")
    tn: int = Field(..., description="True negatives: predicted=0, actual=0")
    fp: int = Field(..., description="False positives: predicted=1, actual=0")
    fn: int = Field(..., description="False negatives: predicted=0, actual=1")
    accuracy: float | None = Field(None, description="(TP+TN)/evaluable_dates; null if evaluable_dates==0")
    precision: float | None = Field(None, description="TP/(TP+FP); null if no positives predicted")
    recall: float | None = Field(None, description="TP/(TP+FN); null if no actual positives in the window")


class MultiagentPredictionsResponse(BaseModel):
    requested_forecast_start_date: str
    requested_horizon: int
    requested_n_last_dates: int
    rows_returned: int
    predictions: list[MultiagentSinglePrediction]
    metrics: PredictionMetrics


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
