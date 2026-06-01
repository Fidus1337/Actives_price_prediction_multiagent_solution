import contextlib
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from MultiagentSystem.dataset_pipeline.FeaturesGetterModule.FeaturesGetter import FeaturesGetter
from MultiagentSystem.dataset_pipeline.SharedDataCache.SharedBaseDataCache import SharedBaseDataCache


# Max length for any single string kept in a trace payload. Normal prompts are
# a few KB and pass through whole; only multi-MB blobs (embedded data_json,
# aggregated tweet/news dumps in agent reasoning) get truncated. Full prompts
# are still saved to debug_prompts/ on disk via save_agent_io, so nothing is
# lost — this only keeps each LangSmith POST small enough to upload reliably.
_MAX_TRACE_STR = 30_000


def _redact_heavy(obj):
    """Shrink bulky values in the trace copy of run inputs/outputs.

    cached_dataset (the full SharedBaseDataCache base_df) lives in the graph
    state and is serialized into every node's input/output; large agent
    reasoning strings propagate the same way. Without trimming, one prediction's
    trace is tens of MB and fails to ingest (20MB hard limit + upload timeouts).
    Runs only on the trace copy — the live state and the real LLM prompts are
    untouched.
    """
    try:
        if isinstance(obj, pd.DataFrame):
            return f"<DataFrame shape={obj.shape}>"
        if isinstance(obj, pd.Series):
            return f"<Series len={len(obj)}>"
        if isinstance(obj, str):
            if len(obj) > _MAX_TRACE_STR:
                return obj[:_MAX_TRACE_STR] + f"...[truncated {len(obj) - _MAX_TRACE_STR} chars]"
            return obj
        if isinstance(obj, dict):
            return {k: _redact_heavy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_redact_heavy(v) for v in obj]
    except Exception:
        pass
    return obj


def _configure_tracing_redaction() -> None:
    """Make the LangSmith auto-tracer drop DataFrames from trace payloads.

    Sets hide_inputs/hide_outputs on the shared cached client that the
    env-based tracer (and wait_for_all_tracers) already use, so no per-invoke
    callback wiring is needed. No-op when tracing is disabled.
    """
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        return
    try:
        from langsmith.run_trees import get_cached_client
        client = get_cached_client()
        client._hide_inputs = _redact_heavy
        client._hide_outputs = _redact_heavy
    except Exception as exc:
        print(f"[predictions] Could not configure trace redaction: {exc}")


def _batch_trace(config: dict, last_days: int):
    """Parent trace wrapping a whole N-day backtest.

    Each day's app.invoke nests under this root, so LangSmith shows one trace
    ("backtest <date> N=<n>") whose cost/tokens = the sum across all days.
    No-op context when tracing is disabled.
    """
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        return contextlib.nullcontext()
    try:
        from langsmith import trace
        return trace(
            name=f"backtest {config['forecast_start_date']} N={last_days}",
            run_type="chain",
            project_name=os.getenv("LANGSMITH_PROJECT"),
            tags=["backtest", f"horizon={config['horizon']}"],
            metadata={
                "last_days": last_days,
                "horizon": config["horizon"],
                "forecast_start_date": config["forecast_start_date"],
            },
        )
    except Exception as exc:
        print(f"[predictions] Could not open batch trace: {exc}")
        return contextlib.nullcontext()


def make_one_prediction(
    app,
    config: dict,
    forecast_start_date: str,
    cached_dataset: pd.DataFrame | None,
    save_results: bool = False,
    save_path: str | None = None,
    debug_run_dir: str | None = None,
) -> dict:
    final_state = app.invoke(
        {
            "config": config,
            "horizon": config["horizon"],
            "forecast_start_date": forecast_start_date,
            "agent_envolved_in_prediction": config["agent_envolved_in_prediction"],
            "cached_dataset": cached_dataset,
            "general_prediction_by_all_reports": None,
            "general_reports_summary": "",
            "general_reports_reasoning": "",
            "general_reports_risks": "",
            "confidence_score": 0.0,
            "save_results": save_results,
            "save_path": save_path,
            "debug_run_dir": debug_run_dir,
            "agent_signals": {},
            "retry_agents": [],
        },
        config={
            "run_name": f"prediction {forecast_start_date}",
            "tags": ["prediction", f"horizon={config['horizon']}"],
            "metadata": {
                "forecast_start_date": forecast_start_date,
                "horizon": config["horizon"],
                "agents": config.get("agent_envolved_in_prediction", []),
            },
        },
    )

    row = {
        "forecast_start_date": forecast_start_date,
        "y_predict": final_state.get("general_prediction_by_all_reports"),
        "y_predict_confidence": final_state.get("confidence_score"),
        "summary": final_state.get("general_reports_summary"),
        "reasoning": final_state.get("general_reports_reasoning"),
        "risks": final_state.get("general_reports_risks"),
    }

    # Flatten per-agent signals into columns: prediction, confidence, avg_score, reasoning, summary, risks
    for agent_name, signal in (final_state.get("agent_signals") or {}).items():
        short = agent_name.replace("agent_for_", "").replace("agent_for_analysing_", "")
        row[f"{short}__prediction"] = signal.get("prediction")
        row[f"{short}__confidence"] = signal.get("confidence")
        row[f"{short}__avg_score"] = signal.get("avg_score")
        row[f"{short}__reasoning"] = signal.get("reasoning")
        row[f"{short}__summary"] = signal.get("summary")
        row[f"{short}__risks"] = signal.get("risks")

    if save_results and save_path:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(
            target,
            mode="a",
            header=not target.exists(),
            index=False,
        )

    return row


def make_prediction_for_last_N_days(
    app,
    config: dict,
    last_days: int,
    checkpoint_every: int = 0,
    cm_path: Path | None = None,
    save_results: bool = False,
    save_path: str | None = None,
    cache=None,
    config_hash: str | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Run predictions for the last ``last_days`` dates.

    Optional caching (used by the API layer): when ``cache`` (a duck-typed
    predictions_database.Database) and ``config_hash`` are given, each date's
    LLM prediction is read from / written to the cache keyed by
    (config_hash, forecast_start_date). Only the prediction is cached — y_true is
    recomputed downstream via add_y_true. ``force_recompute`` ignores cached reads
    but still overwrites entries with fresh values. The module stays decoupled: it
    only calls cache.get_cached_prediction / cache.upsert_prediction if provided.
    """
    end_date = datetime.strptime(config["forecast_start_date"], "%Y-%m-%d")
    print(f"FORECAST_DATE: {end_date}")

    # Keep the full cached_dataset out of LangSmith trace payloads (else one
    # prediction's trace exceeds the 20MB/run ingest limit and is dropped).
    _configure_tracing_redaction()

    # One run-scoped folder for all agent prompt/response debug dumps.
    debug_run_dir: str | None = None
    if config.get("debug_save_prompts", True):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_run_dir = str(Path(__file__).resolve().parent / "debug_prompts" / f"run_{ts}")
        print(f"[predictions] Saving agent prompts to {debug_run_dir}")

    if save_results and save_path:
        Path(save_path).unlink(missing_ok=True)

    cache_enabled = cache is not None and bool(config_hash)

    # Ordered list of dates this run must produce (newest first, как раньше).
    forecast_dates = [
        (end_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(last_days)
    ]

    # Resolve cache hits up front: if every requested date is already cached we can
    # skip the heavy SharedBaseDataCache/CoinGlass build entirely. force_recompute
    # ignores existing entries but still overwrites them with fresh values below.
    cached_rows: dict[str, dict] = {}
    if cache_enabled and not force_recompute:
        for d in forecast_dates:
            hit = cache.get_cached_prediction(config_hash, d)
            if hit is not None:
                cached_rows[d] = hit
        if cached_rows:
            print(
                f"[predictions] Cache: {len(cached_rows)}/{len(forecast_dates)} "
                f"dates served from cache (config {config_hash})"
            )

    missing_dates = [d for d in forecast_dates if d not in cached_rows]

    active_agents = set(config.get("agent_envolved_in_prediction", []))

    # Agents which use tech indicators
    _DATASET_AGENTS = {
        "agent_for_analysing_tech_indicators",
        "agent_for_analysing_onchain_indicators",
    }

    # Build the shared dataset only when a dataset-dependent agent has at least one
    # date that is NOT already cached — all-hits runs never touch CoinGlass.
    needs_dataset = bool(_DATASET_AGENTS & active_agents) and bool(missing_dates)
    cached_dataset: pd.DataFrame | None = None
    if needs_dataset:
        api_key = os.environ["COINGLASS_API_KEY"]
        print("[predictions] Building SharedBaseDataCache...")
        base_cache = SharedBaseDataCache(api_key=api_key)
        cached_dataset = base_cache.get_base_df()
        if cached_dataset is None or cached_dataset.empty:
            raise RuntimeError(
                "[predictions] SharedBaseDataCache returned empty dataframe."
            )
        print(
            f"[predictions] SharedBaseDataCache loaded: {cached_dataset.shape} "
            f"({cached_dataset['date'].min().date()} → {cached_dataset['date'].max().date()})"
        )
    elif not missing_dates:
        print("[predictions] All requested dates served from cache — skipping CoinGlass fetch")
    else:
        print("[predictions] No dataset-dependent agents — skipping CoinGlass fetch")

    rows = []
    with _batch_trace(config, last_days):
        for i, forecast_date in enumerate(forecast_dates):
            print(f"\n{'='*60}")
            print(f"[predictions] Day {i + 1}/{last_days} — forecast_date={forecast_date}")
            print(f"{'='*60}")

            cached_row = cached_rows.get(forecast_date)
            if cached_row is not None:
                print(f"[predictions] CACHE HIT — {forecast_date}")
                rows.append(cached_row)
                continue

            print("DATE PREDICT:", forecast_date)
            row = make_one_prediction(
                app,
                config,
                forecast_date,
                cached_dataset,
                save_results=save_results,
                save_path=save_path,
                debug_run_dir=debug_run_dir,
            )

            if cache_enabled:
                cache.upsert_prediction(config_hash, config, forecast_date, row)

            rows.append(row)

            # UPDATE CONFUSION MATRIX AFTER N PREDICTS
            if (
                checkpoint_every > 0
                and cm_path is not None
                and (i + 1) % checkpoint_every == 0
            ):
                partial = add_y_true(pd.DataFrame(rows), config["horizon"])
                print(f"[predictions] Checkpoint CM after {i + 1}/{last_days} predictions...")
                build_confusion_matrix(partial, config["horizon"], cm_path)

    return pd.DataFrame(rows)


def _fetch_bybit_btc_spot_via_coinglass() -> pd.DataFrame:
    """Fetch BTCUSDT daily spot data from CoinGlass with exchange=Bybit.

    Returns indexed DataFrame with:
    - close: daily close price
    - high: daily high price
    - low: daily low price

    Single endpoint call — does not load the full SharedBaseDataCache pipeline.
    """
    api_key = os.environ["COINGLASS_API_KEY"]
    getter = FeaturesGetter(api_key=api_key)
    df = getter.get_history(
        endpoint_name="spot_price_history",
        exchange="Bybit",
        symbol="BTCUSDT",
        interval="1d",
        prefix="spot",
    )
    if df.empty:
        raise RuntimeError("CoinGlass returned empty spot_price_history for Bybit BTCUSDT")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").set_index("date")

    close = df["spot__close"].astype(float)
    open_ = df["spot__open"].astype(float) if "spot__open" in df.columns else pd.Series(index=df.index, dtype=float)
    high = df["spot__high"].astype(float) if "spot__high" in df.columns else pd.Series(index=df.index, dtype=float)
    low = df["spot__low"].astype(float) if "spot__low" in df.columns else pd.Series(index=df.index, dtype=float)

    return pd.DataFrame(
        {
            "close": close,
            "open": open_,
            "high": high,
            "low": low,
        }
    ).sort_index()


def add_y_true(
    df: pd.DataFrame,
    horizon: int,
    close_col: str = "spot_price_history__close",
) -> pd.DataFrame:
    """Добавляет колонку y_true: реальное направление BTC через horizon дней.

    Источник цен — Bybit BTCUSDT spot через CoinGlass (один эндпоинт
    /spot/price/history, без полного SharedBaseDataCache).
    """
    if df.empty:
        df = df.copy()
        df["start_date_price"] = None
        df["btc_bybit_close_price"] = None
        df["btc_bybit_high_price"] = None
        df["btc_bybit_low_price"] = None
        df["horizon_close_price"] = None
        df["horizon_max_high"] = None
        df["horizon_min_low"] = None
        df["y_true"] = None
        return df

    spot_data = _fetch_bybit_btc_spot_via_coinglass()

    open_values = []
    close_now_values = []
    high_values = []
    low_values = []
    horizon_close_values = []
    horizon_max_high_values = []
    horizon_min_low_values = []
    y_true = []
    for _, row in df.iterrows():
        forecast_date = pd.Timestamp(row["forecast_start_date"]).normalize()
        target_date = forecast_date + timedelta(days=horizon)

        price_now = (
            spot_data.loc[forecast_date, "close"]
            if forecast_date in spot_data.index
            else None
        )
        price_then = (
            spot_data.loc[target_date, "close"]
            if target_date in spot_data.index
            else None
        )
        open_price = (
            spot_data.loc[forecast_date, "open"]
            if forecast_date in spot_data.index
            else None
        )
        high_price = (
            spot_data.loc[forecast_date, "high"]
            if forecast_date in spot_data.index
            else None
        )
        low_price = (
            spot_data.loc[forecast_date, "low"]
            if forecast_date in spot_data.index
            else None
        )
        open_values.append(float(open_price) if pd.notna(open_price) else None)
        close_now_values.append(float(price_now) if price_now is not None else None)
        high_values.append(float(high_price) if pd.notna(high_price) else None)
        low_values.append(float(low_price) if pd.notna(low_price) else None)

        # OHLC over the prediction window (forecast_date+1 .. forecast_date+horizon)
        window_dates = [forecast_date + timedelta(days=i) for i in range(1, horizon + 1)]
        if all(d in spot_data.index for d in window_dates):
            window_high = spot_data.loc[window_dates, "high"]
            window_low = spot_data.loc[window_dates, "low"]
            horizon_max_high_values.append(float(window_high.max()) if window_high.notna().all() else None)
            horizon_min_low_values.append(float(window_low.min()) if window_low.notna().all() else None)
            horizon_close_values.append(float(price_then) if price_then is not None else None)
        else:
            horizon_max_high_values.append(None)
            horizon_min_low_values.append(None)
            horizon_close_values.append(None)

        if price_now is None or price_then is None:
            y_true.append(None)
        else:
            y_true.append("LONG" if float(price_then) > float(price_now) else "SHORT")

    df = df.copy()
    df["start_date_price"] = open_values
    df["btc_bybit_close_price"] = close_now_values
    df["btc_bybit_high_price"] = high_values
    df["btc_bybit_low_price"] = low_values
    df["horizon_close_price"] = horizon_close_values
    df["horizon_max_high"] = horizon_max_high_values
    df["horizon_min_low"] = horizon_min_low_values
    df["y_true"] = y_true
    return df


def build_confusion_matrix(results_df: pd.DataFrame, horizon: int, output_path: Path) -> None:
    """Compare predicted LONG/SHORT against actual BTC price movement.

    Pulls BTCUSDT spot daily close from Bybit (via add_y_true) and for each
    forecast_start_date checks whether close[date + horizon] > close[date].
    Saves a confusion matrix plot to output_path.
    """
    # Lazy imports — keep matplotlib/sklearn out of the API runtime image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Если y_true ещё не посчитан — считаем на месте
    if "y_true" not in results_df.columns:
        results_df = add_y_true(results_df, horizon)

    # Оставляем только строки с валидными прогнозом и y_true
    valid = results_df[
        results_df["y_predict"].isin(["LONG", "SHORT"]) &
        results_df["y_true"].isin(["LONG", "SHORT"])
    ]

    if valid.empty:
        print("[confusion_matrix] Not enough matched dates to build confusion matrix")
        return

    actuals     = valid["y_true"].tolist()      # true_y: реальное направление BTC
    predictions = valid["y_predict"].tolist()   # predict_y: прогноз мультиагентной системы

    # --- Строим матрицу ошибок: строки = true_y, столбцы = predict_y ---
    labels = ["LONG", "SHORT"]
    cm = confusion_matrix(actuals, predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # --- Рисуем и сохраняем график ---
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    accuracy = sum(a == p for a, p in zip(actuals, predictions)) / len(actuals)
    ax.set_title(f"Multiagent predictions  |  horizon={horizon}d  |  n={len(actuals)}  |  acc={accuracy:.1%}")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[confusion_matrix] Saved → {output_path}  (n={len(actuals)}, acc={accuracy:.1%})")