"""
Shared base data cache for Predictor instances.

Caches the result of the expensive shared pipeline steps
(API fetch + merge + ffill + date filter + drop sparse + engineered features + TA)
that are identical for ALL models regardless of horizon or type.

Each Predictor then applies only model-specific target engineering
on top of a copy of this shared DataFrame.
"""

import json
import os
import threading
import time
from typing import Optional

import pandas as pd

from MultiagentSystem.dataset_pipeline.FeaturesGetterModule.FeaturesGetter import FeaturesGetter
from MultiagentSystem.dataset_pipeline.Dataset_builder_pipeline import get_features
from MultiagentSystem.dataset_pipeline.FeaturesGetterModule.helpers._merge_features_by_date import merge_by_date
from MultiagentSystem.dataset_pipeline.FeaturesEngineer.FeaturesEngineer import FeaturesEngineer
from MultiagentSystem.dataset_pipeline.FeaturesEngineer.ta_features import add_ta_features_selected


class SharedBaseDataCache:
    """
    Thread-safe cache for the base DataFrame shared across all Predictor instances.

    The base DataFrame includes all pipeline steps up to (and including) TA indicators,
    but EXCLUDES model-specific target columns (y_up_Nd, range targets).
    """

    def __init__(self, api_key: str, ttl_seconds: float = 3600.0):
        self._api_key = api_key
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._base_df: Optional[pd.DataFrame] = None
        self._fetched_at: float = 0.0
        self._getter = FeaturesGetter(api_key=api_key)
        self._features_engineer = FeaturesEngineer()

    @property
    def is_stale(self) -> bool:
        if self._base_df is None:
            return True
        return (time.time() - self._fetched_at) > self._ttl_seconds

    def get_base_df(self) -> pd.DataFrame:
        """
        Return a COPY of the cached base DataFrame, refreshing if stale.

        Always returns a copy so callers can mutate freely
        (e.g., adding target columns) without corrupting the shared cache.
        """
        if self.is_stale:
            with self._lock:
                # Double-check after acquiring lock
                if self.is_stale:
                    self._base_df = self._fetch_base_data()
                    self._fetched_at = time.time()
        return self._base_df.copy()

    def clear(self) -> None:
        """Clear the cached base DataFrame (e.g., after retraining)."""
        with self._lock:
            self._base_df = None
            self._fetched_at = 0.0

    def refresh(self) -> None:
        """Force-refresh base data from API (called on every /api/predictions)."""
        print("SharedBaseDataCache: Refreshing data...")
        with self._lock:
            self._base_df = self._fetch_base_data()
            self._fetched_at = time.time()
        print(f"SharedBaseDataCache: Refreshed. Shape: {self._base_df.shape}")

    @staticmethod
    def _trim_to_longest_continuous_segment(df: pd.DataFrame) -> pd.DataFrame:
        """
        Find the longest run of consecutive calendar days and return only that segment.
        Drops rows that belong to shorter segments separated by date gaps.
        """
        dates = pd.to_datetime(df["date"]).sort_values()
        diffs = dates.diff()
        is_gap = diffs > pd.Timedelta(days=1)
        group_id = is_gap.cumsum()
        largest_group = group_id.value_counts().idxmax()
        mask = group_id == largest_group

        trimmed = df.loc[mask].reset_index(drop=True)

        dropped = len(df) - len(trimmed)
        if dropped > 0:
            print(
                f"SharedBaseDataCache: Date continuity check — dropped {dropped} rows with gaps. "
                f"Kept {len(trimmed)} consecutive days "
                f"({trimmed['date'].iloc[0]} to {trimmed['date'].iloc[-1]})"
            )

        print(f"SharedBaseDataCache: Last date in dataset: {trimmed['date'].iloc[-1]}")

        return trimmed

    # Columns with too many NaN (data starts much later than other sources)
    _SPARSE_COLUMNS = [
        # Orderbook: ~337 NaN out of 1000 rows
        'futures_orderbook_aggregated_ask_bids_history__aggregated_asks_usd',
        'futures_orderbook_aggregated_ask_bids_history__aggregated_bids_usd',
        'futures_orderbook_aggregated_ask_bids_history__aggregated_bids_quantity',
        'futures_orderbook_ask_bids_history__asks_quantity',
        'futures_orderbook_ask_bids_history__asks_usd',
        'futures_orderbook_ask_bids_history__bids_quantity',
        'futures_orderbook_ask_bids_history__bids_usd',
        'futures_orderbook_aggregated_ask_bids_history__aggregated_asks_quantity',
        # CGDI index: ~222 NaN
        'cgdi_dev_from_base',
        'cgdi_log_level',
        'cgdi_index_value',
        'cgdi_dev_softsign',
    ]

    _DATE_WINDOW_DAYS = 1000
    _LAG_PERIODS = [1, 3, 5, 7, 15]

    def _fetch_base_data(self) -> pd.DataFrame:
        """
        Execute shared pipeline (identical for all models).

        1. get_features() + merge_by_date() + sort
        2. ensure_spot_prefix()
        3. ffill()
        4. Date filter (last _DATE_WINDOW_DAYS days)
        5. Drop sparse columns + re-ffill + dropna
        6. add_engineered_features()  -> diff1, pct1, imbalances
        7. add_ta_features_selected() x4  -> 8 TA indicators per asset
        8. Add lag features (1, 3, 5, 7, 15 days) for base columns
        9. _trim_to_longest_continuous_segment()
        """
        # 1. Raw data
        print("SharedBaseDataCache: Fetching features from API...")
        dfs = get_features(self._getter, self._api_key)
        df = merge_by_date(dfs, how="outer", dedupe="last")
        df = df.sort_values("date").reset_index(drop=True)
        print(f"SharedBaseDataCache: Raw data: {df.shape}")

        # 2. Normalize spot columns
        df = self._features_engineer.ensure_spot_prefix(df)

        # 3. Forward-fill gaps (weekends/holidays)
        feature_cols = [c for c in df.columns if c != "date"]
        df[feature_cols] = df[feature_cols].ffill()

        # 4. Date filter
        df['date'] = pd.to_datetime(df['date'])
        cutoff = df['date'].max() - pd.Timedelta(days=self._DATE_WINDOW_DAYS)
        df = df[df['date'] >= cutoff]

        # 5. Drop sparse columns + clean up remaining NaN
        cols_to_drop = [c for c in self._SPARSE_COLUMNS if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        feature_cols = [c for c in df.columns if c != "date"]
        df[feature_cols] = pd.DataFrame(df[feature_cols]).ffill()
        df = pd.DataFrame(df.dropna())

        # 6. Engineered features (diff1, pct1, imbalances)
        df = self._features_engineer.add_engineered_features(df)

        # 6.1. Price MA features (SMA 7/14/21/50, relative return, z-score)
        df = self._features_engineer.add_price_ma_features(df)

        # 7. TA indicators (8 per asset: ADX, CCI, RSI, ROC, ATR, BBW, OBV, MFI)
        df = add_ta_features_selected(df, prefix="gold")
        df = add_ta_features_selected(df, prefix="sp500")
        df = add_ta_features_selected(df, prefix="igv")
        df = add_ta_features_selected(
            df, prefix="spot_price_history",
            volume_col_override="spot_price_history__volume_usd"
        )

        # 8. Lag features for base columns (excluding diff1/pct1 derivatives)
        base_cols = [
            c for c in df.columns
            if c != "date" and "__pct1" not in c and "__diff1" not in c
        ]
        cols_before = df.shape[1]
        lag_frames = []
        for lag in self._LAG_PERIODS:
            lagged = df[base_cols].shift(lag)
            lagged.columns = [f"{col}__lag{lag}" for col in base_cols]
            lag_frames.append(lagged)
        df = pd.concat([df] + lag_frames, axis=1)
        print(f"SharedBaseDataCache: Added lags {self._LAG_PERIODS}: "
              f"{cols_before} -> {df.shape[1]} columns (+{df.shape[1] - cols_before})")

        # 9. Drop NaN rows first (lag/TA lookbacks), THEN trim to longest segment —
        # otherwise dropna creates gaps that trim can no longer clean up.
        df = df.dropna()
        df = self._trim_to_longest_continuous_segment(df)

        # Save available features list to Logs/
        feature_names = sorted([c for c in df.columns if c != "date"])
        os.makedirs("Logs", exist_ok=True)
        features_path = os.path.join("Logs", "available_features.json")
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump({"features": feature_names, "count": len(feature_names)}, f, indent=2, ensure_ascii=False)
        print(f"SharedBaseDataCache: Saved {len(feature_names)} features to available_features.json")

        print(f"SharedBaseDataCache: Base data ready. Shape: {df.shape}")
        return df
