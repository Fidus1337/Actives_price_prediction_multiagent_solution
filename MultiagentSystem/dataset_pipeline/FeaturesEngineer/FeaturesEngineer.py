import numpy as np
import pandas as pd

class FeaturesEngineer:
    
    # ---------- 1) Spot column normalization ----------
    def ensure_spot_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        mapping = {
            "open": "spot_price_history__open",
            "high": "spot_price_history__high",
            "low": "spot_price_history__low",
            "close": "spot_price_history__close",
            "volume_usd": "spot_price_history__volume_usd",
        }
        # Rename only if the target prefixed column does not exist yet.
        rename = {}
        for old, new in mapping.items():
            if old in out.columns and new not in out.columns:
                rename[old] = new
        if rename:
            out = out.rename(columns=rename)
        return out
    
    # ---------- 3) Binary target for a custom prediction horizon ----------
    def add_y_up_custom(self, df: pd.DataFrame, horizon: int, close_col: str = "spot_price_history__close") -> pd.DataFrame:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("date", kind="stable").reset_index(drop=True)
    
        target_column_name = f"y_up_{horizon}d"
        c = pd.to_numeric(out[close_col], errors="coerce")
        out[target_column_name] = (c.shift(-horizon) > c).astype("Int64")
        return out
    
        # ---------- 3) Feature engineering: diff/pct + imbalance ----------
    def add_engineered_features(self, df: pd.DataFrame, horizon=1) -> pd.DataFrame:
        out = df.copy()
        eps = 1e-12
        target_column_name = f"y_up_{horizon}d"

        # diff/pct for all numeric columns (excluding target)
        base_numeric = [
            c for c in out.columns
            if c not in {"date", target_column_name}
            and pd.api.types.is_numeric_dtype(out[c])
        ]
        new_cols = {}
        for c in base_numeric:
            new_cols[c + "__diff1"] = out[c].diff(1)
            pct1 = out[c].pct_change(1)
            # Volume columns are 0 on weekends/holidays → 0/0 = NaN, 0→X = inf.
            # Treat both as "no change" (0.0) to avoid dropping non-trading days.
            pct1 = pct1.replace([np.inf, -np.inf], 0.0)
            # Keep first-row NaN (legitimate "no previous value") — it gets dropped later anyway.
            # Fill only interior NaN (from 0/0) with 0.
            if len(pct1) > 1:
                pct1.iloc[1:] = pct1.iloc[1:].fillna(0.0)
            new_cols[c + "__pct1"] = pct1
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

        # Imbalances (if both columns in a pair exist)
        def _imbalance(num_col_a, num_col_b, new_col):
            if num_col_a in out.columns and num_col_b in out.columns:
                a = pd.to_numeric(out[num_col_a], errors="coerce")
                b = pd.to_numeric(out[num_col_b], errors="coerce")
                out[new_col] = (a - b) / (a + b + eps)

        _imbalance(
            "futures_v2_taker_buy_sell_volume_history__taker_buy_volume_usd",
            "futures_v2_taker_buy_sell_volume_history__taker_sell_volume_usd",
            "feat__taker_imbalance_v2",
        )
        _imbalance(
            "futures_aggregated_taker_buy_sell_volume_history__aggregated_buy_volume_usd",
            "futures_aggregated_taker_buy_sell_volume_history__aggregated_sell_volume_usd",
            "feat__taker_imbalance_agg",
        )
        _imbalance(
            "futures_liquidation_history__short_liquidation_usd",
            "futures_liquidation_history__long_liquidation_usd",
            "feat__liq_imbalance_short_minus_long",
        )
        _imbalance(
            "futures_orderbook_ask_bids_history__bids_usd",
            "futures_orderbook_ask_bids_history__asks_usd",
            "feat__orderbook_imbalance_usd",
        )

        # Narrow 1d rebuild features: only a few short-horizon signals
        # that are not covered by generic diff/pct transforms above.
        if {
            "spot_price_history__high",
            "spot_price_history__low",
            "spot_price_history__close",
        }.issubset(out.columns):
            high = pd.to_numeric(out["spot_price_history__high"], errors="coerce")
            low = pd.to_numeric(out["spot_price_history__low"], errors="coerce")
            close = pd.to_numeric(out["spot_price_history__close"], errors="coerce")
            intraday_range = high - low
            out["spot_price_history__intraday_range_pct"] = intraday_range / (close + eps)
            out["spot_price_history__close_to_high"] = (high - close) / (close + eps)
            out["spot_price_history__close_to_low"] = (close - low) / (close + eps)

            close_ret1 = close.pct_change(1)
            out["spot_price_history__realized_vol_3d"] = close_ret1.rolling(3).std()
            out["spot_price_history__realized_vol_7d"] = close_ret1.rolling(7).std()

        if {
            "futures_open_interest_aggregated_history__close",
            "spot_price_history__volume_usd",
        }.issubset(out.columns):
            oi_agg = pd.to_numeric(out["futures_open_interest_aggregated_history__close"], errors="coerce")
            spot_volume = pd.to_numeric(out["spot_price_history__volume_usd"], errors="coerce")
            out["feat__oi_to_volume"] = oi_agg / (spot_volume + eps)

        # feat__stablecoin_oi_share / feat__coin_margin_oi_share были удалены:
        # CoinGlass v4 возвращает `aggregated-stablecoin-history.close` и
        # `aggregated-coin-margin-history.close` в РАЗНЫХ единицах (одна — USD,
        # другая — нативные coin-количества), из-за чего shares получались почти
        # константами (~0.00007 / ~0.99993) и не несли сигнала.
        # Для scale-invariant сигнала по этим источникам используйте
        # `futures_open_interest_aggregated_stablecoin_history__close__pct1`
        # и `futures_open_interest_aggregated_coin_margin_history__close__pct1`,
        # которые автоматически генерируются блоком pct1 выше.

        if {
            "futures_funding_rate_history__close",
            "futures_funding_rate_oi_weight_history__close",
        }.issubset(out.columns):
            funding = pd.to_numeric(out["futures_funding_rate_history__close"], errors="coerce")
            funding_oi_weight = pd.to_numeric(
                out["futures_funding_rate_oi_weight_history__close"],
                errors="coerce",
            )
            out["feat__funding_minus_oi_weight"] = funding - funding_oi_weight

        if {
            "futures_liquidation_history__long_liquidation_usd",
            "futures_liquidation_history__short_liquidation_usd",
        }.issubset(out.columns):
            long_liq = pd.to_numeric(out["futures_liquidation_history__long_liquidation_usd"], errors="coerce")
            short_liq = pd.to_numeric(out["futures_liquidation_history__short_liquidation_usd"], errors="coerce")
            liq_total = long_liq + short_liq
            liq_total_pct1 = liq_total.pct_change(1).replace([np.inf, -np.inf], 0.0)
            if len(liq_total_pct1) > 1:
                liq_total_pct1.iloc[1:] = liq_total_pct1.iloc[1:].fillna(0.0)
            out["feat__liq_total_usd"] = liq_total
            out["feat__liq_total_pct1"] = liq_total_pct1

        # Clean up infinities
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    # ---------- 4) SMA, relative change, Z-score ----------
    def add_price_ma_features(
        self,
        df: pd.DataFrame,
        close_col: str = "spot_price_history__close",
        windows: list = [7, 14, 21, 50],
    ) -> pd.DataFrame:
        out = df.copy()
        if close_col not in out.columns:
            return out

        close = out[close_col]
        new_cols = {}
        for w in windows:
            sma = close.rolling(w).mean()
            std = close.rolling(w).std()
            new_cols[f"{close_col}__sma{w}"]     = sma
            new_cols[f"{close_col}__sma{w}_rel"] = close / sma - 1
            new_cols[f"{close_col}__zscore{w}"]  = (close - sma) / std

        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
        return out