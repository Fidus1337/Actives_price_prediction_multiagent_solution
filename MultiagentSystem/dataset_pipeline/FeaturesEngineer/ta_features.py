import pandas as pd
import ta


def add_ta_features_selected(df: pd.DataFrame, prefix: str, volume_col_override: str | None = None) -> pd.DataFrame:
    """Adds 8 curated TA indicators for an asset (2 per aspect).

    Trend:      ADX (trend strength), CCI (deviation from mean)
    Momentum:   RSI (overbought/oversold), ROC (rate of change)
    Volatility: ATR (average true range), BBW (Bollinger bandwidth)
    Volume:     OBV (on-balance volume), MFI (money flow index)

    Max lookback = 20 bars.
    """
    df = df.copy()

    col_map = {col: f"{prefix}__{col}" for col in ['open', 'close', 'high', 'low', 'volume']}
    if volume_col_override:
        col_map['volume'] = volume_col_override

    missing = [col_map[c] for c in col_map if col_map[c] not in df.columns]
    if missing:
        print(f"  Missing columns for {prefix}: {missing}")
        return df

    h, l, c, v = df[col_map['high']], df[col_map['low']], df[col_map['close']], df[col_map['volume']]

    df[f"{prefix}__ta_adx"] = ta.trend.ADXIndicator(h, l, c).adx()
    df[f"{prefix}__ta_cci"] = ta.trend.CCIIndicator(h, l, c).cci()
    df[f"{prefix}__ta_rsi"] = ta.momentum.RSIIndicator(c).rsi()
    df[f"{prefix}__ta_roc"] = ta.momentum.ROCIndicator(c).roc()
    df[f"{prefix}__ta_atr"] = ta.volatility.AverageTrueRange(h, l, c).average_true_range()
    df[f"{prefix}__ta_bbw"] = ta.volatility.BollingerBands(c).bollinger_wband()
    df[f"{prefix}__ta_obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df[f"{prefix}__ta_mfi"] = ta.volume.MFIIndicator(h, l, c, v).money_flow_index()

    print(f"  +8 TA features for {prefix}")
    return df


def add_ta_features_for_asset(df: pd.DataFrame, prefix: str, volume_col_override: str | None = None) -> pd.DataFrame:
    """Add TA indicators for an asset with the given prefix.

    Parameters:
        prefix: asset column prefix (e.g. "gold", "sp500", "spot_price_history")
        volume_col_override: full volume column name if it is not {prefix}__volume
                             (e.g. "spot_price_history__volume_usd" for BTC)
    """
    df = df.copy()

    required = ['open', 'close', 'high', 'low', 'volume']
    col_map = {col: f"{prefix}__{col}" for col in required}

    # Allow overriding the volume column name
    if volume_col_override:
        col_map['volume'] = volume_col_override

    missing = [col_map[c] for c in required if col_map[c] not in df.columns]
    if missing:
        print(f"  Missing columns for {prefix}: {missing}")
        return df

    temp_df = pd.DataFrame({
        'open': df[col_map['open']].values,
        'high': df[col_map['high']].values,
        'low': df[col_map['low']].values,
        'close': df[col_map['close']].values,
        'volume': df[col_map['volume']].values
    })

    temp_with_ta = ta.add_all_ta_features(
        temp_df,
        open="open", high="high", low="low", close="close", volume="volume",
        fillna=False
    )

    original_cols = {'open', 'high', 'low', 'close', 'volume'}
    ta_cols = [c for c in temp_with_ta.columns if c not in original_cols]

    for col in ta_cols:
        df.loc[df.index, f"{prefix}__{col}"] = temp_with_ta[col].values

    print(f"  Added {len(ta_cols)} TA features for {prefix}")
    return df
