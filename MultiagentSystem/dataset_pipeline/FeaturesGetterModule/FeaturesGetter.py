import pandas as pd
import numpy as np
import os
import json
import time
import threading
import io
import re
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from dotenv import load_dotenv
import yfinance as yf

## HELPER FUNCTIONS - support both module and direct execution
try:
    from .helpers._coinglass_get_dataframe import _coinglass_get_dataframe, CoinGlassError
    from .helpers._coinglass_normalize_time_to_date import _coinglass_normalize_time_to_date
    from .helpers._prefix_columns import _prefix_columns
except ImportError:
    from helpers._coinglass_get_dataframe import _coinglass_get_dataframe, CoinGlassError
    from helpers._coinglass_normalize_time_to_date import _coinglass_normalize_time_to_date
    from helpers._prefix_columns import _prefix_columns

# Load endpoints config from JSON
_ENDPOINTS_PATH = Path(__file__).parent / "features_endpoints.json"
with open(_ENDPOINTS_PATH, "r", encoding="utf-8") as f:
    ENDPOINTS = json.load(f)

_YFINANCE_LOCK = threading.Lock()
_YFINANCE_CACHE_DIR = Path(__file__).parent / "yfinance_cache"


class FeaturesGetter:
    """
    Класс для получения исторических данных с CoinGlass API.
    
    Attributes:
        api_key: API ключ CoinGlass
    
    Example:
        >>> getter = FeaturesGetter(api_key="your_api_key")
        >>> df = getter.get_history("open_interest_history")
    """
    
    def __init__(self, api_key: str):
        """
        Инициализирует FeaturesGetter с API ключом.
        
        Args:
            api_key: API ключ CoinGlass
        """
        self.api_key = api_key

    @staticmethod
    def _safe_yfinance_history(
        symbol: str,
        days: int,
        interval: str = "1d",
        attempts: int = 4,
        base_sleep_seconds: float = 1.0,
    ) -> pd.DataFrame:
        """
        Robust wrapper around yfinance history() to handle transient
        DNS/network/provider errors without crashing the whole pipeline.
        """
        last_exc: Exception | None = None
        _YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe_symbol = re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol)
        cache_path = _YFINANCE_CACHE_DIR / f"{safe_symbol}_{interval}.csv"

        for attempt in range(1, attempts + 1):
            try:
                # yfinance can intermittently fail under concurrent calls.
                # Keep requests serialized for stability in multi-threaded pipelines.
                with _YFINANCE_LOCK:
                    history_exc: Exception | None = None
                    try:
                        ticker = yf.Ticker(symbol)
                        df = ticker.history(period=f"{days}d", interval=interval, timeout=20)
                    except Exception as exc:
                        history_exc = exc
                        df = pd.DataFrame()

                    if df is None or df.empty:
                        # Fallback path: sometimes Ticker.history fails while download works.
                        try:
                            # yfinance may print noisy internal errors even when retries succeed.
                            # Suppress raw downloader output and keep only our controlled logs.
                            with io.StringIO() as _buf_out, io.StringIO() as _buf_err:
                                with redirect_stdout(_buf_out), redirect_stderr(_buf_err):
                                    df = yf.download(
                                        symbol,
                                        period=f"{days}d",
                                        interval=interval,
                                        progress=False,
                                        threads=False,
                                        auto_adjust=False,
                                        timeout=20,
                                    )
                        except Exception as download_exc:
                            if history_exc is not None:
                                raise RuntimeError(
                                    f"history failed: {history_exc}; download failed: {download_exc}"
                                ) from download_exc
                            raise
                if df is None or df.empty:
                    raise RuntimeError(f"Empty response from yfinance for {symbol}")
                try:
                    df.to_csv(cache_path)
                except Exception as cache_exc:
                    print(f"[FeaturesGetter] Could not write yfinance cache for {symbol}: {cache_exc}")
                return df
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    sleep_s = base_sleep_seconds * (2 ** (attempt - 1))
                    print(
                        f"[FeaturesGetter] yfinance fetch failed for {symbol} "
                        f"(attempt {attempt}/{attempts}): {exc}. Retrying in {sleep_s:.1f}s..."
                    )
                    time.sleep(sleep_s)
                else:
                    print(
                        f"[FeaturesGetter] yfinance fetch failed for {symbol} "
                        f"after {attempts} attempts: {exc}. Returning empty DataFrame."
                    )
        if cache_path.exists():
            try:
                cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if not cached_df.empty:
                    if cached_df.index.name is None:
                        cached_df.index.name = "Date"
                    print(
                        f"[FeaturesGetter] Using cached yfinance data for {symbol}: "
                        f"{cached_df.shape[0]} rows from {cache_path.name}"
                    )
                    return cached_df
            except Exception as cache_read_exc:
                print(f"[FeaturesGetter] Failed to read yfinance cache for {symbol}: {cache_read_exc}")
        _ = last_exc
        return pd.DataFrame()
    
    def get_history(
        self,
        endpoint_name: str,
        prefix: str | None = None,
        limit: int = 1250,
        **params,
    ) -> pd.DataFrame:
        """
        Получает исторические данные с CoinGlass API.

        Args:
            endpoint_name: Имя эндпоинта из ENDPOINTS (например, "open_interest_history")
            prefix: Префикс для колонок (по умолчанию = endpoint_name)
            limit: Максимальное количество записей (по умолчанию 1250)
            **params: Параметры запроса (exchange, symbol, interval и т.д.)
                      Если не указаны, используются default_params из конфига.

        Returns:
            DataFrame с колонкой date и данными с префиксами.

        Raises:
            ValueError: Если endpoint_name не найден в ENDPOINTS
            CoinGlassError: При ошибках API
        """
        params["limit"] = limit
        if endpoint_name not in ENDPOINTS:
            available = ", ".join(sorted(ENDPOINTS.keys()))
            raise ValueError(f"Unknown endpoint: '{endpoint_name}'. Available: {available}")
        
        cfg = ENDPOINTS[endpoint_name]
        
        # Merge default params with user params (user params override defaults)
        request_params = {**cfg["default_params"], **params}
        
        # Fetch data
        df = _coinglass_get_dataframe(
            endpoint=cfg["path"],
            api_key=self.api_key,
            params=request_params,
        )
        
        if df.empty:
            return df
        
        # time -> date
        if "time" in df.columns:
            df["date"] = _coinglass_normalize_time_to_date(df["time"])
            df = df.drop(columns=["time"])
        else:
            df["date"] = pd.NA
        
        # Convert all non-date columns to numeric
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Add prefix
        if prefix is None:
            prefix = endpoint_name
        
        df = _prefix_columns(df, prefix=prefix, keep=("date",))
        
        return df
    
    @staticmethod
    def list_endpoints() -> list[str]:
        """Возвращает список доступных эндпоинтов."""
        return sorted(ENDPOINTS.keys())
    
    def get_bitcoin_lth_supply(
        self,
        pct_window: int = 30,
        z_window: int = 180,
        slope_window: int = 14,
        prefix: str = "index_btc_lth_supply",
    ) -> pd.DataFrame:
        """
        Bitcoin Long-Term Holder Supply с расчётными фичами для прогнозирования.
        
        Args:
            pct_window: Окно для процентного изменения (дней)
            z_window: Окно для z-score (дней)
            slope_window: Окно для slope/velocity (дней)
            prefix: Префикс для колонок
        
        Returns:
            DataFrame с колонками:
              - date
              - {prefix}__price
              - {prefix}__lth_supply
              - {prefix}__supply_pct{pct_window}
              - {prefix}__supply_z{z_window}
              - {prefix}__supply_slope{slope_window}
        """
        df = _coinglass_get_dataframe(
            endpoint="/index/bitcoin-long-term-holder-supply",
            api_key=self.api_key,
        )
        
        if df.empty:
            return df
        
        # timestamp -> date (этот эндпоинт использует timestamp, а не time)
        if "timestamp" in df.columns:
            df["date"] = _coinglass_normalize_time_to_date(df["timestamp"])
            df = df.drop(columns=["timestamp"])
        
        # Нормализация числовых колонок
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
        df["lth_supply"] = pd.to_numeric(df.get("long_term_holder_supply"), errors="coerce")
        
        # Очистка и сортировка
        df = (
            df[["date", "price", "lth_supply"]]
            .dropna(subset=["date"])
            .sort_values("date", kind="stable")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        
        s = df["lth_supply"].astype(float)
        
        # Feature 1: pct change over N days (supply expansion/contraction proxy)
        df[f"supply_pct{pct_window}"] = s / s.shift(pct_window) - 1.0
        
        # Feature 2: rolling z-score (regime-normalized supply)
        minp = max(30, z_window // 3)
        roll = s.rolling(z_window, min_periods=minp)
        mu = roll.mean()
        sd = roll.std(ddof=0).replace(0.0, np.nan)
        df[f"supply_z{z_window}"] = (s - mu) / sd
        
        # Feature 3: slope / velocity
        df[f"supply_slope{slope_window}"] = s.diff(slope_window) / float(slope_window)
        
        # Префикс
        df = _prefix_columns(df, prefix=prefix, keep=("date",))
        
        return df

    def get_bitcoin_active_addresses(
        self,
        pct_window: int = 7,
        z_window: int = 180,
        slope_window: int = 14,
        prefix: str = "index_btc_active_addresses",
    ) -> pd.DataFrame:
        """
        Bitcoin Active Addresses с расчётными фичами для прогнозирования.
        
        Args:
            pct_window: Окно для процентного изменения (дней)
            z_window: Окно для z-score (дней)
            slope_window: Окно для slope/velocity (дней)
            prefix: Префикс для колонок
        
        Returns:
            DataFrame с колонками:
              - date
              - {prefix}__price
              - {prefix}__active_address_count
              - {prefix}__aa_pct{pct_window}
              - {prefix}__aa_z{z_window}
              - {prefix}__aa_slope{slope_window}
        """
        df = _coinglass_get_dataframe(
            endpoint="/index/bitcoin-active-addresses",
            api_key=self.api_key,
        )
        
        if df.empty:
            return df
        
        # timestamp -> date (этот эндпоинт использует timestamp, а не time)
        if "timestamp" in df.columns:
            df["date"] = _coinglass_normalize_time_to_date(df["timestamp"])
            df = df.drop(columns=["timestamp"])
        
        # Нормализация числовых колонок
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
        df["active_address_count"] = pd.to_numeric(df.get("active_address_count"), errors="coerce")
        
        # Очистка и сортировка
        df = (
            df[["date", "price", "active_address_count"]]
            .dropna(subset=["date"])
            .sort_values("date", kind="stable")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        
        aa = df["active_address_count"].astype(float)
        
        # Feature 1: short-horizon pct change (activity impulse)
        df[f"aa_pct{pct_window}"] = aa / aa.shift(pct_window) - 1.0
        
        # Feature 2: rolling z-score (regime normalized activity)
        minp = max(30, z_window // 3)
        roll = aa.rolling(z_window, min_periods=minp)
        mu = roll.mean()
        sd = roll.std(ddof=0).replace(0.0, np.nan)
        df[f"aa_z{z_window}"] = (aa - mu) / sd
        
        # Feature 3: slope / velocity
        df[f"aa_slope{slope_window}"] = aa.diff(slope_window) / float(slope_window)
        
        # Префикс
        df = _prefix_columns(df, prefix=prefix, keep=("date",))
        
        return df

    def get_bitcoin_sth_supply(
        self,
        pct_window: int = 30,
        z_window: int = 180,
        slope_window: int = 14,
        prefix: str = "index_btc_sth_supply",
    ) -> pd.DataFrame:
        """
        Bitcoin Short-Term Holder Supply с расчётными фичами для прогнозирования.
        
        Args:
            pct_window: Окно для процентного изменения (дней)
            z_window: Окно для z-score (дней)
            slope_window: Окно для slope/velocity (дней)
            prefix: Префикс для колонок
        
        Returns:
            DataFrame с колонками:
              - date
              - {prefix}__price
              - {prefix}__sth_supply
              - {prefix}__supply_pct{pct_window}
              - {prefix}__supply_z{z_window}
              - {prefix}__supply_slope{slope_window}
        """
        df = _coinglass_get_dataframe(
            endpoint="/index/bitcoin-short-term-holder-supply",
            api_key=self.api_key,
        )
        
        if df.empty:
            return df
        
        # timestamp -> date (этот эндпоинт использует timestamp, а не time)
        if "timestamp" in df.columns:
            df["date"] = _coinglass_normalize_time_to_date(df["timestamp"])
            df = df.drop(columns=["timestamp"])
        
        # Нормализация числовых колонок
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
        df["sth_supply"] = pd.to_numeric(df.get("short_term_holder_supply"), errors="coerce")
        
        # Очистка и сортировка
        df = (
            df[["date", "price", "sth_supply"]]
            .dropna(subset=["date"])
            .sort_values("date", kind="stable")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        
        s = df["sth_supply"].astype(float)
        
        # Feature 1: pct change over N days (distribution / accumulation proxy)
        df[f"supply_pct{pct_window}"] = s / s.shift(pct_window) - 1.0
        
        # Feature 2: rolling z-score (regime-normalized supply)
        minp = max(30, z_window // 3)
        roll = s.rolling(z_window, min_periods=minp)
        mu = roll.mean()
        sd = roll.std(ddof=0).replace(0.0, np.nan)
        df[f"supply_z{z_window}"] = (s - mu) / sd
        
        # Feature 3: slope / velocity
        df[f"supply_slope{slope_window}"] = s.diff(slope_window) / float(slope_window)
        
        # Префикс
        df = _prefix_columns(df, prefix=prefix, keep=("date",))
        
        return df

    def get_bitcoin_reserve_risk(
        self,
        z_window: int = 180,
        slope_window: int = 14,
        prefix: str = "index_btc_reserve_risk",
    ) -> pd.DataFrame:
        """
        Bitcoin Reserve Risk с расчётными фичами для прогнозирования.
        
        Reserve Risk = price / HODL Bank. Низкие значения = хорошее время для покупки
        (высокая уверенность HODLеров при низкой цене). Высокие значения = перегрев.
        
        Args:
            z_window: Окно для z-score (дней)
            slope_window: Окно для slope/velocity (дней)
            prefix: Префикс для колонок
        
        Returns:
            DataFrame с колонками:
              - date
              - {prefix}__price
              - {prefix}__reserve_risk_index
              - {prefix}__movcd (Market Value to Opportunity Cost Days)
              - {prefix}__hodl_bank (накопленный opportunity cost)
              - {prefix}__vocd (Value of Opportunity Cost Days)
              - {prefix}__log_rr (log-трансформация reserve risk)
              - {prefix}__rr_z{z_window} (z-score)
              - {prefix}__rr_slope{slope_window} (скорость изменения)
        """
        df = _coinglass_get_dataframe(
            endpoint="/index/bitcoin-reserve-risk",
            api_key=self.api_key,
        )
        
        if df.empty:
            return df
        
        # timestamp -> date (этот эндпоинт использует timestamp, а не time)
        if "timestamp" in df.columns:
            df["date"] = _coinglass_normalize_time_to_date(df["timestamp"])
            df = df.drop(columns=["timestamp"])
        
        # Нормализация числовых колонок
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
        df["reserve_risk_index"] = pd.to_numeric(df.get("reserve_risk_index"), errors="coerce")
        df["movcd"] = pd.to_numeric(df.get("movcd"), errors="coerce")
        df["hodl_bank"] = pd.to_numeric(df.get("hodl_bank"), errors="coerce")
        df["vocd"] = pd.to_numeric(df.get("vocd"), errors="coerce")
        
        # Очистка и сортировка
        df = (
            df[["date", "price", "reserve_risk_index", "movcd", "hodl_bank", "vocd"]]
            .dropna(subset=["date"])
            .sort_values("date", kind="stable")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        
        rr = df["reserve_risk_index"].astype(float)
        
        # Feature 1: log transform (reserve risk spans orders of magnitude)
        df["log_rr"] = np.log(np.where(rr > 0, rr, np.nan))
        
        # Feature 2: rolling z-score (regime-normalized)
        minp = max(30, z_window // 3)
        roll = rr.rolling(z_window, min_periods=minp)
        mu = roll.mean()
        sd = roll.std(ddof=0).replace(0.0, np.nan)
        df[f"rr_z{z_window}"] = (rr - mu) / sd
        
        # Feature 3: slope / velocity
        df[f"rr_slope{slope_window}"] = rr.diff(slope_window) / float(slope_window)
        
        # Префикс
        df = _prefix_columns(df, prefix=prefix, keep=("date",))
        
        return df

    def get_bitfinex_margin_long_short(
        self,
        symbol: str = "BTC",
        interval: str = "1d",
        eps: float = 1e-9,
        prefix: str = "bitfinex_margin_ls",
    ) -> pd.DataFrame:
        """
        Bitfinex Margin Long/Short positions с расчётными фичами.
        
        Args:
            symbol: Символ (BTC, ETH, etc.)
            interval: Интервал (1d, 4h, etc.)
            eps: Epsilon для избежания деления на ноль
            prefix: Префикс для колонок
        
        Returns:
            DataFrame с колонками:
              - date
              - {prefix}__long_quantity
              - {prefix}__short_quantity
              - {prefix}__total_qty (long + short)
              - {prefix}__long_share (long / total)
              - {prefix}__log_long_short (log ratio)
        """
        df = _coinglass_get_dataframe(
            endpoint="/bitfinex-margin-long-short",
            api_key=self.api_key,
            params={"symbol": symbol, "interval": interval},
        )
        
        if df.empty:
            return df
        
        # time -> date
        if "time" in df.columns:
            df["date"] = _coinglass_normalize_time_to_date(df["time"])
            df = df.drop(columns=["time"])
        
        # Нормализация числовых колонок
        df["long_quantity"] = pd.to_numeric(df.get("long_quantity"), errors="coerce")
        df["short_quantity"] = pd.to_numeric(df.get("short_quantity"), errors="coerce")
        
        # Очистка и сортировка
        df = (
            df[["date", "long_quantity", "short_quantity"]]
            .dropna(subset=["date"])
            .sort_values("date", kind="stable")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        
        lng = df["long_quantity"].astype(float)
        sht = df["short_quantity"].astype(float)
        total = lng + sht
        
        # Feature 1: total quantity
        df["total_qty"] = total
        
        # Feature 2: long share (0..1)
        df["long_share"] = lng / (total + eps)
        
        # Feature 3: log ratio (bias in log scale)
        df["log_long_short"] = np.log((lng + eps) / (sht + eps))
        
        # Префикс
        df = _prefix_columns(df, prefix=prefix, keep=("date",))
        
        return df

    def get_coinbase_premium_index(
        self,
        interval: str = "1d",
        eps: float = 1e-9,
        rate_is_percent: bool = True,
        prefix: str = "coinbase_premium",
    ) -> pd.DataFrame:
        """
        Coinbase Premium Index с расчётными фичами для прогнозирования.

        Coinbase Premium показывает разницу цены BTC на Coinbase vs другие биржи.
        Положительный premium = покупательский спрос со стороны US институционалов.

        Args:
            interval: Интервал (1d, 4h, etc.)
            eps: Epsilon для избежания деления на ноль
            rate_is_percent: True если premium_rate в процентах (0.17 = 0.17%),
                             False если уже в долях (0.0017 = 0.17%).
                             По наблюдениям, CoinGlass v4 возвращает rate в процентах.
            prefix: Префикс для колонок

        Returns:
            DataFrame с колонками:
              - date
              - {prefix}__premium (сырой premium в $)
              - {prefix}__premium_rate (rate как пришёл с API)
              - {prefix}__premium_abs (абсолютное значение)
              - {prefix}__premium_softsign (нормированный -1..1)
              - {prefix}__premium_rate_bps (rate в базисных пунктах)
              - {prefix}__implied_ref_price (расчётная референсная цена)
        """
        df = _coinglass_get_dataframe(
            endpoint="/coinbase-premium-index",
            api_key=self.api_key,
            params={"interval": interval},
        )
        
        if df.empty:
            return df
        
        # time -> date
        if "time" in df.columns:
            df["date"] = _coinglass_normalize_time_to_date(df["time"])
            df = df.drop(columns=["time"])
        
        # Нормализация числовых колонок
        df["premium"] = pd.to_numeric(df.get("premium"), errors="coerce")
        df["premium_rate"] = pd.to_numeric(df.get("premium_rate"), errors="coerce")
        
        # Очистка и сортировка
        df = (
            df[["date", "premium", "premium_rate"]]
            .dropna(subset=["date"])
            .sort_values("date", kind="stable")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        
        prem = df["premium"].astype(float)
        rate = df["premium_rate"].astype(float)
        aprem = np.abs(prem)
        
        # Feature 1: absolute premium
        df["premium_abs"] = aprem

        # Feature 2: softsign normalized premium (-1..1 bounded)
        df["premium_softsign"] = prem / (1.0 + aprem + eps)

        # Feature 3: premium rate in basis points.
        # CoinGlass v4 отдаёт rate в процентах (0.17 = 0.17%), поэтому множитель = 100
        # (1% = 100 bps). Если API когда-то перейдёт на доли — выставить rate_is_percent=False.
        if rate_is_percent:
            df["premium_rate_bps"] = rate * 100.0
        else:
            df["premium_rate_bps"] = rate * 10_000.0

        # Feature 4: implied reference price (обратный расчёт цены из premium и rate).
        # premium = ref_price * (rate / 100)  если rate в процентах -> ref = premium * 100 / rate.
        # premium = ref_price * rate          если rate в долях      -> ref = premium / rate.
        if rate_is_percent:
            df["implied_ref_price"] = prem * 100.0 / (rate + eps)
        else:
            df["implied_ref_price"] = prem / (rate + eps)
        
        # Префикс
        df = _prefix_columns(df, prefix=prefix, keep=("date",))
        
        return df

    def get_cgdi_index(
        self,
        interval: str = "1d",
        base_level: float = 1000.0,
        eps: float = 1e-9,
        prefix: str = "cgdi",
    ) -> pd.DataFrame:
        """
        CoinGlass Derivatives Index (CGDI) с расчётными фичами.
        
        Args:
            interval: Интервал (1d, 4h, etc.)
            base_level: Базовый уровень индекса (обычно 1000)
            eps: Epsilon для избежания log(0)
            prefix: Префикс для колонок
        
        Returns:
            DataFrame с колонками:
              - date
              - {prefix}__index_value
              - {prefix}__log_level (log сжатие)
              - {prefix}__dev_from_base (отклонение от базы)
              - {prefix}__dev_softsign (нормированный сигнал)
        """
        df = _coinglass_get_dataframe(
            endpoint="/futures/cgdi-index/history",
            api_key=self.api_key,
            params={"interval": interval},
        )
        
        if df.empty:
            return df
        
        # time -> date
        if "time" in df.columns:
            df["date"] = _coinglass_normalize_time_to_date(df["time"])
            df = df.drop(columns=["time"])
        
        # Нормализация числовых колонок
        df["index_value"] = pd.to_numeric(df.get("cgdi_index_value"), errors="coerce")
        
        # Очистка и сортировка
        df = (
            df[["date", "index_value"]]
            .dropna(subset=["date"])
            .sort_values("date", kind="stable")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        
        x = df["index_value"].astype(float)
        dev = x - float(base_level)
        adev = np.abs(dev)
        
        # Feature 1: log level (сжатие масштаба)
        df["log_level"] = np.log(x + eps)
        
        # Feature 2: deviation from base level
        df["dev_from_base"] = dev
        
        # Feature 3: softsign normalized deviation
        df["dev_softsign"] = dev / (adev + float(base_level) + eps)
        
        # Префикс
        df = _prefix_columns(df, prefix=prefix, keep=("date",))

        return df

    def get_sp500_ohlcv(
        self,
        days: int = 1250,
        prefix: str = "sp500",
    ) -> pd.DataFrame:
        """
        S&P 500 Index OHLCV данные через yfinance.

        Args:
            days: Количество дней истории
            prefix: Префикс для колонок

        Returns:
            DataFrame: date, {prefix}__open, {prefix}__close, {prefix}__high, {prefix}__low, {prefix}__volume
        """
        df = self._safe_yfinance_history(symbol="^GSPC", days=days, interval="1d")

        if df.empty:
            return df

        df = df.reset_index()
        df = df.rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # Оставляем только OHLCV
        df = df[["date", "Open", "Close", "High", "Low", "Volume"]].copy()
        df.columns = ["date", "open", "close", "high", "low", "volume"]

        # Конвертация в numeric
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Дедупликация и сортировка
        df = (
            df.dropna(subset=["date"])
            .sort_values("date", kind="stable")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        df = _prefix_columns(df, prefix=prefix, keep=("date",))
        return df

    def get_gold_ohlcv(
        self,
        days: int = 1250,
        prefix: str = "gold",
    ) -> pd.DataFrame:
        """
        Gold Futures (GC=F) OHLCV данные через yfinance.

        Args:
            days: Количество дней истории
            prefix: Префикс для колонок

        Returns:
            DataFrame: date, {prefix}__open, {prefix}__close, {prefix}__high, {prefix}__low, {prefix}__volume
        """
        df = self._safe_yfinance_history(symbol="GC=F", days=days, interval="1d")

        if df.empty:
            return df

        df = df.reset_index()
        df = df.rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        df = df[["date", "Open", "Close", "High", "Low", "Volume"]].copy()
        df.columns = ["date", "open", "close", "high", "low", "volume"]

        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = (
            df.dropna(subset=["date"])
            .sort_values("date", kind="stable")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        df = _prefix_columns(df, prefix=prefix, keep=("date",))
        return df

    def get_igv_ohlcv(
        self,
        days: int = 1250,
        prefix: str = "igv",
    ) -> pd.DataFrame:
        """
        iShares Expanded Tech-Software Sector ETF (IGV) OHLCV via yfinance.

        Args:
            days: Number of historical days
            prefix: Feature prefix

        Returns:
            DataFrame: date, {prefix}__open, {prefix}__close, {prefix}__high, {prefix}__low, {prefix}__volume
        """
        df = self._safe_yfinance_history(symbol="IGV", days=days, interval="1d")

        if df.empty:
            return df

        df = df.reset_index()
        df = df.rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        df = df[["date", "Open", "Close", "High", "Low", "Volume"]].copy()
        df.columns = ["date", "open", "close", "high", "low", "volume"]

        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = (
            df.dropna(subset=["date"])
            .sort_values("date", kind="stable")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        df = _prefix_columns(df, prefix=prefix, keep=("date",))
        return df


# ============================================================================
# Примеры использования
# ============================================================================
if __name__ == "__main__":
    load_dotenv("dev.env")
    API_KEY = os.getenv("COINGLASS_API_KEY")
    
    if not API_KEY:
        raise ValueError("COINGLASS_API_KEY не найден в dev.env")
    
    # Создаём экземпляр FeaturesGetter
    getter = FeaturesGetter(api_key=API_KEY)
    
    print("Доступные эндпоинты:")
    for name in getter.list_endpoints():
        print(f"  - {name}")
    print()
    
    # -------------------------------------------------------------------------
    # Пример 1: Open Interest History
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 1: Open Interest History (с дефолтными параметрами)")
    print("=" * 60)
    
    try:
        df = getter.get_history("open_interest_history")
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.tail())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 2: Funding Rate с кастомными параметрами
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 2: Funding Rate для ETH")
    print("=" * 60)
    
    try:
        df = getter.get_history(
            "funding_rate_history",
            symbol="ETHUSDT",
        )
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.head())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 3: Long/Short Ratio с кастомным префиксом
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 3: Long/Short Ratio с кастомным префиксом")
    print("=" * 60)
    
    try:
        df = getter.get_history(
            "global_long_short_account_ratio",
            prefix="ls_ratio",
        )
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.head())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 4: Aggregated данные (без exchange)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 4: Open Interest Aggregated")
    print("=" * 60)
    
    try:
        df = getter.get_history(
            "open_interest_aggregated",
            symbol="ETH",
        )
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.head())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 5: Bitcoin LTH Supply
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 5: Bitcoin Long-Term Holder Supply")
    print("=" * 60)
    
    try:
        df = getter.get_bitcoin_lth_supply()
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.tail())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 6: Bitcoin Active Addresses
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 6: Bitcoin Active Addresses")
    print("=" * 60)
    
    try:
        df = getter.get_bitcoin_active_addresses()
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.tail())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 7: Bitcoin STH Supply
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 7: Bitcoin Short-Term Holder Supply")
    print("=" * 60)
    
    try:
        df = getter.get_bitcoin_sth_supply()
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.tail())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 8: Bitcoin Reserve Risk
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 8: Bitcoin Reserve Risk")
    print("=" * 60)
    
    try:
        df = getter.get_bitcoin_reserve_risk()
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.tail())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 9: Bitfinex Margin Long/Short
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 9: Bitfinex Margin Long/Short")
    print("=" * 60)
    
    try:
        df = getter.get_bitfinex_margin_long_short()
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.tail())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 10: Coinbase Premium Index
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 10: Coinbase Premium Index")
    print("=" * 60)
    
    try:
        df = getter.get_coinbase_premium_index()
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.tail())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 11: CGDI Index
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 11: CoinGlass Derivatives Index (CGDI)")
    print("=" * 60)
    
    try:
        df = getter.get_cgdi_index()
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.tail())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    print("=" * 60)
    print("Все примеры выполнены!")
    print("=" * 60)