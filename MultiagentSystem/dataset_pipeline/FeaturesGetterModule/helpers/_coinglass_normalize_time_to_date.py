import pandas as pd

def _coinglass_normalize_time_to_date(series: pd.Series) -> pd.Series:
    """
    CoinGlass обычно возвращает time в ms, 
    
    Unix timestamp — это количество времени, прошедшее с 1 января 1970 года 00:00:00 UTC.
    Такой формат удобен для описания точного времени, но мы его должны конвертировать в формат YYYY-MM-DD.
    """
    s = pd.to_numeric(series, errors="coerce")

    # Берем медианную запись в серии, чтобы определить единицу измерения.
    median = s.dropna().median() if s.notna().any() else None
    if median is None:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="string")

    if median >= 1e17:
        unit = "ns"
    elif median >= 1e14:
        unit = "us"
    elif median >= 1e11:
        unit = "ms"
    else:
        unit = "s"

    dt = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d")

if __name__ == "__main__":

    # Пример 1: миллисекунды (типичный формат CoinGlass)
    timestamps_ms = pd.Series([1704067200000, 1704153600000, 1704240000000])
    result = _coinglass_normalize_time_to_date(timestamps_ms)
    print(result)
    # 0    2024-01-01
    # 1    2024-01-02
    # 2    2024-01-03

    # Пример 2: секунды (Unix timestamp)
    timestamps_s = pd.Series([1704067200, 1704153600, 1704240000])
    result = _coinglass_normalize_time_to_date(timestamps_s)
    print(result)
    # 0    2024-01-01
    # 1    2024-01-02
    # 2    2024-01-03

    # Пример 3: с пропущенными значениями
    timestamps_with_nan = pd.Series([1704067200000, None, 1704240000000])
    result = _coinglass_normalize_time_to_date(timestamps_with_nan)
    print(result)
    # 0    2024-01-01
    # 1          <NA>
    # 2    2024-01-03
