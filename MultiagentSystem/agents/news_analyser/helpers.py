"""Утилиты для работы с CoinGlass API (новости)."""

import os
import re
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / "dev.env")

BASE_URL = "https://open-api-v4.coinglass.com/api"
API_KEY = os.getenv("COINGLASS_API_KEY")


def coinglass_get_raw(endpoint: str, params: dict | None = None) -> dict:
    """Дёргает CoinGlass endpoint и возвращает сырой JSON-ответ."""
    url = f"{BASE_URL}{endpoint}"
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    r = requests.get(url, headers=headers, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()


def parse_release_time(value) -> datetime | None:
    """Пробует распарсить article_release_time (может быть unix ms или строка)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
    if isinstance(value, str) and value.isdigit():
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
    if isinstance(value, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def strip_html(html: str) -> str:
    """Убирает HTML-теги, оставляет чистый текст."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_articles_in_window(
    dt_from: datetime,
    dt_to: datetime,
    max_pages: int = 50,
) -> list[dict]:
    """Загружает статьи из CoinGlass /article/list с пагинацией,
    возвращает только те, что попадают в окно [dt_from, dt_to].

    API возвращает 20 статей/страницу (newest-first).
    Ранний выход, когда все статьи на странице старше dt_from.
    """
    results: list[dict] = []
    pages_fetched = 0

    for page in range(1, max_pages + 1):
        resp = coinglass_get_raw("/article/list", {"page": page})
        data = resp.get("data")
        if not data or not isinstance(data, list):
            break

        pages_fetched = page
        oldest_on_page: datetime | None = None

        for item in data:
            dt = parse_release_time(item.get("article_release_time"))
            if dt is None:
                continue
            if oldest_on_page is None or dt < oldest_on_page:
                oldest_on_page = dt
            if dt_from <= dt <= dt_to:
                results.append(item)

        # Early stop: все статьи на странице старше начала окна
        if oldest_on_page is not None and oldest_on_page < dt_from:
            break

    print(
        f"[fetch_articles] {pages_fetched} стр. загружено, "
        f"{len(results)} статей в окне {dt_from.date()} → {dt_to.date()}"
    )

    if pages_fetched >= max_pages and not results:
        print(
            f"[fetch_articles] WARN: окно {dt_from.date()} → {dt_to.date()} "
            f"может быть за пределами истории API (~23 дня назад)"
        )

    return results
