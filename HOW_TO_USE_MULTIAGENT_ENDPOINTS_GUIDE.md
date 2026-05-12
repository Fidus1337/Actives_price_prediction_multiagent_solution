# Як користуватися мультиагентними API-ендпоінтами

Документ для **споживача API**: ви викликаєте ендпоінти проти вже запущеного сервісу. Запуск сервісу, налаштування `dev.env`, релогін Twitter cookies — це задача того, хто розгортає API, тут не описується.

Базовий URL — `http://<host>:<port>/api/...` (далі в прикладах — `http://localhost:8000`). Інтерактивна Swagger-документація: `http://<host>:<port>/docs`.

Канонічний конфіг, на якому побудовані всі приклади нижче, активує **рівно двох агентів** — `agent_for_twitter_analysis` і `agent_for_analysing_tech_indicators`. Решта блоків у `agent_settings` присутні «про запас» і ігноруються, бо їх немає в `agent_envolved_in_prediction`.

```json
{
    "forecast_start_date": "2026-05-11",
    "horizon": 1,
    "agent_envolved_in_prediction": [
        "agent_for_twitter_analysis",
        "agent_for_analysing_tech_indicators"
    ],
    "neutral_threshold": 0.0
}
```

Повний цикл одного прогнозу — три кроки:

| Крок | Метод | Шлях | Призначення |
| --- | --- | --- | --- |
| 1 | GET  | `/api/agents/data-status` | Перевірити свіжість архівів |
| 2 | POST | `/api/system/collect_agent_data` | Долити свіжі дані в архіви |
| 3 | POST | `/api/multiagent_predictions` | Запустити прогноз |

---

## Крок 1. Перевіряємо свіжість архівів

```bash
curl http://localhost:8000/api/agents/data-status
```

Відповідь:

```json
{
    "news_analyser": "2026-05-08",
    "economic_calendar_analyser": "2026-05-08",
    "twitter_analyser": "2026-04-25"
}
```

**Що повертає ендпоінт.** Тільки `MAX(date)` по кожному з трьох архівів — це **дата найсвіжішого запису**, а **не кількість записів**. Підрахунку «скільки новин у нас зараз» цей ендпоінт не дає; його єдина задача — сказати, чи дотягується архів до потрібної дати прогнозу.

**Як читати під наш прогноз (`forecast_start_date = 2026-05-11`).** Активних в нашому конфізі два агенти — `agent_for_twitter_analysis` і `agent_for_analysing_tech_indicators`. Tech читає базовий датасет, який сервіс кешує самостійно — на нього `data-status` не впливає. Перевіряємо лише twitter:

| Агент | `window_to_analysis` | Потрібний діапазон | У відповіді | Дія |
| --- | --- | --- | --- | --- |
| `twitter_analyser` | 14 | `2026-04-28 … 2026-05-11` | `2026-04-25` | долити (Крок 2) |

Правило: якщо `data-status` для активного агента **< `forecast_start_date`** — переходимо до Кроку 2. Якщо **≥ `forecast_start_date`** — пропускаємо колекцію для цього агента.

> **Якщо потрібен саме лічильник записів** (а не дата), цей ендпоінт не допоможе. Звертайтесь до того, хто розгорнув сервіс — підрахунок робиться напряму по архівних файлах (`news_archive.json`, `twitter_archive.db`), а не через API.

#### Можливі помилки

| Код | Тіло |
| --- | --- |
| 500 | `{"detail": "Failed to query agent archives: <текст>"}` |

---

## Крок 2. Доливаємо дані — `POST /api/system/collect_agent_data`

Цей ендпоінт наповнює архіви **news / calendar / twitter**. Tech-агент сюди не входить — для нього сервіс автоматично оновлює свій внутрішній датасет.

#### Поля запиту

| Поле | Тип | Default | Опис |
| --- | --- | --- | --- |
| `agents` | `list[str]` | усі три | Які архіви наповнювати. Допустимі значення — рівно `"news_analyser"`, `"economic_calendar_analyser"`, `"twitter_analyser"`. |
| `twitter_authors` | `list[str] \| null` | `null` | Whitelist Twitter-авторів. **Впливає тільки на `twitter_analyser`.** Якщо `null` — використовуються всі активні акаунти зі стандартного списку сервісу. На news/calendar поле ігнорується. |
| `twitter_since_date` | `str \| null` (`YYYY-MM-DD`) | `null` | Початок діапазону для twitter. Якщо `null` — інкремент від `MAX(date)` у БД. |
| `twitter_until_date` | `str \| null` (`YYYY-MM-DD`) | `null` | Кінець діапазону для twitter. Якщо `null` — сьогодні. |

**Два ключових правила:**

1. **`twitter_*` поля впливають тільки на `twitter_analyser`.** News і calendar завжди йдуть інкрементально (`MAX(date)` → сьогодні), параметрів дат у запиті в них немає.
2. **Класифікація щойно завантажених твітів LLM-моделлю запускається лише коли передані обидва `twitter_since_date` і `twitter_until_date`.** Без обох — тільки фетч (інкрементальний режим без класифікації).

#### Запит

```bash
curl -X POST http://localhost:8000/api/system/collect_agent_data \
    -H "Content-Type: application/json" \
    -d '{
        "agents": [
            "news_analyser",
            "economic_calendar_analyser",
            "twitter_analyser"
        ],
        "twitter_authors": [
            "CarpeNoctom",
            "JSeyff",
            "AltcoinPsycho",
            "DavidDuong",
            "TraderMercury",
            "_Checkmatey_",
            "CryptoHayes",
            "rektcapital"
        ],
        "twitter_since_date": "2026-04-28",
        "twitter_until_date": "2026-05-11"
    }'
```

**Що відбудеться:**
- `news_analyser` і `economic_calendar_analyser` — інкремент від їхнього `MAX(date)` до сьогодні. `twitter_*` поля ігноруються.
- `twitter_analyser` — фетч твітів від восьми перелічених авторів за діапазон `2026-04-28 … 2026-05-11` + автоматична LLM-класифікація щойно завантажених твітів (бо передані обидві дати, які точно покривають `forecast_start_date - window + 1 … forecast_start_date`).

#### Відповідь

```json
{
    "results": [
        {"agent": "news_analyser", "before": 9421, "fetched": 78, "new": 78, "after": 9499, "date_range": "2026-05-08 … 2026-05-13"},
        {"agent": "economic_calendar_analyser", "before": 612, "fetched": 14, "new": 14, "after": 626, "date_range": "2026-05-08 … 2026-05-13"},
        {"agent": "twitter_analyser", "before": 1284, "fetched": 312, "new": 217, "after": 1501, "date_range": "2026-04-28 … 2026-05-11"}
    ]
}
```

`before` / `after` — кількість записів в архіві до і після виклику; `fetched` — скільки завантажено; `new` — скільки реально додано (без дублікатів).

> **Зверніть увагу.** Діапазон `twitter_since_date … twitter_until_date` має покривати щонайменше `forecast_start_date - window_to_analysis + 1 … forecast_start_date`. Для нашого конфігу (`forecast_start_date = 2026-05-11`, `window = 14`) — це `2026-04-28 … 2026-05-11`. Якщо діапазон менший — після Кроку 3 ви побачите частину дат у вікні з нульовою активністю, і twitter-агент буде «too weak — abstain».

#### Можливі помилки

| Код | Умова | Тіло |
| --- | --- | --- |
| 422 | У `agents` є невідоме ім'я | `{"detail": "Unknown agents: ['<...>']"}` |
| 409 | Колекція для того ж агента вже виконується | `{"detail": "Collection for '<agent>' is already running"}` — зачекайте і повторіть |
| 500 | Помилка під час колекції | `{"detail": "Collection failed for '<agent>': <текст>"}` |

---

## Крок 3. Запускаємо прогноз — `POST /api/multiagent_predictions`

Сервіс прорахує `n_last_dates` останніх придатних дат, рахуючи вглиб від `forecast_start_date`. Для одного прогнозу — `n_last_dates: 1`.

> ### ⚠️ Семантика `forecast_start_date`
>
> `forecast_start_date` — це **день ухвалення рішення**, тобто **остання дата з відомими даними**. Це **НЕ та дата, на яку прогнозуємо**.
>
> Прогноз ціни робиться на `forecast_start_date + horizon` днів уперед.
>
> **Приклад.** Хочете прогноз **на 2026-05-12**, горизонт 1 день:
> - беремо дату на день раніше → `forecast_start_date = 2026-05-11`
> - `horizon = 1`
> - відповідь скаже, чи піде ціна вгору/вниз станом на `2026-05-11 + 1 день = 2026-05-12`
>
> Те саме правило для більших горизонтів: `horizon = 3` із `forecast_start_date = 2026-05-11` → прогноз на `2026-05-14`.
>
> Звідси випливає: щоб отримати прогноз на день уперед, **ніколи не вказуйте сьогоднішню чи майбутню дату** в `forecast_start_date` — у системи ще немає закритих свічок і архівних даних на той день. Завжди `forecast_start_date ≤ вчора`.

#### Поля верхнього рівня

| Поле | Тип | Опис |
| --- | --- | --- |
| `forecast_start_date` | `str` (`YYYY-MM-DD`) | **День ухвалення рішення** — остання дата з відомими даними. **Це не дата, на яку прогнозуємо.** Прогноз робиться на `forecast_start_date + horizon`. Має бути ≤ вчора. |
| `horizon` | `int` (1–30) | Горизонт у днях. Прогноз = чи піде ціна вгору/вниз станом на `forecast_start_date + horizon`. |
| `n_last_dates` | `int` (1–365, default 10) | Скільки дат прорахувати, рахуючи вглиб від `forecast_start_date`. |
| `agent_envolved_in_prediction` | `list[str]` (≥1) | Активні агенти. Допустимі імена: `agent_for_analysing_tech_indicators`, `agent_for_twitter_analysis`, `agent_for_news_analysis`, `agent_for_economic_calendar_analysis`, `agent_for_analysing_onchain_indicators`. **Хто не в списку — не голосує**, навіть якщо його налаштування присутні в `agent_settings`. |
| `neutral_threshold` | `float` | Поріг нейтральності для `confidence_score`. `0` → нейтральної зони немає (будь-який ненульовий score дає LONG/SHORT). Підніміть до `1.0+`, щоб фільтрувати слабкі сигнали. |
| `agent_settings` | `dict` | Per-agent блоки. Зарезервований ключ `verdicts_validator` — для моделі валідатора. |

#### Поля всередині `agent_settings`-блоку

| Поле | Який агент читає | Що означає |
| --- | --- | --- |
| `system_prompt_file` | tech / news / onchain | Шлях до системного промпту відносно `MultiagentSystem/`. |
| `llm_model` | tech / calendar / onchain / verdicts_validator | Модель LLM. Префікс `claude-` → Anthropic, інше → OpenAI. |
| `window_to_analysis` | усі агенти | Розмір вікна аналізу (днів). |
| `base_feats` | tech / onchain | Перелік фіч, які агент бере з базового датасету. |
| `authors` | twitter | Whitelist Twitter-авторів. |
| `decay_rate`, `decay_start_day`, `initial_weight` | twitter / news | Параметри експоненційного загасання ваги по днях у вікні: до `decay_start_day` вага = `1.0`, далі — `initial_weight * (1 - decay_rate) ** (age - decay_start_day)`. |

#### Запит

```bash
curl -X POST http://localhost:8000/api/multiagent_predictions \
    -H "Content-Type: application/json" \
    -d '{
        "forecast_start_date": "2026-05-11",
        "horizon": 1,
        "n_last_dates": 1,
        "neutral_threshold": 0,
        "agent_envolved_in_prediction": [
            "agent_for_twitter_analysis",
            "agent_for_analysing_tech_indicators"
        ],
        "agent_settings": {
            "agent_for_analysing_tech_indicators": {
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
                ],
                "llm_model": "gpt-4.1",
                "system_prompt_file": "agents/tech_indicators/system_prompt_general.md",
                "window_to_analysis": 21
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
                "decay_rate": 0.05,
                "decay_start_day": 1,
                "initial_weight": 1,
                "window_to_analysis": 14
            },
            "agent_for_news_analysis": {
                "decay_rate": 0.2,
                "decay_start_day": 1,
                "initial_weight": 1,
                "system_prompt_file": "agents/news_analyser/system_prompt.md",
                "window_to_analysis": 1
            },
            "agent_for_economic_calendar_analysis": {
                "llm_model": "gpt-4.1",
                "window_to_analysis": 1
            },
            "agent_for_analysing_onchain_indicators": {
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
                ],
                "llm_model": "gpt-4.1",
                "system_prompt_file": "agents/onchain_indicators/system_prompt_1d.md",
                "window_to_analysis": 21
            },
            "verdicts_validator": {
                "llm_model": "gpt-4.1"
            }
        }
    }'
```

Активні в цьому виклику — `agent_for_twitter_analysis` і `agent_for_analysing_tech_indicators`. Налаштування `agent_for_analysing_onchain_indicators`, `agent_for_economic_calendar_analysis`, `agent_for_news_analysis` присутні в `agent_settings`, але **не використовуються**, бо їх немає в `agent_envolved_in_prediction` — це нормально й безпечно (можна тримати їх «про запас» і вмикати додаванням у список).

#### Відповідь

```json
{
    "requested_forecast_start_date": "2026-05-11",
    "requested_horizon": 1,
    "requested_n_last_dates": 1,
    "rows_returned": 1,
    "predictions": [
        {
            "date": "2026-05-11",
            "base_price": 81739.3,
            "y_true": 0,
            "y_prediction": 1,
            "confidence_score": 2
        }
    ]
}
```

**Як читати:**
- `y_prediction`: `1` — LONG (ціна на `forecast_start_date + horizon` буде **вищою**, ніж на `forecast_start_date`), `0` — SHORT (нижчою), `null` — нейтрально (`|score| ≤ neutral_threshold`).
- `confidence_score`: середнє знакових голосів **тих агентів, що проголосували** (агенти-абстейнери не входять у дільник). Кожен проголосувавший дає `sign(prediction) * weight`, де `weight ∈ {1=low, 2=medium, 3=high}`. У нас два потенційні агенти, тому крайні значення `-3` і `+3`. Знак `confidence_score` збігається зі знаком `y_prediction`.
- `base_price`: BTC spot close (Bybit) **на `forecast_start_date`** — це база, відносно якої система міряє «вгору/вниз». `null` — на цю дату ще не підтягнулися ціни.
- `y_true`: фактичний результат (`1` — ціна **на `forecast_start_date + horizon` справді була вище** за `base_price`, `0` — нижче). `null` — день `forecast_start_date + horizon` ще не настав. У нашому запиті `forecast_start_date + horizon = 2026-05-12` уже в минулому, тому `y_true` заповнений.

> **Що сталося в цьому прикладі.** Twitter-агент за вікно 14 днів від `2026-05-11` отримав занадто слабкі сигнали (взаємні гасіння BULL/BEAR + значна частка `NO_CORRELATION_TO_BTC`) і **утримався від голосу** (`prediction: null`). Tech-агент проголосував LONG із `confidence: medium` (+2). Дільник = 1 (один голос), тому `confidence_score = +2/1 = 2` → `y_prediction = 1`. Фактично ціна впала (`y_true = 0`) — це й корисний приклад того, що окремий бичий технічний сигнал без підтвердження від twitter ризикує помилитися.

#### Типові сценарії за конфігом

| Twitter | Tech | `confidence_score` | `y_prediction` | Як читати |
| --- | --- | --- | --- | --- |
| BULL low (+1) | BULL medium (+2) | `+1.5` | `1` (LONG) | Обидва за зростання — найсильніший консенсус, доступний для цього конфігу |
| abstain | BULL medium (+2) | `+2` | `1` (LONG) | Голосує тільки tech, дільник = 1 |
| BULL low (+1) | BEAR medium (-2) | `-0.5` | `0` (SHORT) | Tech переважає, бо в нього вищий confidence |
| abstain | abstain | `0` | `null` | Ніхто не проголосував — нейтраль |

#### Можливі помилки

| Код | Умова | Тіло |
| --- | --- | --- |
| 409 | Інший прогноз ще виконується | `{"detail": "Multiagent prediction is already running"}` — зачекайте 5–30 с і повторіть |
| 422 | Pydantic-валідація (формат дати, діапазон `horizon`/`n_last_dates`, порожній `agent_envolved_in_prediction`) | стандартне тіло `422` |
| 500 | Невловлене виключення під час прогнозу | `{"detail": "Failed to run multiagent predictions: <текст>"}` |
