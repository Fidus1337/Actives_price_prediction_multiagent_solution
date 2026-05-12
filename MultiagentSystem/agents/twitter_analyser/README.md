# Twitter Analyser Agent

## Purpose
Collects tweets from configured accounts, classifies BTC signal, and stores only directional items (`BULL` or `BEAR`) in SQLite archive.

## Main files
- `full_scrapping_pipeline.py` - collection entrypoint and run modes
- `twitter_news_classifier/classifier.py` - LLM tweet signal classifier
- `twitter_scrapper/twscraper.py` - Selenium tweet extraction and scrolling
- `twitter_scrapper/chrome_login_before_scrapping.py` - Chrome driver init and Twitter authentication
- `twitter_collector_settings.json` - accounts and collection window settings

## How to run
- Fetch new tweets and classify:
  - `python -m MultiagentSystem.agents.twitter_analyser.full_scrapping_pipeline --fetch-new`
- Reclassify all tweets currently in DB:
  - `python -m MultiagentSystem.agents.twitter_analyser.full_scrapping_pipeline --reclassify-all`
- Reclassify all and refetch for configured date range:
  - `python -m MultiagentSystem.agents.twitter_analyser.full_scrapping_pipeline --reclassify-and-refetch`

## Inputs and outputs
- Input source: X/Twitter profile pages (Selenium)
- Archive DB: `twitter_archive.db`
- Session cookies: `twitter_scrapper/twitter_cookies.json`

