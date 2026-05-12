# News Analyser Agent

## Purpose
Collects crypto news, classifies each article by BTC impact, and produces a daily sentiment verdict.

## Main files
- `news_collector.py` - fetch/archive/classify news
- `news_classifier.py` - LLM classifier (`bull` / `bear` / `not_correlated`, with strength)
- `agent_for_news_analysis.py` - aggregation logic used by the multi-agent pipeline

## How to run
- Collect new news and classify new items:
  - `python -m MultiagentSystem.agents.news_analyser.news_collector`
- Backfill missing classifications in archive:
  - `python -m MultiagentSystem.agents.news_analyser.news_collector --backfill`
- Reclassify full archive:
  - `python -m MultiagentSystem.agents.news_analyser.news_collector --reclassify`

## Inputs and outputs
- Input source: CoinGlass article API
- Archive: `news_archive.json`
- Debug output: `news_predict.json`

