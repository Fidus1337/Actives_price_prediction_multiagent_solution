# Economic Calendar Analyser Agent

## Purpose
Collects macro calendar events and produces a BTC impact verdict for the forecast window.

## Main files
- `calendar_collector.py` - fetch and archive calendar events
- `agent_for_economic_calendar_analysis.py` - event filtering and LLM analysis

## How to run
- Collect/update calendar archive:
  - `python -m MultiagentSystem.agents.economic_calendar_analyser.calendar_collector`
- Analysis step is usually called by the orchestrator (`multiagent_system_main.py`).

## Inputs and outputs
- Input source: calendar API used in `calendar_collector.py`
- Archive: `calendar_archive.json`
- Debug output: `calendar_predict.json`

