# Tech Indicators Agent

## Purpose
Analyzes technical indicators and returns a directional report used by the multi-agent pipeline.

## Main files
- `agent_for_analysing_tech_indicators.py` - main agent function (`agent_a_tech`)
- `system_prompt_1d.md`, `system_prompt_7d.md`, `system_prompt_general.md` - prompt templates

## How to run
- Usually executed by orchestrator (`multiagent_system_main.py`) as part of graph run.
- Direct invocation is possible by importing `agent_a_tech(state)` in Python.

## Inputs and outputs
- Input snapshot: `input_data.json`
- Debug output: `tech_predict.json`

