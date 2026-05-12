# Onchain Indicators Agent

## Purpose
Analyzes on-chain indicators and generates BTC direction report for the forecast horizon.

## Main files
- `agent_for_analysing_onchain_indicators.py` - main agent function (`agent_b_onchain`)
- `system_prompt_1d.md`, `system_prompt_7d.md` - prompt templates

## How to run
- Usually executed by orchestrator (`multiagent_system_main.py`) as part of graph run.
- Direct invocation is possible by importing `agent_b_onchain(state)` in Python.

## Inputs and outputs
- Input snapshot: `input_data.json`
- Debug output: `onchain_predict.json`

