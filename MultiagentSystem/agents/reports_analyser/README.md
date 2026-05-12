# Reports Analyser Agent

## Purpose
Combines validated agent reports into final confidence score and final system verdict.

## Main files
- `agent_for_reports_analysis.py` - report aggregation and confidence computation
- `system_prompt.md` - prompt definition (if used by your pipeline setup)

## How to run
- Usually executed by orchestrator (`multiagent_system_main.py`) after validator step.
- Direct invocation is possible by importing `agent_reports_analyser(state)` in Python.

## Inputs and outputs
- Input: validated outputs from other agents
- Output: final aggregate report in pipeline state

