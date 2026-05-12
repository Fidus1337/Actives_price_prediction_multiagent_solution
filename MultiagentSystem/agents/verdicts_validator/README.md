# Verdicts Validator Agent

## Purpose
Validates intermediate agent verdicts before final aggregation step.

## Main files
- `agent_for_verdicts_validation.py` - validation logic
- `system_prompt.md` - validator prompt

## How to run
- Usually executed by orchestrator (`multiagent_system_main.py`) between agent runs and reports analyser.
- Direct invocation is possible by importing `agent_for_verdicts_validation(state)` in Python.

## Inputs and outputs
- Input: reports from parallel agents
- Output: validated reports with checks and corrections

