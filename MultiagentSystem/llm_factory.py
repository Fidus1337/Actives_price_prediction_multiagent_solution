import os

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


def make_chat_llm(model: str, temperature: float = 0.0, **kwargs):
    if model.startswith("claude"):
        api_key = os.getenv("CLAUDE_KEY")
        if not api_key:
            raise RuntimeError(
                f"CLAUDE_KEY env var is required for model '{model}' but is not set."
            )
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **kwargs,
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            f"OPENAI_API_KEY env var is required for model '{model}' but is not set."
        )
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)
