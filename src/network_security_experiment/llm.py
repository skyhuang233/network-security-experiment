"""OpenAI-compatible LLM client (reuses SILO's semantics, §13).

Tool calls use XML *text* format parsed by the engine — not function
calling — so switching models does not depend on function-calling support.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

import httpx
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


class LLMProvider(Protocol):
    def __call__(self, messages: list[dict[str, str]]) -> dict[str, Any]: ...


@retry(
    wait=wait_fixed(2),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def call_llm(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Call the chat-completions API; returns content + token counts."""
    client = OpenAI(base_url=api_base, api_key=api_key, timeout=httpx.Timeout(3600.0))
    response = client.chat.completions.create(model=model, messages=messages)
    choice = response.choices[0]
    usage = response.usage
    return {
        "content": choice.message.content or "",
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }


class OpenAICompatibleProvider:
    """Provider bound to one model configuration (A and B share one)."""

    def __init__(self, api_base: str, api_key: str, model: str) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

    def __call__(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return call_llm(self.api_base, self.api_key, self.model, messages)


def provider_from_env() -> OpenAICompatibleProvider:
    """Configuration from env: NSE_API_BASE / NSE_API_KEY / NSE_MODEL."""
    api_base = os.environ.get("NSE_API_BASE", "https://api.openai.com/v1")
    api_key = os.environ["NSE_API_KEY"]
    model = os.environ["NSE_MODEL"]
    return OpenAICompatibleProvider(api_base, api_key, model)


class ScriptedProvider:
    """DEVELOPMENT-ONLY deterministic provider for harness smoke tests.

    NOT used in any recorded experiment run: the study requires real LLM
    task agents (§1).  This provider replays a fixed per-agent script of XML
    tool-call turn outputs so the round loop, channel plumbing and
    measurement layer can be integration-tested offline.  Runs are stamped
    `provider: scripted` and flagged non-usable in reports.
    """

    def __init__(self, scripts: dict[int, list[str]]) -> None:
        # scripts: agent_id -> list of XML tool-call turn outputs (indexed
        # by the number of assistant messages already in that agent context)
        self.scripts = scripts

    def __call__(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        agent_id = 0
        for m in messages:
            if m["role"] == "system" and "Your unique agent ID is: 1" in m["content"]:
                agent_id = 1
        turn = sum(1 for m in messages if m["role"] == "assistant")
        script = self.scripts.get(agent_id, [])
        content = script[turn % len(script)] if script else ""
        return {"content": content, "input_tokens": 0, "output_tokens": 0}