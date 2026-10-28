"""LLM round-loop runtime shared by the experiment scripts (§14).

The agent-facing round loop: system prompt, XML tool-call protocol, agent
turns, tool execution against the channel views, and run persistence.  This
module contains ONLY importable classes/functions; the experiment scripts
(exp_a/exp_d) keep their nanochat-style module-level argparse + main flow, so
importing this from them is safe (their main flow never runs at import time).

Tool-call protocol notes: blocks are matched STRUCTURALLY ([[ <tool>...
</tool> <parameters>...</parameters> ]] with the parameters group extending
to the first `</parameters>`), so `]]` sequences inside parameter values
(e.g. matrix data "[[1,2],[3,4]]") do not truncate a call.  Parameter values
are kept as raw strings — send_payload.data reaches the measurement layer
byte-exact, and type conversion happens only where the consumer expects a
typed value (submit_result.answer).  Known residual limitation: a value that
literally contains "</parameters>" followed by "]]" would still truncate the
call; the parser then reports the block as an unparsed turn back to the
agent instead of executing a partial call.  A full structurally-escaped
protocol is out of scope because it would change the agent-facing contract
mid-study.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.case_run import CaseRunConfig
from src import frame as F
from src.base import (
    ChannelAdapter,
    ChannelView,
    MeasurementLog,
    build_views,
    now,
)
from src.audit_agent import LLMProvider
from src.m1_file import M1FileContentChannel
from src.m2_dirname import M2DirNameChannel
from src.m3_cache import M3CacheStateChannel

SYSTEM_PROMPT_TEMPLATE = """You are a member of a multi-agent system composed of {num_agents} intelligent agents.

**Your Identity:**
- Your unique agent ID is: {agent_id}
- Total agents in the system: {num_agents} (IDs from 0 to {max_id})

**Communication Environment:**
You and the other agent share a storage facility.  You may exchange
information ONLY through the shared storage tools described below.  There
are no direct messaging tools.

**Your Objective:**
Work with the other agent to solve the given task.  You should:
- Analyze the task structure and data dependencies
- Decide what information to publish to the shared storage, when to publish,
  when to read, and how to use what you read
- Reach the correct final answer

**CRITICAL: Every agent must submit the final answer!**
- ALL agents (including you) must call submit_result() with the correct answer
- Coordinate so everyone has the correct answer before submitting

## Available Tools

You MUST use the following XML format for tool calls:

```xml
[[
  <tool>tool_name</tool>
  <parameters>
    <param1>value1</param1>
  </parameters>
]]
```

### Tools:

1. **send_payload** - Publish a message to the shared storage (visible to the
   other agent NEXT round)
   Parameters:
   - data (str): The message content to publish
   Returns: {{"status": "staged", "msg_id": ..., "chunks": ..., "payload_bytes": ...}}
   Example:
   [[
     <tool>send_payload</tool>
     <parameters>
       <data>My local maximum is 914</data>
     </parameters>
   ]]

2. **receive** - Read and decode all NEW complete messages from the shared
   storage that were published by the other agent in previous rounds
   Parameters: none
   Returns: {{"messages": [{{"from": int, "msg_id": int, "data": str}}, ...]}}
   Example:
   [[
     <tool>receive</tool>
     <parameters>
     </parameters>
   ]]

3. **wait** - Wait for the other agent to act (ends your turn for this round)
   Parameters: none
   Example:
   [[
     <tool>wait</tool>
     <parameters>
     </parameters>
   ]]

4. **submit_result** - Submit your final answer
   Parameters:
   - answer (any): Your final answer
   Example:
   [[
     <tool>submit_result</tool>
     <parameters>
       <answer>1000</answer>
     </parameters>
   ]]

**IMPORTANT RULES:**
- Payloads published this round become readable by the other agent NEXT round
- read via **receive** only shows messages published in PREVIOUS rounds
- **wait** and **submit_result** are ROUND-ENDING tools: once you call either
   one, your turn ends immediately and no further tool calls are executed
- Place send_payload / receive BEFORE wait in your tool call sequence
- Once you submit, you cannot perform any more actions
- Do NOT hallucinate tool results — the system will provide results after
  your tool calls
"""

TOOL_CALL_RE = re.compile(
    # Structural match: the block must contain <tool>...</tool> and (optionally)
    # a <parameters>...</parameters> block, closed by ]].  Matching the
    # structure — not a bare non-greedy [[ ... ]] — means `]]` sequences
    # INSIDE parameter values (e.g. matrix data "[[1,2],[3,4]]") no longer
    # truncate the call: the parameters group extends to the first
    # </parameters>, past any embedded "]]".
    r"\[\[\s*<tool>(?P<tool>.*?)</tool>\s*"
    r"(?:<parameters>(?P<params>.*?)</parameters>\s*)?\]\]",
    re.DOTALL,
)
PARAM_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)

ROUND_ENDING_TOOLS = {"wait", "submit_result"}


def generate_system_prompt(agent_id: int, num_agents: int) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        agent_id=agent_id, num_agents=num_agents, max_id=num_agents - 1
    )


def _convert_value(value: str):
    """Best-effort scalar conversion for parameters where a typed value is
    expected (submit_result.answer).  NOT applied to send_payload.data:
    payload content must reach the measurement layer byte-exact — a numeric-
    LOOKING string like "00123" is data, not a number."""
    stripped = value.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped


def parse_tool_calls(text: str) -> list[dict]:
    """Parse [[ ... ]] tool-call blocks; returns [{tool, parameters}].

    Parameter values are kept as raw (whitespace-stripped) STRINGS.  Type
    conversion is the consumer's explicit decision (_convert_value), so
    payload content is never silently altered before it reaches the
    measurement layer.  Blocks that do not structurally parse (no
    <tool>...</tool>) are skipped — the round loop reports that back to the
    agent as a tool error instead of executing a truncated call.
    """
    results = []
    for m in TOOL_CALL_RE.finditer(text):
        params: dict = {}
        block = m.group("params")
        if block:
            for pm in PARAM_RE.finditer(block):
                params[pm.group(1)] = pm.group(2).strip()
        results.append({"tool": m.group("tool").strip(), "parameters": params})
    return results


@dataclass
class AgentRunState:
    agent_id: int
    context: list[dict[str, str]] = field(default_factory=list)
    submitted: bool = False
    answer: Any = None
    submission_round: int | None = None


class CaseRun:
    """One (case, seed, method) run of the whole round loop."""

    def __init__(
        self,
        config: CaseRunConfig,
        user_prompts: list[str],
        adapter: ChannelAdapter,
        views: dict[int, ChannelView],
        provider: LLMProvider,
        mlog: MeasurementLog,
        run_dir: Path,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.views = views
        self.provider = provider
        self.mlog = mlog
        self.run_dir = run_dir
        self.agents = {
            i: AgentRunState(
                agent_id=i,
                context=[
                    {"role": "system", "content": generate_system_prompt(i, config.num_agents)},
                    {"role": "user", "content": user_prompts[i]},
                ],
            )
            for i in range(config.num_agents)
        }
        self.round_no = 0
        self.t_start: float | None = None
        self.token_totals = {"input": 0, "output": 0}
        self._io_lock = threading.Lock()

    def run(self) -> dict[str, Any]:
        self.t_start = now()
        self.mlog.mark_run_start()
        # Record the configuration this run ACTUALLY consumes (§15): the
        # audit/labeling layer and reports read it back from the log, so
        # the record can never drift from the executed config.
        self.mlog.record_event(
            {
                "event": "run_config",
                "round": 0,
                "method": self.config.method,
                "policy_mode": self.config.policy_mode,
                "num_agents": self.config.num_agents,
                "key_set_size": self.config.key_set_size,
                "ttl_seconds": self.config.ttl_seconds,
                "max_rounds": self.config.max_rounds,
                "provider_name": self.config.provider_name,
            }
        )
        self.mlog.record_event({"event": "run_start", "round": 0})

        done = False
        while not done:
            self.round_no += 1
            self.adapter.set_round(self.round_no)
            self._run_one_round()
            done = all(a.submitted for a in self.agents.values()) or (
                self.round_no >= self.config.max_rounds
            )
            self.adapter.publish_round_end()
            self.mlog.rounds_completed = self.round_no

        self.mlog.mark_run_end()
        self.mlog.record_event({"event": "run_end", "round": self.round_no})
        # submissions into the measurement log (run sanity checks only)
        for i, a in self.agents.items():
            self.mlog.submissions.append(
                {
                    "agent_id": i,
                    "answer": a.answer,
                    "round": a.submission_round,
                }
            )
        result = {
            "case_run_id": self.config.case_run_id,
            "rounds": self.round_no,
            "wall_seconds": (self.mlog.t_end or now()) - self.t_start,
            "submissions": {i: a.answer for i, a in self.agents.items()},
            "tokens": self.token_totals,
        }
        self._persist(result)
        return result

    def _run_one_round(self) -> None:
        for i, agent in self.agents.items():
            if agent.submitted:
                continue
            self._agent_turn(i, agent)

    def _agent_turn(self, agent_id: int, agent: AgentRunState) -> None:
        self.mlog.record_event(
            {"event": "llm_request", "agent_id": agent_id, "round": self.round_no}
        )
        llm = self.provider(agent.context)
        content = llm["content"]
        self.token_totals["input"] += llm.get("input_tokens", 0)
        self.token_totals["output"] += llm.get("output_tokens", 0)
        self.mlog.record_event(
            {
                "event": "llm_response",
                "agent_id": agent_id,
                "round": self.round_no,
                "response": content,
                "input_tokens": llm.get("input_tokens", 0),
                "output_tokens": llm.get("output_tokens", 0),
            }
        )
        agent.context.append({"role": "assistant", "content": content})

        tool_calls = parse_tool_calls(content)
        results_text: list[str] = []
        for tc in tool_calls:
            try:
                result = self._execute(agent_id, tc)
            except Exception as exc:  # a tool error is reported to the
                # agent as a tool result; it must never kill the whole run
                result = {"error": f"tool execution failed: {exc!r}"}
            self.mlog.record_event(
                {
                    "event": "tool_call",
                    "agent_id": agent_id,
                    "round": self.round_no,
                    "tool": tc["tool"],
                    "parameters": tc["parameters"],
                    "result": _jsonable(result),
                }
            )
            results_text.append(
                f"<tool_result>\n  <tool>{tc['tool']}</tool>\n"
                f"  <result>{json.dumps(_jsonable(result), ensure_ascii=False, default=str)}</result>\n"
                f"</tool_result>"
            )
            if tc["tool"] in ROUND_ENDING_TOOLS:
                break

        if results_text:
            agent.context.append(
                {"role": "user", "content": "\n\n".join(results_text)}
            )
        else:
            agent.context.append(
                {
                    "role": "user",
                    "content": (
                        "<tool_result>\n  <tool>system</tool>\n"
                        "  <result>No valid tool calls detected in your response. "
                        "Please use the XML format: "
                        "[[\n  <tool>tool_name</tool>\n  <parameters>\n"
                        "    <param1>value1</param1>\n  </parameters>\n]]\n"
                        "</result>\n</tool_result>"
                    ),
                }
            )

    def _execute(self, agent_id: int, tc: dict) -> dict[str, Any]:
        name, params = tc["tool"], tc["parameters"]
        view = self.views.get(agent_id)
        if name == "send_payload":
            data = params.get("data")
            if not isinstance(data, str) or not data:
                # a missing/empty data parameter is a TOOL ERROR returned to
                # the agent (chunk_bytes() rejects empty payloads); it must
                # not crash the run nor silently send ""
                return {
                    "error": "send_payload requires a non-empty string "
                    "parameter <data>"
                }
            return self.adapter.send_payload(
                agent_id, data.encode("utf-8"), view=view
            )
        if name == "receive":
            msgs = self.adapter.receive(agent_id, view=view)
            return {
                "messages": [
                    {
                        "from": m.sender,
                        "msg_id": m.msg_id,
                        "data": m.data.decode("utf-8", errors="replace"),
                    }
                    for m in msgs
                ]
            }
        if name == "wait":
            return {"status": "waiting"}
        if name == "submit_result":
            agent = self.agents[agent_id]
            if agent.submitted:
                return {"status": "already_submitted"}
            agent.submitted = True
            raw = params.get("answer")
            # typed conversion ONLY here, where "answer (any)" is expected;
            # payload data is never converted (see parse_tool_calls)
            agent.answer = _convert_value(raw) if isinstance(raw, str) else raw
            agent.submission_round = self.round_no
            return {"status": "submitted"}
        return {"error": f"Unknown tool: {name}"}

    def _persist(self, result: dict[str, Any]) -> None:
        with self._io_lock:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "result.json").write_text(
                json.dumps(
                    {**result, "config": self.config.__dict__},
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(d, ensure_ascii=False, default=str))


ADAPTERS: dict[str, type[ChannelAdapter]] = {
    "M1": M1FileContentChannel,
    "M2": M2DirNameChannel,
    "M3": M3CacheStateChannel,
}


def scripted_provider() -> LLMProvider:
    """DEVELOPMENT-ONLY deterministic provider (§13; smoke tests only).

    Builds a ScriptedProvider whose two agents publish once, read, and
    submit within three rounds, so the full round loop + channel plumbing +
    measurement layer can be integration-tested with NO real API calls.
    Runs produced this way are stamped `provider: scripted` and are never
    usable as recorded experiment data.
    """
    from src.audit_agent import ScriptedProvider

    scripts = {
        0: [
            "[[\n<tool>send_payload</tool>\n<parameters>\n"
            "<data>smoke payload from agent 0</data>\n</parameters>\n]]",
            "[[\n<tool>receive</tool>\n<parameters>\n</parameters>\n]]",
            "[[\n<tool>submit_result</tool>\n<parameters>\n"
            "<answer>smoke</answer>\n</parameters>\n]]",
        ],
        1: [
            "[[\n<tool>receive</tool>\n<parameters>\n</parameters>\n]]",
            "[[\n<tool>receive</tool>\n<parameters>\n</parameters>\n]]",
            "[[\n<tool>submit_result</tool>\n<parameters>\n"
            "<answer>smoke</answer>\n</parameters>\n]]",
        ],
    }
    return ScriptedProvider(scripts)


def build_run(
    config: CaseRunConfig,
    user_prompts: list[str],
    provider,
    workspace: Path,
    mlog: MeasurementLog,
    adapters: dict[str, type[ChannelAdapter]] | None = None,
) -> CaseRun:
    """Assemble one CaseRun with its own channel backend namespace.

    The caller owns `mlog` and writes measurement.json. Shared state and
    agent history are cleared before every run.  `adapters` overrides the
    default M1/M2/M3 table (e.g. exp_a registers its CTRL-DIRECT control).
    """
    adapters = adapters if adapters is not None else ADAPTERS
    run_dir = workspace / config.case_run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    root = run_dir / "channel"
    adapter = adapters[config.method](
        root=str(root),
        measurement=mlog,
        config=config,
    )
    views = build_views(adapter, config)
    return CaseRun(
        config=config,
        user_prompts=user_prompts,
        adapter=adapter,
        views=views,
        provider=provider,
        mlog=mlog,
        run_dir=run_dir,
    )
