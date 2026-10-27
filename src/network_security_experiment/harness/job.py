"""One job: prompt, round loop, and channel assembly (§3, §4, §6, §13).

The agent-facing interface is the same for M1, M2, and M3: the agent sees
only send_payload / receive / wait / submit_result. Encoding, chunking,
framing, and storage stay behind the channel adapter. Tool calls are SILO
XML text parsed here, not function calls. wait and submit_result end the turn.

The coordinator distributes shards, executes tools, keeps the clock, and
stops when both agents submit_result or max_rounds is reached (default 100).
It does not decide what is exchanged. A publication becomes visible next round.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..channels.base import ChannelAdapter, ChannelPolicy, ChannelView
from ..channels.m1_file import M1FileContentChannel
from ..channels.m2_dirname import M2DirNameChannel
from ..channels.m3_cache import M3CacheStateChannel
from ..llm import LLMProvider
from ..measurement.recorder import MeasurementLog, now

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

TOOL_CALL_RE = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)
TOOL_NAME_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
PARAMS_BLOCK_RE = re.compile(r"<parameters>(.*?)</parameters>", re.DOTALL)
PARAM_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)

ROUND_ENDING_TOOLS = {"wait", "submit_result"}


def generate_system_prompt(agent_id: int, num_agents: int) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        agent_id=agent_id, num_agents=num_agents, max_id=num_agents - 1
    )


def _convert_value(value: str):
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
    """Parse [[ ... ]] blocks; returns [{tool, parameters}]."""
    results = []
    for block in TOOL_CALL_RE.findall(text):
        name_m = TOOL_NAME_RE.search(block)
        if not name_m:
            continue
        params: dict = {}
        pm = PARAMS_BLOCK_RE.search(block)
        if pm:
            for m in PARAM_RE.finditer(pm.group(1)):
                params[m.group(1)] = _convert_value(m.group(2))
        results.append({"tool": name_m.group(1).strip(), "parameters": params})
    return results


@dataclass
class AgentRunState:
    agent_id: int
    context: list[dict[str, str]] = field(default_factory=list)
    submitted: bool = False
    answer: Any = None
    submission_round: int | None = None


@dataclass
class CaseRunConfig:
    case_run_id: str
    case_id: str
    seed: int
    method: str  # M1 | M2 | M3
    num_agents: int = 2
    max_rounds: int = 100
    policy_mode: str = "shared"
    writable_subset: tuple[str, ...] | None = None  # control (c)
    key_set_size: int = 128  # M3
    ttl_seconds: float | None = None  # M3, experiment C
    provider_name: str = "openai"  # or "scripted" (dev-only)
    direction_label: str | None = None  # precheck stratification label


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
            result = self._execute(agent_id, tc)
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
            data = str(params.get("data", ""))
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
            agent.answer = params.get("answer")
            agent.submission_round = self.round_no
            return {"status": "submitted"}
        return {"error": f"Unknown tool: {name}"}

    def _persist(self, result: dict[str, Any]) -> None:
        with self._io_lock:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "measurement.json").write_text(
                json.dumps(self.mlog.to_dict(), indent=2, ensure_ascii=False, default=str)
            )
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


def build_policy(config: CaseRunConfig) -> ChannelPolicy:
    subset = (
        frozenset(config.writable_subset)
        if config.writable_subset is not None
        else None
    )
    return ChannelPolicy(
        mode=config.policy_mode,
        writable_subset=subset,
        ttl_seconds=config.ttl_seconds,
        key_set_size=config.key_set_size,
    )


def build_views(adapter: ChannelAdapter, config: CaseRunConfig) -> dict[int, ChannelView]:
    if config.policy_mode == "separated":
        # control (a): each agent gets an independent backend; in this
        # adapter each agent can read only its OWN namespace -> receive()
        # always yields nothing cross-agent (channel cut)
        return {
            i: ChannelView(
                agent_id=i,
                adapter=adapter,
                can_read_senders=frozenset({i}),
                writes_dropped=False,
            )
            for i in range(config.num_agents)
        }
    if config.policy_mode == "receiver_readonly":
        # control (b): B (agent 1) reads everything but its writes drop
        return {
            i: ChannelView(
                agent_id=i,
                adapter=adapter,
                can_read_senders=frozenset(range(config.num_agents)),
                writes_dropped=(i == 1),
            )
            for i in range(config.num_agents)
        }
    return {
        i: ChannelView(
            agent_id=i,
            adapter=adapter,
            can_read_senders=frozenset(range(config.num_agents)),
            writes_dropped=False,
        )
        for i in range(config.num_agents)
    }


def build_run(
    config: CaseRunConfig,
    user_prompts: list[str],
    provider,
    workspace: Path,
) -> CaseRun:
    """Assemble one CaseRun with its own channel backend namespace.

    Shared state and agent history are cleared before every run.
    """
    run_dir = workspace / config.case_run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    root = run_dir / "channel"
    mlog = MeasurementLog(method=config.method, case_run_id=config.case_run_id)
    adapter = ADAPTERS[config.method](
        root=str(root),
        measurement=mlog,
        policy=build_policy(config),
        num_agents=config.num_agents,
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
