"""Round-loop engine: deterministic coordinator (§3, §4).

Borrowed from SILO: task input + private shards, per-agent context,
round-based execution, submit/termination semantics ("publish this round,
visible next round").  Rewritten here: tool dispatch targets the three real
storage backends through the measurement layer; no shared_kv.json.

The coordinator never decides *what* is exchanged; it only distributes
shards, executes tools, keeps the clock, logs, and terminates (both agents
submit_result, or max_rounds reached, default 100).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..channels.base import ChannelAdapter, ChannelView
from ..llm import LLMProvider
from ..measurement.recorder import MeasurementLog, now
from .prompts import ROUND_ENDING_TOOLS, generate_system_prompt, parse_tool_calls


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

    # ------------------------------------------------------------------ run

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

    # ------------------------------------------------------------ one round

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

    # -------------------------------------------------------------- tools

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

    # ------------------------------------------------------------ persist

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