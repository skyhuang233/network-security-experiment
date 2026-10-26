"""System prompt + XML tool-call parsing for the harness.

The agent-facing interface is the SAME for all three channel methods (§6):
the agent sees only `send_payload` / `receive` / `wait` / `submit_result`.
Encoding, chunking, framing and storage are transparent.  Tool calls use the
SILO XML text format (engine parses them itself — no function-calling, §13).
`wait` and `submit_result` are round-ending.
"""

from __future__ import annotations

import re

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