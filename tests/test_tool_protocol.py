"""Tool-call protocol tests (issue 2).

- `]]` inside parameter values (matrix data) must not truncate a call
- numeric-LOOKING payload strings must stay strings (no silent conversion)
- a missing/empty `data` must surface as a tool error, never an empty send
"""

from __future__ import annotations

import unittest

from tasks.round_loop import _convert_value, parse_tool_calls


class ParseToolCallsTest(unittest.TestCase):
    def test_matrix_data_not_truncated(self) -> None:
        text = (
            "[[\n<tool>send_payload</tool>\n<parameters>\n"
            "<data>[[1,2],[3,4]]</data>\n</parameters>\n]]"
        )
        calls = parse_tool_calls(text)
        self.assertEqual(
            calls,
            [{"tool": "send_payload", "parameters": {"data": "[[1,2],[3,4]]"}}],
        )

    def test_matrix_and_real_call_in_same_text(self) -> None:
        # a bare `[[...]]` in prose must not swallow the following call
        text = (
            "The matrix is [[1,2],[3,4]].\n"
            "[[\n<tool>send_payload</tool>\n<parameters>\n"
            "<data>hello</data>\n</parameters>\n]]"
        )
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["parameters"]["data"], "hello")

    def test_numeric_looking_string_preserved(self) -> None:
        text = (
            "[[\n<tool>send_payload</tool>\n<parameters>\n"
            "<data>00123</data>\n</parameters>\n]]"
        )
        calls = parse_tool_calls(text)
        data = calls[0]["parameters"]["data"]
        self.assertEqual(data, "00123")
        self.assertIsInstance(data, str)

    def test_multiple_calls_parsed(self) -> None:
        text = (
            "[[\n<tool>send_payload</tool>\n<parameters>\n<data>a</data>\n</parameters>\n]]\n"
            "[[\n<tool>receive</tool>\n<parameters>\n</parameters>\n]]"
        )
        calls = parse_tool_calls(text)
        self.assertEqual(
            [(c["tool"], c["parameters"]) for c in calls],
            [("send_payload", {"data": "a"}), ("receive", {})],
        )

    def test_parameters_block_optional(self) -> None:
        self.assertEqual(
            parse_tool_calls("[[<tool>wait</tool>]]"),
            [{"tool": "wait", "parameters": {}}],
        )

    def test_non_tool_block_skipped(self) -> None:
        self.assertEqual(parse_tool_calls("random [[1,2]] text"), [])

    def test_submit_answer_still_convertible(self) -> None:
        # conversion remains the CONSUMER's decision (submit_result.answer)
        calls = parse_tool_calls(
            "[[\n<tool>submit_result</tool>\n<parameters>\n"
            "<answer>1000</answer>\n</parameters>\n]]"
        )
        self.assertEqual(calls[0]["parameters"]["answer"], "1000")
        self.assertEqual(_convert_value(calls[0]["parameters"]["answer"]), 1000)


class ExecuteToolErrorTest(unittest.TestCase):
    """A malformed send_payload must return an error to the agent, not crash."""

    def _run_turn(self, content: str) -> list[str]:
        from tasks.round_loop import CaseRun
        from config.case_run import CaseRunConfig
        from src.base import MeasurementLog
        from src.m1_file import M1FileContentChannel
        from src.base import build_views
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            config = CaseRunConfig(
                case_run_id="t", case_id="x", seed=0, method="M1",
                provider_name="scripted",
            )
            mlog = MeasurementLog("M1", "t")
            adapter = M1FileContentChannel(
                root=td, measurement=mlog, config=config
            )
            views = build_views(adapter, config)
            run = CaseRun(
                config=config,
                user_prompts=["u0", "u1"],
                adapter=adapter,
                views=views,
                provider=lambda messages: {"content": content, "input_tokens": 0, "output_tokens": 0},
                mlog=mlog,
                run_dir=Path(td),
            )
            run.round_no = 1
            agent = run.agents[0]
            run._agent_turn(0, agent)
            # the user turn carries the tool results
            return [m["content"] for m in agent.context if m["role"] == "user"]

    def test_send_payload_without_data_returns_error(self) -> None:
        users = self._run_turn(
            "[[\n<tool>send_payload</tool>\n<parameters>\n</parameters>\n]]"
        )
        self.assertIn("error", users[-1])

    def test_valid_send_still_works(self) -> None:
        users = self._run_turn(
            "[[\n<tool>send_payload</tool>\n<parameters>\n"
            "<data>hello</data>\n</parameters>\n]]"
        )
        self.assertIn("staged", users[-1])


if __name__ == "__main__":
    unittest.main()