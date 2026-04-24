import unittest

from overcooked_benchmark.agents.base import parse_agent_response
from overcooked_benchmark.runners.paired import _build_no_op_warning
from overcooked_benchmark.runners.paired import run_agent_pair


class ExperimentTest(unittest.TestCase):
    def test_response_parser_accepts_json(self):
        action, message, plan, valid, reason = parse_agent_response(
            '{"action":"left","message":"taking onion","plan":"get onion then pot"}'
        )
        self.assertEqual(action, "left")
        self.assertEqual(message, "taking onion")
        self.assertEqual(plan, "get onion then pot")
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_response_parser_falls_back_to_stay(self):
        action, _, _, valid, reason = parse_agent_response("launch rocket")
        self.assertEqual(action, "stay")
        self.assertFalse(valid)
        self.assertIsNotNone(reason)

    def test_repeated_no_op_warning(self):
        self.assertEqual(_build_no_op_warning("interact", 1), "")
        self.assertIn("interact", _build_no_op_warning("interact", 2))

    def test_scripted_pair_smoke_run(self):
        summary, trajectory = run_agent_pair(
            pair="scripted-scripted",
            layout_name="cramped_room",
            task_id="cramped_room_single_delivery",
            max_ticks=4,
            collect_trajectory=True,
        )
        self.assertIn("metrics", summary)
        self.assertIn("tick_events", trajectory)
        self.assertIn("feedbackAfter", trajectory["prompt_logs"][0])
        self.assertIn("planAfter", trajectory["prompt_logs"][0])
        self.assertIn("phaseHint", trajectory["prompt_logs"][0])
        self.assertIn("plan", trajectory["tick_events"][0]["decisions"][0])
        self.assertEqual(summary["metrics"]["invalid_action_count"], 0)


if __name__ == "__main__":
    unittest.main()
