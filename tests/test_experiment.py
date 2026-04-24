import unittest

from overcooked_benchmark.agents.base import parse_agent_response
from overcooked_benchmark.runners.paired import run_agent_pair


class ExperimentTest(unittest.TestCase):
    def test_response_parser_accepts_json(self):
        action, message, valid, reason = parse_agent_response('{"action":"left","message":"taking onion"}')
        self.assertEqual(action, "left")
        self.assertEqual(message, "taking onion")
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_response_parser_falls_back_to_stay(self):
        action, _, valid, reason = parse_agent_response("launch rocket")
        self.assertEqual(action, "stay")
        self.assertFalse(valid)
        self.assertIsNotNone(reason)

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
        self.assertEqual(summary["metrics"]["invalid_action_count"], 0)


if __name__ == "__main__":
    unittest.main()
