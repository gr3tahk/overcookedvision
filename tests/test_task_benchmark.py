import unittest

from overcooked_ai_py.mdp.actions import Action

from overcooked_benchmark.evaluation import evaluate_task_trajectory
from overcooked_benchmark.runners.paired import run_agent_pair
from overcooked_benchmark.symbolic import classify_player_action
from overcooked_benchmark.tasks import get_task_by_id, load_tasks


class TaskBenchmarkTest(unittest.TestCase):
    def test_task_loading(self):
        tasks = load_tasks()
        self.assertGreaterEqual(len(tasks), 1)
        self.assertIn("task_id", tasks[0])
        self.assertIn("reference_trajectories", tasks[0])

    def test_symbolic_extraction_detects_key_actions(self):
        before = {
            "players": [
                {"position": (1, 1), "orientation": (0, -1), "held_object": None},
                {"position": (2, 1), "orientation": (0, -1), "held_object": None},
            ],
            "pots": {"pot0": {"ingredients": [], "is_cooking": False, "is_ready": False}},
        }
        after_pickup = {
            "players": [
                {"position": (1, 1), "orientation": (0, -1), "held_object": {"name": "onion"}},
                {"position": (2, 1), "orientation": (0, -1), "held_object": None},
            ],
            "pots": {"pot0": {"ingredients": [], "is_cooking": False, "is_ready": False}},
        }
        action = classify_player_action(before, after_pickup, 0, {}, Action.INTERACT)
        self.assertEqual(action, "pickup_onion")

        before_place = after_pickup
        after_place = {
            "players": [
                {"position": (1, 1), "orientation": (0, -1), "held_object": None},
                {"position": (2, 1), "orientation": (0, -1), "held_object": None},
            ],
            "pots": {"pot0": {"ingredients": ["onion"], "is_cooking": False, "is_ready": False}},
        }
        action = classify_player_action(before_place, after_place, 0, {}, Action.INTERACT)
        self.assertEqual(action, "place_onion_in_pot")

    def test_exact_reference_scores_best(self):
        task = get_task_by_id("cramped_room_single_delivery")
        executed = list(task["reference_trajectories"][0]["actions"])
        results = evaluate_task_trajectory(task, executed)
        self.assertEqual(results["best_reference_id"], task["reference_trajectories"][0]["id"])
        self.assertEqual(results["tes"], 1.0)

    def test_paired_benchmark_path_runs(self):
        summary, trajectory = run_agent_pair(
            pair="scripted-scripted",
            layout_name="cramped_room",
            task_id="cramped_room_single_delivery",
            max_ticks=8,
            collect_trajectory=True,
        )
        self.assertIn("tick_events", trajectory)
        self.assertIn("metrics", trajectory)
        self.assertIn("progress_completeness", trajectory["metrics"])
        self.assertIn("score", summary)


if __name__ == "__main__":
    unittest.main()
