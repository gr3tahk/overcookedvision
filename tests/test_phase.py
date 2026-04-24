import unittest

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from overcooked_benchmark.phase import task_phase_hint


class PhaseHintTest(unittest.TestCase):
    def test_initial_phase_collects_onions(self):
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        state = mdp.get_standard_start_state()
        self.assertIn("collect onions", task_phase_hint(state, mdp))

    def test_onion_carrier_phase_mentions_pot(self):
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        state = mdp.get_standard_start_state()
        state, _ = mdp.get_state_transition(state, [Action.STAY, (1, 0)])
        state, _ = mdp.get_state_transition(state, [Action.STAY, Action.INTERACT])
        hint = task_phase_hint(state, mdp)
        self.assertIn("hold onion", hint)
        self.assertIn("face it", hint)


if __name__ == "__main__":
    unittest.main()
