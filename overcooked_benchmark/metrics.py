from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ActionSequence = Sequence[str]
Reference = Mapping[str, Any] | ActionSequence


def _reference_id(reference: Reference, fallback: str) -> str:
    if isinstance(reference, Mapping):
        return str(reference.get("id", fallback))
    return fallback


def _reference_actions(reference: Reference) -> list[str]:
    if isinstance(reference, Mapping):
        return [str(action) for action in reference.get("actions", [])]
    return [str(action) for action in reference]


def _longest_ordered_match(executed: ActionSequence, reference: ActionSequence) -> int:
    """Length of the longest order-preserving subsequence in executed matching reference."""
    if not executed or not reference:
        return 0

    ref_len = len(reference)
    previous = [0] * (ref_len + 1)
    for action in executed:
        current = previous.copy()
        for ref_index, ref_action in enumerate(reference, start=1):
            if action == ref_action:
                current[ref_index] = max(current[ref_index], previous[ref_index - 1] + 1)
            current[ref_index] = max(current[ref_index], current[ref_index - 1])
        previous = current
    return previous[ref_len]


def compute_tes(executed: ActionSequence, reference: ActionSequence, beta: float = 1.0) -> float:
    """Compute Trajectory Efficiency Score from Collab-Overcooked.

    TES rewards ordered progress against a reference action trajectory while penalizing
    redundant actions in the executed history.
    """
    executed_actions = [str(action) for action in executed]
    reference_actions = [str(action) for action in reference]

    if not executed_actions and not reference_actions:
        return 1.0
    if not executed_actions or not reference_actions:
        return 0.0

    match_len = _longest_ordered_match(executed_actions, reference_actions)
    beta_squared = beta**2
    denominator = len(reference_actions) + beta_squared * len(executed_actions)
    if denominator == 0:
        return 0.0
    return ((1 + beta_squared) * match_len) / denominator


def compute_ites(
    new_actions: str | ActionSequence,
    history: ActionSequence,
    reference: ActionSequence | None = None,
    beta: float = 1.0,
) -> float:
    """Compute Incremental TES for one action, a batch, or the last step of a trajectory."""
    if reference is None:
        executed = [str(action) for action in ([] if isinstance(new_actions, str) else new_actions)]
        reference = history
        if not executed:
            return 0.0
        return compute_tes(executed, reference, beta=beta) - compute_tes(executed[:-1], reference, beta=beta)

    if isinstance(new_actions, str):
        action_batch = [new_actions]
    else:
        action_batch = [str(action) for action in new_actions]
    before = compute_tes(history, reference, beta=beta)
    after = compute_tes([*history, *action_batch], reference, beta=beta)
    return after - before


def score_against_references(
    executed: ActionSequence,
    references: Sequence[Reference],
    beta: float = 1.0,
) -> dict[str, Any]:
    """Score a trajectory against several RATs and return the best TES match."""
    if not references:
        return {
            "tes": 0.0,
            "best_reference_id": None,
            "best_reference_actions": [],
            "reference_scores": [],
        }

    reference_scores = []
    for index, reference in enumerate(references):
        actions = _reference_actions(reference)
        reference_scores.append(
            {
                "id": _reference_id(reference, f"reference_{index}"),
                "tes": compute_tes(executed, actions, beta=beta),
                "actions": actions,
            }
        )

    best = max(reference_scores, key=lambda item: item["tes"])
    return {
        "tes": best["tes"],
        "best_reference_id": best["id"],
        "best_reference_actions": best["actions"],
        "reference_scores": reference_scores,
    }


def progress_completeness(
    agent_histories: Mapping[str | int, ActionSequence],
    references_by_agent: Mapping[str | int, Sequence[Reference]],
    beta: float = 1.0,
) -> dict[str, Any]:
    """Average each agent's best TES into Progress Completeness."""
    agent_scores = {}
    for agent_id, history in agent_histories.items():
        references = references_by_agent.get(agent_id, [])
        agent_scores[str(agent_id)] = score_against_references(history, references, beta=beta)

    if not agent_scores:
        pc = 0.0
    else:
        pc = sum(score["tes"] for score in agent_scores.values()) / len(agent_scores)
    return {"pc": pc, "agents": agent_scores}


def capability_rate(
    evaluated_actions: Sequence[Mapping[str, Any]],
    references_by_agent: Mapping[str | int, Sequence[Reference]],
    beta: float = 1.0,
) -> dict[str, Any]:
    """Compute IC/RC-style rate from annotated request or response actions.

    Each item should include `agent_id`, `history_before`, and `actions`. The action is
    counted as correct when its ITES against the best available reference is positive.
    """
    scored = []
    for item in evaluated_actions:
        agent_id = item.get("agent_id")
        references = references_by_agent.get(agent_id, references_by_agent.get(str(agent_id), []))
        history = [str(action) for action in item.get("history_before", [])]
        actions = item.get("actions", [])
        if isinstance(actions, str):
            actions = [actions]

        best_reference = score_against_references(history, references, beta=beta)["best_reference_actions"]
        ites = compute_ites(actions, history, best_reference, beta=beta) if best_reference else 0.0
        scored.append({**dict(item), "ites": ites, "correct": ites > 0})

    rate = sum(1 for item in scored if item["correct"]) / len(scored) if scored else None
    return {"rate": rate, "count": len(scored), "items": scored}
