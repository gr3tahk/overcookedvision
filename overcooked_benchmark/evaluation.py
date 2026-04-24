from __future__ import annotations

from typing import Any

from overcooked_benchmark.metrics import capability_rate, progress_completeness, score_against_references


def evaluate_task_trajectory(
    task: dict[str, Any],
    executed_actions: list[str],
    agent_histories: dict[str | int, list[str]] | None = None,
    collaboration_events: dict[str, list[dict[str, Any]]] | None = None,
    beta: float = 1.0,
) -> dict[str, Any]:
    references = task.get("reference_trajectories", [])
    flat_references = [reference for reference in references if "actions" in reference]
    result = score_against_references(executed_actions, flat_references, beta=beta)

    if agent_histories is not None:
        references_by_agent = task.get("agent_references", {})
        result["progress_completeness"] = progress_completeness(agent_histories, references_by_agent, beta=beta)

    collaboration_events = collaboration_events or {}
    if "initiations" in collaboration_events:
        result["initiating_capability"] = capability_rate(
            collaboration_events["initiations"],
            task.get("agent_references", {}),
            beta=beta,
        )
    if "responses" in collaboration_events:
        result["responding_capability"] = capability_rate(
            collaboration_events["responses"],
            task.get("agent_references", {}),
            beta=beta,
        )
    return result
