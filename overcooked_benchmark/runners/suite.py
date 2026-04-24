from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from overcooked_benchmark.config import DEFAULT_OPENAI_MODEL, DEFAULT_VISION_MODEL
from overcooked_benchmark.runners.paired import run_agent_pair
from overcooked_benchmark.tasks import get_task_by_id


def run_experiment_suite(
    pair: str,
    task_ids: list[str],
    trials: int,
    max_ticks: int,
    text_model: str = DEFAULT_OPENAI_MODEL,
    vision_model: str = DEFAULT_VISION_MODEL,
    output_path: str | Path = "experiment_results.json",
) -> dict[str, Any]:
    results = []
    for task_id in task_ids:
        task = get_task_by_id(task_id)
        for trial in range(trials):
            summary = run_agent_pair(
                pair=pair,
                layout_name=task["layout"],
                task_id=task_id,
                max_ticks=max_ticks,
                text_model=text_model,
                vision_model=vision_model,
                collect_trajectory=False,
            )
            summary["trial"] = trial
            results.append(summary)

    aggregate = {
        "pair": pair,
        "trials": trials,
        "task_ids": task_ids,
        "results": results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(aggregate, handle, indent=2)
    return aggregate
