"""Rollout and evaluation utilities for platformer controllers."""

import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .environment import (
    ACTION_DISTURBANCE_MAP,
    AVAILABLE_ACTIONS,
    LevelScenario,
    ObservationData,
    PlatformerEnv,
    plan_astar_actions,
)

MetricValue = float | str
MetricRow = dict[str, MetricValue]


class SupportsController(Protocol):
    """Minimal controller protocol used by rollout and evaluation helpers."""

    name: str

    def select_action(
        self,
        observation: ObservationData,
        environment: PlatformerEnv | None = None,
    ) -> int: ...


# =============================================================================
# Rollout helpers.
# =============================================================================
def sample_noisy_action(
    action: int,
    rng: np.random.Generator,
    noise_prob: float,
) -> int:
    """Applies the configured action-noise model to a chosen action."""
    disturbed_actions = ACTION_DISTURBANCE_MAP.get(action, AVAILABLE_ACTIONS)
    return (
        action
        if noise_prob <= 1e-8 or float(rng.random()) >= noise_prob
        else int(
            disturbed_actions[int(rng.integers(0, len(disturbed_actions)))]
        )
    )


def run_rollout(
    level: LevelScenario,
    controller: SupportsController,
    max_steps: int = 120,
) -> MetricRow:
    """Executes one closed-loop evaluation run and collects task metrics."""
    environment = PlatformerEnv(level, max_steps)
    times: list[float] = []
    hazard_failure = 0.0
    for _ in range(max_steps):
        observation = environment.observation()
        start_time = time.perf_counter()
        action = controller.select_action(observation, environment)
        times.append((time.perf_counter() - start_time) * 1000.0)
        step_result = environment.step(action)
        hazard_failure = max(
            hazard_failure,
            float(
                step_result.info["hazard_collision"] > 0
                and step_result.done
                and not environment.success
            ),
        )
        if step_result.done:
            break
    plan = plan_astar_actions(
        level, (level.start[0], level.start[1], 0), set()
    )
    path_efficiency = (
        len(plan) / environment.path_length
        if environment.success and environment.path_length > 0 and plan
        else 0.0
    )
    return {
        "success": float(environment.success),
        "steps": float(environment.steps),
        "blocked_moves": float(environment.blocked_actions),
        "path_efficiency": float(path_efficiency),
        "avg_decision_time_ms": float(np.mean(times) if times else 0.0),
        "hazard_failures": hazard_failure,
    }


def evaluate_controller_set(
    levels: Sequence[LevelScenario],
    controller: SupportsController,
    max_steps: int = 120,
) -> list[MetricRow]:
    """Evaluates one controller over a level batch and emits per-level rows."""
    name = controller.name
    rows = []
    for level_index, level in enumerate(levels):
        rollout = run_rollout(level, controller, max_steps)
        rows.append(
            {
                "controller_name": name,
                "level_index": float(level_index),
                "success": float(rollout["success"]),
                "steps": float(rollout["steps"]),
                "blocked_moves": float(rollout["blocked_moves"]),
                "path_efficiency": float(rollout["path_efficiency"]),
                "avg_decision_time_ms": float(rollout["avg_decision_time_ms"]),
                "hazard_failures": float(rollout["hazard_failures"]),
            }
        )
    return rows


# =============================================================================
# Summary helpers.
# =============================================================================
def summarize_evaluation_records(
    records: Sequence[MetricRow],
) -> list[MetricRow]:
    """Aggregates level rows into one summary record per controller."""
    rows_by_controller: dict[str, list[MetricRow]] = {}
    for record in records:
        rows_by_controller.setdefault(
            str(record["controller_name"]),
            [],
        ).append(record)
    rows: list[MetricRow] = []
    for controller_name, controller_rows in rows_by_controller.items():
        rows.append(
            {
                "controller_name": controller_name,
                "success_rate": float(
                    np.mean([float(row["success"]) for row in controller_rows])
                ),
                "avg_steps": float(
                    np.mean([row["steps"] for row in controller_rows])
                ),
                "avg_blocked_moves": float(
                    np.mean([row["blocked_moves"] for row in controller_rows])
                ),
                "avg_path_efficiency": float(
                    np.mean(
                        [row["path_efficiency"] for row in controller_rows]
                    )
                ),
                "avg_hazard_failures": float(
                    np.mean(
                        [row["hazard_failures"] for row in controller_rows]
                    )
                ),
                "avg_decision_time_ms": float(
                    np.mean(
                        [row["avg_decision_time_ms"] for row in controller_rows]
                    )
                ),
            }
        )
    return sorted(rows, key=lambda x: x["success_rate"], reverse=True)
