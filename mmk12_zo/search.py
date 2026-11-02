"""Vanilla ARS selection and ordinary ES z-score weighting."""

from __future__ import annotations

import math


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def select_directions(
    rewards_plus: list[float],
    rewards_minus: list[float],
    top_k: int,
    sigma_eps: float = 1e-12,
) -> dict:
    """Vanilla ARS: top-k by max(R+, R-), smaller index wins ties.

    Sigma is the population std of (R+ + R-) / 2 over the selected directions.
    """
    scores = [max(pos, neg) for pos, neg in zip(rewards_plus, rewards_minus)]
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    selected = order[:top_k]
    means = [(rewards_plus[index] + rewards_minus[index]) / 2.0 for index in selected]
    diffs = [rewards_plus[index] - rewards_minus[index] for index in selected]
    sigma = population_std(means)
    skip = sigma < sigma_eps or all(diff == 0.0 for diff in diffs)
    if skip:
        reason = "zero_variance" if sigma < sigma_eps else "zero_reward_difference"
    else:
        reason = None
    return {
        "selected": selected,
        "scores": scores,
        "means": means,
        "diffs": diffs,
        "sigma": sigma,
        "skip": skip,
        "skip_reason": reason,
    }


def es_weights(rewards: list[float], sigma_eps: float = 1e-8) -> dict:
    """Ordinary ES: z-score every one-sided reward, then use all of them.

    This follows ES at Scale: Z = (R - mean) / std, and the update is
    theta += (alpha / N) * sum Z_n * epsilon_n. If the population std is
    ~0, the step is skipped.
    """
    mean = sum(rewards) / len(rewards)
    sigma = population_std(rewards)
    if sigma < sigma_eps:
        return {
            "weights": [0.0 for _ in rewards],
            "mean": mean,
            "sigma": sigma,
            "skip": True,
            "skip_reason": "zero_variance",
        }
    return {
        "weights": [(reward - mean) / sigma for reward in rewards],
        "mean": mean,
        "sigma": sigma,
        "skip": False,
        "skip_reason": None,
    }
