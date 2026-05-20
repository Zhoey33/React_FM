"""Replay validation protocol for branched rollouts.

Phase A: Outcome agreement (same checkpoint, same arm=none, 3 replays).
         Binary outcome must agree >= 80% of checkpoints.

Phase B: Utility ranking stability (same checkpoint, 3 arms, 3 replays).
         Kendall tau of utility ranking across replay sets >= 0.6.
"""

import logging
from collections import defaultdict

import numpy as np
from scipy import stats

from src.gate.branched_rollout import RolloutResult, compute_arm_utilities, GATE_ARMS

logger = logging.getLogger(__name__)


def phase_a_outcome_agreement(
    rollouts: dict[str, list[RolloutResult]],
    threshold: float = 0.8,
) -> dict:
    """Phase A: Check binary outcome agreement across replays.

    Args:
        rollouts: {checkpoint_id: [RolloutResult, ...]} for arm=none replays
        threshold: minimum fraction of checkpoints with agreement

    Returns:
        dict with 'passed', 'agreement_rate', 'details'
    """
    agreements = 0
    total = 0

    details = []
    for cp_id, replays in rollouts.items():
        if len(replays) < 2:
            continue
        total += 1

        # Binary outcome: did progress improve? (progress > 0)
        outcomes = [1 if r.progress > 0 else 0 for r in replays]
        # Agreement = all same outcome
        if len(set(outcomes)) == 1:
            agreements += 1
            details.append({"checkpoint_id": cp_id, "agreed": True, "outcomes": outcomes})
        else:
            details.append({"checkpoint_id": cp_id, "agreed": False, "outcomes": outcomes})

    agreement_rate = agreements / total if total > 0 else 0.0
    passed = agreement_rate >= threshold

    result = {
        "passed": passed,
        "agreement_rate": round(agreement_rate, 4),
        "total_checkpoints": total,
        "agreed_checkpoints": agreements,
        "threshold": threshold,
        "details": details,
    }

    status = "PASSED" if passed else "FAILED"
    logger.info(
        f"Phase A: {status} — agreement={agreement_rate:.1%} "
        f"({agreements}/{total}), threshold={threshold:.0%}"
    )
    return result


def phase_b_ranking_stability(
    rollouts_by_cp: dict[str, dict[str, list[RolloutResult]]],
    beta: float = 0.3,
    threshold: float = 0.6,
) -> dict:
    """Phase B: Check utility ranking stability across replay subsets.

    For each checkpoint, compute utility rankings from different replay subsets
    and check Kendall tau correlation.

    Args:
        rollouts_by_cp: {checkpoint_id: {arm: [RolloutResult, ...]}}
        beta: disruption penalty weight
        threshold: minimum Kendall tau

    Returns:
        dict with 'passed', 'mean_tau', 'details'
    """
    taus = []
    details = []

    for cp_id, arm_rollouts in rollouts_by_cp.items():
        # Need at least 2 replays per arm to split
        min_replays = min(len(rs) for rs in arm_rollouts.values() if rs)
        if min_replays < 2:
            continue

        # Split replays into two halves
        half_a = {}
        half_b = {}
        for arm, replays in arm_rollouts.items():
            mid = len(replays) // 2
            half_a[arm] = replays[:mid] if mid > 0 else replays[:1]
            half_b[arm] = replays[mid:] if mid > 0 else replays[:1]

        # Compute utility rankings for each half
        utils_a = compute_arm_utilities(half_a, cp_id, beta=beta)
        utils_b = compute_arm_utilities(half_b, cp_id, beta=beta)

        # Get rankings (only for gate arms present in both)
        common_arms = [a for a in GATE_ARMS if a in utils_a and a in utils_b]
        if len(common_arms) < 2:
            continue

        rank_a = [utils_a[a].utility for a in common_arms]
        rank_b = [utils_b[a].utility for a in common_arms]

        if len(common_arms) >= 3:
            tau, p_value = stats.kendalltau(rank_a, rank_b)
        else:
            # With only 2 arms, Kendall tau is either 1 or -1
            tau = 1.0 if (rank_a[0] >= rank_a[1]) == (rank_b[0] >= rank_b[1]) else -1.0
            p_value = 0.0

        # Skip NaN taus (can happen with tied utilities)
        if np.isnan(tau):
            continue

        taus.append(tau)
        details.append({
            "checkpoint_id": cp_id,
            "tau": round(tau, 4),
            "arms": common_arms,
            "utilities_a": {a: round(utils_a[a].utility, 4) for a in common_arms},
            "utilities_b": {a: round(utils_b[a].utility, 4) for a in common_arms},
        })

    mean_tau = float(np.mean(taus)) if taus else 0.0
    passed = mean_tau >= threshold

    result = {
        "passed": passed,
        "mean_tau": round(mean_tau, 4),
        "total_checkpoints": len(taus),
        "threshold": threshold,
        "details": details,
    }

    status = "PASSED" if passed else "FAILED"
    logger.info(
        f"Phase B: {status} — mean Kendall τ={mean_tau:.3f} "
        f"({len(taus)} checkpoints), threshold={threshold:.2f}"
    )
    return result
