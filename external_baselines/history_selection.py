"""Observed-history selection for a native DFoT transfer pilot; no model imports."""
import importlib.util
from pathlib import Path

import numpy as np

POLICIES = ("fifo", "uniform", "keepsake")


def load_scorer(memcam_root):
    path = Path(memcam_root) / "diffsynth/pipelines/memory_policies.py"
    spec = importlib.util.spec_from_file_location("keepsake_transfer_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_slam_covisibility_scores, path


def camera_to_c2w(conditions):
    conditions = np.asarray(conditions, dtype=np.float64)
    if conditions.ndim != 2 or conditions.shape[1] != 16 or not np.isfinite(conditions).all():
        raise ValueError("Expected finite DFoT intrinsics + w2c vectors (N,16)")
    if np.any(conditions[:, :2] <= 0):
        raise ValueError("Focal lengths must be positive")
    w2c = np.repeat(np.eye(4)[None], len(conditions), axis=0)
    w2c[:, :3] = conditions[:, 4:].reshape(-1, 3, 4)
    rotations = w2c[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-3):
        raise ValueError("Non-orthogonal camera rotations")
    if not np.allclose(np.linalg.det(rotations), 1, atol=2e-3):
        raise ValueError("Invalid camera handedness")
    return np.linalg.inv(w2c)


def validate_selection(selected, history_count, budget):
    if (len(selected) != budget or len(set(selected)) != budget
            or sorted(selected) != list(selected)
            or any(not isinstance(i, (int, np.integer)) or not 0 <= i < history_count for i in selected)):
        raise ValueError("Selected history must be distinct, chronological, causal and budget-sized")


def choose_history(policy, conditions, features, budget, scorer):
    """Only past pixels/poses enter this function. Newest observation is protected."""
    c2ws = camera_to_c2w(conditions)
    count = len(c2ws)
    if not 2 <= budget <= count:
        raise ValueError("Require 2 <= budget <= observed history length")
    if policy not in POLICIES:
        raise ValueError(f"Unknown selection policy: {policy}")
    if policy == "fifo":
        selected = list(range(count - budget, count))
        return selected, []
    if policy == "uniform":
        selected = np.rint(np.linspace(0, count - 1, budget)).astype(int).tolist()
        validate_selection(selected, count, budget)
        return selected, []
    features = np.asarray(features)
    if (features.ndim != 2 or features.shape[0] != count or not np.isfinite(features).all()
            or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4)):
        raise ValueError("Expected one normalized appearance feature per observed frame")
    bank, updates = [], []
    for incoming in range(count):
        candidates = bank + [incoming]
        scores, evicted = {}, []
        if len(candidates) > budget:
            # Reuse the production formula, using only the currently live bank.
            scores = scorer(candidates, c2ws[:incoming + 1],
                            dino_features={i: features[i] for i in candidates})
            if set(scores) != set(candidates) or not np.isfinite(list(scores.values())).all():
                raise ValueError("Invalid production retention scores")
            evicted = sorted((i for i in candidates if i != incoming), key=lambda i: (scores[i], i))[:len(candidates) - budget]
        bank = [i for i in candidates if i not in evicted]
        updates.append(dict(incoming=incoming, candidates=candidates, scores=scores,
                            protected=[incoming], evicted=evicted, retained=list(bank)))
    validate_selection(bank, count, budget)
    return bank, updates


def conditioning_indices(selected, history_count, future_count, budget):
    validate_selection(selected, history_count, budget)
    if future_count < 1 or budget + future_count != 8:
        raise ValueError("This pilot requires exactly eight DFoT token slots")
    return list(selected) + list(range(history_count, history_count + future_count))
