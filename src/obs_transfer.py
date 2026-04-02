"""
Observation-space transfer learning for MaskablePPO.

When the observation space grows (new dims added), old model weights are preserved
for the features they already learned. New input connections start at zero — they
contribute nothing initially and are learned on top of the existing policy.

This works because only the first linear layer of each network head depends on
obs_dim. All deeper layers are hidden×hidden and are copied unchanged.

Usage:
    model = load_with_expanded_obs(
        old_path="runs/run_001/models/best_model",
        new_obs_dim=139,
        env=env,
        ppo_kwargs=PPO_KWARGS,
    )

The resulting model can be passed directly to model.learn().
"""

import torch
import numpy as np
from pathlib import Path


def load_with_expanded_obs(
    old_path: str,
    new_obs_dim: int,
    env,
    ppo_kwargs: dict,
) :
    """
    Load an old MaskablePPO checkpoint into a new model with a larger observation space.

    - Weights for features the model already knew: copied exactly.
    - Weights for new features: zero-initialized (contribute nothing until trained).
    - All non-input layers (hidden→hidden, heads): copied unchanged.

    Args:
        old_path:    Path to the old .zip checkpoint (without extension).
        new_obs_dim: Observation dimension of the NEW environment.
        env:         The new (expanded) VecEnv to attach.
        ppo_kwargs:  PPO hyperparameters dict for constructing the new model.

    Returns:
        MaskablePPO model ready for model.learn().
    """
    from sb3_contrib import MaskablePPO

    # ── 1. Load old model (detached from env) ──────────────────────────────
    old_model = MaskablePPO.load(old_path)
    old_obs_dim = old_model.observation_space.shape[0]

    if old_obs_dim == new_obs_dim:
        print(f"  [Transfer] obs_dim unchanged ({old_obs_dim}). Loading directly.")
        return MaskablePPO.load(
            old_path, env=env,
            **{k: v for k, v in ppo_kwargs.items() if k not in ("verbose",)},
        )

    if new_obs_dim < old_obs_dim:
        raise ValueError(
            f"new_obs_dim ({new_obs_dim}) < old_obs_dim ({old_obs_dim}). "
            "Observation space can only grow, not shrink."
        )

    added = new_obs_dim - old_obs_dim
    print(f"  [Transfer] Expanding obs {old_obs_dim} → {new_obs_dim} (+{added} dims)")

    # ── 2. Build fresh model with the new (larger) obs space ───────────────
    kwargs = {k: v for k, v in ppo_kwargs.items() if k not in ("verbose",)}
    new_model = MaskablePPO(env=env, **kwargs)

    # ── 3. Weight surgery ──────────────────────────────────────────────────
    # Identify input-layer keys: those where the old weight has shape [H, old_obs_dim]
    # and the new weight has shape [H, new_obs_dim]. These are the only layers that change.
    old_sd = old_model.policy.state_dict()
    new_sd = new_model.policy.state_dict()

    patched = []
    skipped = []
    for key, new_param in new_sd.items():
        if key not in old_sd:
            # New key (shouldn't happen unless architecture changed) — leave default
            skipped.append(key)
            continue

        old_param = old_sd[key]

        if old_param.shape == new_param.shape:
            # Same shape — direct copy (all hidden layers, biases, heads)
            new_sd[key] = old_param.clone()

        elif (
            old_param.ndim == 2
            and old_param.shape[1] == old_obs_dim
            and new_param.shape[1] == new_obs_dim
            and old_param.shape[0] == new_param.shape[0]
        ):
            # Input-layer weight: shape [H, obs_dim]
            # Copy old columns, zero-init new columns
            expanded = torch.zeros_like(new_param)
            expanded[:, :old_obs_dim] = old_param
            # new columns [:, old_obs_dim:] remain zero
            new_sd[key] = expanded
            patched.append(f"  {key}: {list(old_param.shape)} → {list(new_param.shape)} (zero-padded)")

        else:
            # Unexpected shape mismatch — leave new model's default init, warn
            skipped.append(f"  {key}: old={list(old_param.shape)} new={list(new_param.shape)}")

    new_model.policy.load_state_dict(new_sd)

    if patched:
        print("  Input layers expanded (old weights preserved, new cols zeroed):")
        for p in patched:
            print(p)
    if skipped:
        print("  Skipped (shape mismatch or new key — using default init):")
        for s in skipped:
            print(s)

    return new_model


def obs_dim_of(model_path: str) -> int:
    """Return the observation dimension of a saved checkpoint."""
    from sb3_contrib import MaskablePPO
    m = MaskablePPO.load(model_path)
    return m.observation_space.shape[0]


def is_compatible(model_path: str, current_obs_dim: int) -> bool:
    """True if the checkpoint can be loaded directly (same obs_dim)."""
    try:
        return obs_dim_of(model_path) == current_obs_dim
    except Exception:
        return False
