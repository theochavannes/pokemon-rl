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
        new_obs_dim=153,
        env=env,
        ppo_kwargs=PPO_KWARGS,
    )

The resulting model can be passed directly to model.learn().
"""

import numpy as np
import torch


def _make_dummy_env(new_obs_dim: int, action_dim: int = 10):
    """Build a minimal gymnasium env with the target obs/action shape.

    Used purely to instantiate a MaskablePPO with the right obs_space so we can
    save a checkpoint at the new dim, without constructing the full Gen1Env
    (which triggers poke-env websocket connections and is expensive).
    """
    import gymnasium
    from gymnasium.spaces import Box, Discrete
    from stable_baselines3.common.vec_env import DummyVecEnv

    class _StubEnv(gymnasium.Env):
        observation_space = Box(low=-1.0, high=1.0, shape=(new_obs_dim,), dtype=np.float32)
        action_space = Discrete(action_dim)

        def reset(self, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, True, False, {}

        def action_masks(self):
            return np.ones(action_dim, dtype=bool)

    return DummyVecEnv([lambda: _StubEnv()])


def transfer_and_save(
    old_path: str,
    new_obs_dim: int,
    new_path: str,
    ppo_kwargs: dict,
    action_dim: int = 10,
) -> None:
    """Expand an old checkpoint to a new obs_dim and save it to disk.

    Lightweight helper for cases (e.g., self-play seed) where we need a
    correctly-shaped checkpoint file on disk but don't yet have the full
    training env available. Uses a DummyVecEnv stub — no websockets, no
    poke-env state.

    **Pass the same ppo_kwargs used to train the model** (at minimum
    `policy_kwargs={"net_arch": ...}`) so the rebuilt network has matching
    layer shapes — otherwise obs_transfer silently skips layer copies and
    the saved checkpoint has random weights on the deep layers.
    """
    stub_env = _make_dummy_env(new_obs_dim, action_dim)
    # Strip tensorboard_log (stub env doesn't need logging).
    clean_kwargs = {k: v for k, v in ppo_kwargs.items() if k not in ("verbose", "tensorboard_log")}
    clean_kwargs.setdefault("policy", "MlpPolicy")
    try:
        model = load_with_expanded_obs(
            old_path=old_path,
            new_obs_dim=new_obs_dim,
            env=stub_env,
            ppo_kwargs=clean_kwargs,
        )
        model.save(new_path)
    finally:
        stub_env.close()


def load_with_expanded_obs(
    old_path: str,
    new_obs_dim: int,
    env,
    ppo_kwargs: dict,
):
    """
    Load an old MaskablePPO checkpoint into a new model with a larger observation space.

    - Weights for features the model already knew: copied exactly.
    - Weights for new features: zero-initialized (contribute nothing until trained).
    - All non-input layers (hidden->hidden, heads): copied unchanged.

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
            old_path,
            env=env,
            **{k: v for k, v in ppo_kwargs.items() if k not in ("verbose",)},
        )

    if new_obs_dim < old_obs_dim:
        raise ValueError(
            f"new_obs_dim ({new_obs_dim}) < old_obs_dim ({old_obs_dim}). Observation space can only grow, not shrink."
        )

    added = new_obs_dim - old_obs_dim
    print(f"  [Transfer] Expanding obs {old_obs_dim} -> {new_obs_dim} (+{added} dims)")

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
            patched.append(f"  {key}: {list(old_param.shape)} -> {list(new_param.shape)} (zero-padded)")

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
