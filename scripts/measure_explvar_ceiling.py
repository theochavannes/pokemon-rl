"""
Measure the ExplVar ceiling for the current observation space.

Collects (obs, actual_return) pairs from the current model playing games,
then fits an offline MLP regressor with cross-validation. The CV R² score
IS the ExplVar ceiling — the maximum ExplVar PPO's value function could
achieve given the information in the observation.

If CV R² ≈ 0.12, the ceiling is structural (partial observability).
If CV R² >> 0.20, the online PPO training is the bottleneck.

Usage:
    python scripts/measure_explvar_ceiling.py --model runs/run_054/models/best_model --n 500
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import os

os.chdir(Path(__file__).parent.parent)


async def collect_trajectories(model_path: str, n_games: int, gamma: float = 0.99):
    """Play n_games and collect (obs, discounted_return) pairs."""
    import random
    import string

    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
    from poke_env.ps_client.account_configuration import AccountConfiguration
    from sb3_contrib import MaskablePPO

    from src.agents.heuristic_agent import MaxDamagePlayer, SoftmaxDamagePlayer, TypeMatchupPlayer
    from src.env.gen1_env import Gen1Env, SB3Wrapper

    model = MaskablePPO.load(model_path)

    all_obs = []
    all_returns = []

    # Play against each opponent type
    opponents_cfg = [
        (
            "MaxDmg",
            lambda s: MaxDamagePlayer(
                battle_format="gen1randombattle",
                account_configuration=AccountConfiguration(f"Opp{s}", None),
                start_listening=False,
                log_level=40,
            ),
        ),
        (
            "TypeMatch",
            lambda s: TypeMatchupPlayer(
                battle_format="gen1randombattle",
                account_configuration=AccountConfiguration(f"Opp{s}", None),
                start_listening=False,
                log_level=40,
            ),
        ),
        (
            "SoftmaxDmg",
            lambda s: SoftmaxDamagePlayer(
                temperature=1.0,
                battle_format="gen1randombattle",
                account_configuration=AccountConfiguration(f"Opp{s}", None),
                start_listening=False,
                log_level=40,
            ),
        ),
    ]

    games_per_opp = n_games // len(opponents_cfg)

    for opp_name, opp_factory in opponents_cfg:
        print(f"  Collecting {games_per_opp} games vs {opp_name}...")

        for game_i in range(games_per_opp):
            suffix = "".join(random.choices(string.ascii_lowercase, k=6))
            inner = Gen1Env(
                account_configuration1=AccountConfiguration(f"Collect{suffix}", None),
                account_configuration2=AccountConfiguration(f"Puppet{suffix}", None),
                battle_format="gen1randombattle",
                log_level=40,
                strict=False,
            )
            opponent = opp_factory(suffix)
            wrapped = SB3Wrapper(SingleAgentWrapper(inner, opponent))

            obs, _ = wrapped.reset()
            episode_obs = []
            episode_rewards = []
            done = False

            while not done:
                episode_obs.append(obs.copy())
                mask = wrapped.action_masks()
                action, _ = model.predict(obs, deterministic=False, action_masks=mask)
                obs, reward, terminated, truncated, info = wrapped.step(action)
                episode_rewards.append(reward)
                done = terminated or truncated

            # Compute discounted returns for each timestep
            returns = np.zeros(len(episode_rewards))
            G = 0.0
            for t in reversed(range(len(episode_rewards))):
                G = episode_rewards[t] + gamma * G
                returns[t] = G

            all_obs.extend(episode_obs)
            all_returns.extend(returns.tolist())

            if (game_i + 1) % 50 == 0:
                print(f"    {game_i + 1}/{games_per_opp} games done")

    return np.array(all_obs, dtype=np.float32), np.array(all_returns, dtype=np.float32)


def measure_ceiling(obs: np.ndarray, returns: np.ndarray):
    """Fit offline MLP and measure cross-validated R²."""
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print(f"\n  Dataset: {len(obs)} samples, {obs.shape[1]} features")
    print(
        f"  Returns: mean={returns.mean():.3f}, std={returns.std():.3f}, "
        f"min={returns.min():.3f}, max={returns.max():.3f}"
    )

    # Pipeline: scale + MLP (same size as PPO value network)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(256, 128),
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                ),
            ),
        ]
    )

    # 5-fold cross-validated R² = ExplVar ceiling
    print("\n  Running 5-fold cross-validation...")
    scores = cross_val_score(pipe, obs, returns, cv=5, scoring="r2")

    print(f"\n  {'=' * 50}")
    print("  Cross-validated R² (ExplVar ceiling):")
    print(f"    Per fold: {', '.join(f'{s:.3f}' for s in scores)}")
    print(f"    Mean:  {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  {'=' * 50}")

    if scores.mean() < 0.20:
        print(f"\n  STRUCTURAL CEILING: CV R²={scores.mean():.3f}")
        print(f"  The observation space cannot support ExplVar > ~{scores.mean():.2f}.")
        print("  Fix requires changing observations or reward, not training.")
    else:
        print(f"\n  TRAINING BOTTLENECK: CV R²={scores.mean():.3f}")
        print(f"  An offline MLP achieves {scores.mean():.2f} but PPO only reaches ~0.12.")
        print("  Online training is the problem, not the observation space.")

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/run_054/models/best_model")
    parser.add_argument("--n", type=int, default=300, help="Games to collect")
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    print(f"Measuring ExplVar ceiling for {args.model}")
    print(f"Collecting {args.n} games (gamma={args.gamma})\n")

    obs, returns = asyncio.run(collect_trajectories(args.model, args.n, args.gamma))
    scores = measure_ceiling(obs, returns)

    # Save dataset for future analysis
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "explvar_ceiling_data.npz", obs=obs, returns=returns, cv_scores=scores)
    print("\n  Saved dataset to data/explvar_ceiling_data.npz")


if __name__ == "__main__":
    main()
