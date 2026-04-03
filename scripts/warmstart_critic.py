"""Pre-train the value function (critic) of a BC-warm-started model.

The BC training script only trains the policy (actor) head. The value head
remains randomly initialized, which causes PPO to make bad advantage estimates
early in RL training, eroding the BC prior before the critic catches up.

This script fixes that by:
  1. Playing games with the BC policy to collect (obs, reward) trajectories
  2. Computing discounted returns G_t for each timestep
  3. Training the value head on (obs -> G_t) with MSE loss

Usage:
    python scripts/warmstart_critic.py                         # defaults
    python scripts/warmstart_critic.py --n-games 2000          # more games
    python scripts/warmstart_critic.py --model models/bc_warmstart  # custom model
"""

import argparse
import asyncio
import os
import random
import string
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from poke_env.player.baselines import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from sb3_contrib import MaskablePPO

_ROOT = Path(__file__).parent.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

from src.agents.heuristic_agent import (
    MaxDamagePlayer,
    SmartHeuristicPlayer,
    TypeMatchupPlayer,
)
from src.env.gen1_env import embed_battle

BATTLE_FORMAT = "gen1randombattle"


class ValueDataCollector(SmartHeuristicPlayer):
    """Plays games using the BC policy and records (obs, reward) per step."""

    def __init__(self, model: MaskablePPO, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self._episode_obs: list[np.ndarray] = []
        self._episode_rewards: list[float] = []
        self._prev_value: float = 0.0
        self.all_obs: list[np.ndarray] = []
        self.all_returns: list[float] = []

    def choose_move(self, battle):
        obs = embed_battle(battle)
        self._episode_obs.append(obs)

        # Compute step reward as delta from previous state value
        current_value = self._state_value(battle)
        step_reward = current_value - self._prev_value
        self._episode_rewards.append(step_reward)
        self._prev_value = current_value

        # Use the BC policy to pick the action
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.model.policy.device).unsqueeze(0)
        from poke_env.environment.singles_env import SinglesEnv

        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.model.policy.device).unsqueeze(0)

        with torch.no_grad():
            features = self.model.policy.extract_features(obs_tensor, self.model.policy.pi_features_extractor)
            latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
            logits = self.model.policy.action_net(latent_pi)
            logits[mask_tensor == 0] = -float("inf")
            action = logits.argmax(dim=1).cpu().numpy()[0]

        return SinglesEnv.action_to_order(action, battle)

    @staticmethod
    def _state_value(battle) -> float:
        """Compute absolute state value (not delta) matching Gen1Env reward params."""
        value = 0.0
        for mon in battle.team.values():
            if mon.fainted:
                value -= 0.5
        for mon in battle.opponent_team.values():
            if mon.fainted:
                value += 0.5
        if battle.won:
            value += 1.0
        elif battle.lost:
            value -= 1.0
        return value

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)

        # Add the terminal reward
        terminal_reward = self._state_value(battle) - self._prev_value
        if self._episode_rewards:
            self._episode_rewards[-1] += terminal_reward
        elif self._episode_obs:
            self._episode_rewards.append(terminal_reward)

        # Compute discounted returns
        if self._episode_obs:
            gamma = 0.99
            returns = np.zeros(len(self._episode_rewards))
            g = 0.0
            for t in reversed(range(len(self._episode_rewards))):
                g = self._episode_rewards[t] + gamma * g
                returns[t] = g

            self.all_obs.extend(self._episode_obs)
            self.all_returns.extend(returns.tolist())

        # Reset for next episode
        self._episode_obs = []
        self._episode_rewards = []
        self._prev_value = 0.0


async def collect_value_data(
    model: MaskablePPO,
    n_games: int,
    port: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """Play games with the BC policy and collect (obs, return) data."""
    server_config = ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    suffix = "".join(random.choices(string.ascii_lowercase, k=6))

    collector = ValueDataCollector(
        model=model,
        battle_format=BATTLE_FORMAT,
        account_configuration=AccountConfiguration(f"VF{suffix}", None),
        server_configuration=server_config,
        max_concurrent_battles=10,
        log_level=40,
    )

    # Play against a mix of opponents
    opponents = [
        RandomPlayer(
            battle_format=BATTLE_FORMAT,
            account_configuration=AccountConfiguration(f"VFOpp0{suffix}", None),
            server_configuration=server_config,
            log_level=40,
        ),
        MaxDamagePlayer(
            battle_format=BATTLE_FORMAT,
            account_configuration=AccountConfiguration(f"VFOpp1{suffix}", None),
            server_configuration=server_config,
            log_level=40,
        ),
        TypeMatchupPlayer(
            battle_format=BATTLE_FORMAT,
            account_configuration=AccountConfiguration(f"VFOpp2{suffix}", None),
            server_configuration=server_config,
            log_level=40,
        ),
        SmartHeuristicPlayer(
            battle_format=BATTLE_FORMAT,
            account_configuration=AccountConfiguration(f"VFOpp3{suffix}", None),
            server_configuration=server_config,
            log_level=40,
        ),
    ]

    games_per_opp = n_games // len(opponents)
    for opp in opponents:
        opp_name = opp.__class__.__name__
        print(f"  Playing {games_per_opp} games vs {opp_name}...")
        await collector.battle_against(opp, n_battles=games_per_opp)
        wins = sum(1 for b in collector.battles.values() if b.won)
        total = len(collector.battles)
        print(f"    Record so far: {wins}/{total} ({wins / total * 100:.0f}%)")

    obs = np.array(collector.all_obs, dtype=np.float32)
    returns = np.array(collector.all_returns, dtype=np.float32)
    print(f"\nCollected {len(obs)} (obs, return) pairs from {n_games} games")
    print(f"  Return range: [{returns.min():.3f}, {returns.max():.3f}], mean={returns.mean():.3f}")
    return obs, returns


def train_critic(
    model: MaskablePPO,
    obs: np.ndarray,
    returns: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> float:
    """Train the value head on (obs, return) pairs."""
    policy = model.policy
    device = policy.device

    # Split into train/val
    n = len(obs)
    indices = np.random.permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = indices[:split], indices[split:]

    t_obs = torch.tensor(obs[train_idx], device=device)
    t_ret = torch.tensor(returns[train_idx], device=device).unsqueeze(1)
    v_obs = torch.tensor(obs[val_idx], device=device)
    v_ret = torch.tensor(returns[val_idx], device=device).unsqueeze(1)

    # Only optimize value function parameters
    vf_params = (
        list(policy.vf_features_extractor.parameters())
        + list(policy.mlp_extractor.value_net.parameters())
        + list(policy.value_net.parameters())
    )
    optimizer = torch.optim.Adam(vf_params, lr=lr)

    n_batches = max(1, (len(t_obs) + batch_size - 1) // batch_size)

    print(f"\nTraining critic for {epochs} epochs, {n_batches} batches/epoch, lr={lr}")
    print(f"  Train: {len(t_obs)}, Val: {len(v_obs)}")
    print(f"{'Epoch':>5} | {'Train MSE':>10} | {'Val MSE':>10}")
    print("-" * 35)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        perm = torch.randperm(len(t_obs), device=device)

        policy.train()
        epoch_loss = 0.0
        batches_processed = 0

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(t_obs))
            idx = perm[start:end]
            b_obs = t_obs[idx]
            b_ret = t_ret[idx]

            # Forward pass through value network
            features = policy.extract_features(b_obs, policy.vf_features_extractor)
            latent_vf = policy.mlp_extractor.forward_critic(features)
            values = policy.value_net(latent_vf)

            loss = F.mse_loss(values, b_ret)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches_processed += 1

        epoch_loss /= max(batches_processed, 1)

        # Validation
        policy.eval()
        with torch.no_grad():
            features = policy.extract_features(v_obs, policy.vf_features_extractor)
            latent_vf = policy.mlp_extractor.forward_critic(features)
            values = policy.value_net(latent_vf)
            val_loss = F.mse_loss(values, v_ret).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = " *"
        else:
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
            print(f"{epoch + 1:>5} | {epoch_loss:>10.4f} | {val_loss:>10.4f}{marker}")

    print(f"\nBest validation MSE: {best_val_loss:.4f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/bc_warmstart", help="Path to BC model (without .zip)")
    parser.add_argument("--output", default="models/bc_warmstart", help="Output path (overwrites by default)")
    parser.add_argument("--n-games", type=int, default=1000, help="Games to play for value data")
    parser.add_argument("--epochs", type=int, default=30, help="Critic training epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--port", type=int, default=8000, help="Showdown server port")
    args = parser.parse_args()

    print(f"Loading BC model from {args.model}.zip...")
    model = MaskablePPO.load(args.model)

    print(f"\nCollecting value training data ({args.n_games} games)...")
    obs, returns = asyncio.run(collect_value_data(model, args.n_games, args.port))

    # Save the raw data in case we want to retrain
    data_path = Path("data/critic_training_data.npz")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, observations=obs, returns=returns)
    print(f"Saved value data to {data_path}")

    train_critic(model, obs, returns, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # Save the model with both warm-started policy and critic
    model.save(args.output)
    print(f"\nSaved model with warm-started critic to {args.output}.zip")


if __name__ == "__main__":
    main()
