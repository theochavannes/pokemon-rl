"""
FrozenPolicyPlayer — a poke-env Player that uses a saved MaskablePPO model.

Used as the opponent in self-play training. The model is loaded once and never
updated (frozen), so the learning agent always plays against a fixed snapshot.

Usage:
    opponent = FrozenPolicyPlayer(
        model_path="models/best_model",
        battle_format="gen1randombattle",
        account_configuration=AccountConfiguration("Frozen0abc", None),
        start_listening=False,
    )
"""

import numpy as np
from poke_env.player.player import Player
from poke_env.environment.singles_env import SinglesEnv


class FrozenPolicyPlayer(Player):
    """
    Plays using a frozen MaskablePPO checkpoint.

    The policy is never updated — call swap_model() to load a newer snapshot
    during self-play curriculum without rebuilding the environment.
    """

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(path)

    def swap_model(self, new_path: str) -> None:
        """Hot-swap to a newer checkpoint without rebuilding the environment."""
        self.model_path = new_path
        self._load_model(new_path)

    def choose_move(self, battle):
        from src.env.gen1_env import embed_battle
        obs = embed_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return SinglesEnv.action_to_order(action, battle, strict=False)
