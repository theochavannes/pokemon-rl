"""
Two-tower feature extractor for Pokemon battles.

Splits the 1559-dim observation into own-team and opponent-team towers,
processes them independently, then merges with global features.

This architecture lets the network learn separate representations for
"my situation" vs "their situation" before combining them for decisions.
"""

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

# Observation layout (1559 dims total):
#   Own active   :   16  [0:16]
#   Own moves    :  104  [16:120]   (4 × 25 features + 4 target_statused)
#   Own bench    :  654  [120:774]  (6 × 109)
#   Opp active   :   15  [774:789]
#   Opp bench    :  654  [789:1443] (6 × 109)
#   Opp revealed :  100  [1443:1543] (4 × 25)
#   Global       :   16  [1543:1559] (trapping, speed, alive, volatile, status, toxic, turn)

_OWN_END = 774
_OPP_END = 1543
_OWN_DIM = _OWN_END  # 774
_OPP_DIM = _OPP_END - _OWN_END  # 769
_GLOBAL_DIM = 1559 - _OPP_END  # 16


class PokemonFeatureExtractor(BaseFeaturesExtractor):
    """Two-tower architecture: separate own/opponent processing before merge."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self.own_tower = nn.Sequential(
            nn.Linear(_OWN_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.opp_tower = nn.Sequential(
            nn.Linear(_OPP_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Merge: 128 (own) + 128 (opp) + 16 (global) = 272 -> features_dim
        self.merge = nn.Sequential(
            nn.Linear(128 + 128 + _GLOBAL_DIM, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        own = observations[:, :_OWN_END]
        opp = observations[:, _OWN_END:_OPP_END]
        glob = observations[:, _OPP_END:]

        own_feat = self.own_tower(own)
        opp_feat = self.opp_tower(opp)
        return self.merge(torch.cat([own_feat, opp_feat, glob], dim=1))
