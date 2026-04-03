"""
Two-tower feature extractor for Pokemon battles.

Splits the 1222-dim observation into own-team and opponent-team towers,
processes them independently, then merges with global features.

This architecture lets the network learn separate representations for
"my situation" vs "their situation" before combining them for decisions.
"""

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

# Observation layout (1222 dims total):
#   Own active   :   16  [0:16]
#   Own moves    :   80  [16:96]    (4 × 19 features + 4 target_statused)
#   Own bench    :  510  [96:606]   (6 × 85)
#   Opp active   :   15  [606:621]
#   Opp bench    :  510  [621:1131] (6 × 85)
#   Opp revealed :   76  [1131:1207] (4 × 19)
#   Global       :   15  [1207:1222] (trapping, speed, alive, volatile, status_threat, toxic)

_OWN_END = 606
_OPP_END = 1207
_OWN_DIM = _OWN_END  # 606
_OPP_DIM = _OPP_END - _OWN_END  # 601
_GLOBAL_DIM = 1222 - _OPP_END  # 15


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

        # Merge: 128 (own) + 128 (opp) + 15 (global) = 271 -> features_dim
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
