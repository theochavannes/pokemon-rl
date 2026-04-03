# Sprint 6: Neural Network Architecture + Encoding

Priority: MEDIUM — Ranked items #7-8, #11, #15-16 from panel #5.
Requires custom SB3 policy class for some items. Can be partially done with config changes.

---

## Step 1: VecNormalize wrapper

SB3's built-in observation + reward normalization. Free performance gain — normalizes all input features to zero mean, unit variance automatically. We've never used this despite expert panel #2 recommending it.

In `train.py`, wrap the DummyVecEnv:
```python
from stable_baselines3.common.vec_env import VecNormalize
env = VecNormalize(DummyVecEnv(env_fns), norm_obs=True, norm_reward=True, clip_obs=10.0)
```

**Caution:** Must save/load the normalization stats with the model. VecNormalize stats are separate from the model checkpoint.

---

## Step 2: Learning rate schedule

Start with higher LR (3e-4) for initial exploration, anneal to 1e-4 for refinement.

```python
from stable_baselines3.common.utils import linear_schedule
PPO_KWARGS["learning_rate"] = linear_schedule(3e-4)  # anneals to 0 over training
# Or custom: start 3e-4, end 1e-4
```

---

## Step 3: Two-tower architecture (custom policy)

Separate processing for own-team vs opponent info. Requires custom SB3 feature extractor.

```python
class PokemonFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # Own tower: active(16) + moves(56) + bench(390) = 462 dims
        self.own_tower = nn.Sequential(
            nn.Linear(462, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        # Opponent tower: active(15) + bench(390) + revealed(56) = 461 dims
        self.opp_tower = nn.Sequential(
            nn.Linear(461, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        # Global features: trapping(2) + speed_adv(1) + alive(2) = 5 dims
        # Merge: 128 + 128 + 5 = 261 -> features_dim
        self.merge = nn.Sequential(
            nn.Linear(261, features_dim), nn.ReLU(),
        )

    def forward(self, observations):
        own = observations[:, :462]
        opp = observations[:, 462:923]
        glob = observations[:, 923:]
        own_feat = self.own_tower(own)
        opp_feat = self.opp_tower(opp)
        return self.merge(torch.cat([own_feat, opp_feat, glob], dim=1))
```

Usage in train.py:
```python
policy_kwargs=dict(
    features_extractor_class=PokemonFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[128], vf=[128]),  # smaller heads since extractor is big
)
```

---

## Step 4: Learned type embeddings (custom feature extractor)

Replace type_index floats with 4-dim learned embeddings. The NN discovers that Fire and Ice are "opposite" while Fire and Fighting are "compatible."

Part of the custom feature extractor in Step 3:
```python
self.type_embedding = nn.Embedding(19, 4)  # 18 types + padding
```

Extract type indices from obs, look up embeddings, concatenate with other features.

**Note:** Requires changing how types are encoded in the obs vector — pass raw integer type index instead of normalized float.

---

## Step 5: Gradient clipping + optimizer tuning

With a larger network, gradient stability matters more:
```python
PPO_KWARGS["max_grad_norm"] = 0.5  # SB3 default, verify it's active
```

Consider AdamW instead of Adam for better weight decay behavior with larger networks.

---

## Estimated effort

- Steps 1-2: Config changes only, ~30 min
- Steps 3-4: Custom PyTorch code, ~2-3 hours
- Step 5: Config tuning, ~15 min

---

## Verification

1. VecNormalize: check obs stats converge (print running mean/std)
2. LR schedule: verify LR decreases in TensorBoard
3. Two-tower: compare parameter count and training speed vs flat MLP
4. Type embeddings: visualize learned embeddings — similar types should cluster
