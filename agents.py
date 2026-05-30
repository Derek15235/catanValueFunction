"""agents.py — GreedyValuePlayer + BucketRouter + make_player

Used by eval_online.py to wrap a trained sklearn Pipeline (LR or XGB,
unified or per-bucket) in a one-step lookahead Catanatron player.
"""
from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np

from catanatron.game import Game
from catanatron.models.player import Color, Player

from features import create_sample_92
from schema import FEATURE_ORDERING

VP_BUCKETS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 99)]


class BucketRouter:
    def __init__(self, unified, bucket_models):
        self.unified = unified
        self.bucket_models = bucket_models
        self.unified_fallback_uses = 0
        self.total_picks = 0

    @classmethod
    def from_dir(cls, family_dir):
        family_dir = Path(family_dir)
        unified_path = family_dir / "pipeline_unified.joblib"
        if not unified_path.exists():
            raise FileNotFoundError(
                f"required unified pipeline missing at {unified_path}"
            )
        unified = joblib.load(unified_path)

        bucket_models = {}
        for low, high in VP_BUCKETS:
            p = family_dir / f"pipeline_vp_{low:02d}-{min(high, 15):02d}.joblib"
            if p.exists():
                bucket_models[(low, high)] = joblib.load(p)

        return cls(unified, bucket_models)

    def pick(self, game):
        ps = game.state.player_state
        max_vp = max(ps["P0_VICTORY_POINTS"], ps["P1_VICTORY_POINTS"])
        # Round up due to how we made our buckets
        if max_vp < 2:
            max_vp = 2
        
        self.total_picks += 1
        # Search for bucket which the game applies to (find the approate VP range)
        for low, high in VP_BUCKETS:
            if low <= max_vp < high:
                bucket_key = (low, high)
                break
        else:
            bucket_key = (15, 99)
        # Return the appropriate model and have a failsafe case where it just uses the unified model instead 
        model = self.bucket_models.get(bucket_key)
        if model is None:
            self.unified_fallback_uses += 1
            return self.unified
        return model


class GreedyValuePlayer(Player):
    def __init__(self, color, model=None, bucket_router=None):
        super().__init__(color)
        if (model is None) == (bucket_router is None):
            raise ValueError(
                "GreedyValuePlayer requires exactly one of model OR bucket_router"
            )
        self.model = model
        self.bucket_router = bucket_router

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]


        if self.bucket_router is not None:
            pipeline = self.bucket_router.pick(game)
        else:
            pipeline = self.model

        scores = []
        # Search for best action
        for action in playable_actions:
            g2 = copy.deepcopy(game)
            g2.execute(action)

            sample = create_sample_92(g2, self.color)
            feat = np.array(
                [sample[f] for f in FEATURE_ORDERING], dtype=np.float32
            ).reshape(1, -1)
            scores.append(float(pipeline.predict_proba(feat)[0, 1]))

        best_idx = int(np.argmax(scores))
        return playable_actions[best_idx]


def make_player(color, model, bucket_router=None):
    return GreedyValuePlayer(color, model=model, bucket_router=bucket_router)
