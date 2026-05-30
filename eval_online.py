"""eval_online.py

Plays 4 agents x 4 baselines x N games per pairing, both seats balanced,
writes results/eval/online.json with Wilson CIs + loss-mode breakdown.

Usage:
    uv run python eval_online.py --pilot 25                  # pilot canary
    uv run python eval_online.py                             # full 9,600-game run
    uv run python eval_online.py --agents lr-unified         # subset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest
from tqdm import tqdm

from catanatron import Game, RandomPlayer
from catanatron.models.enums import ActionType
from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from agents import BucketRouter, make_player


VP_BUCKETS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 99)]
N_PER_PAIRING = 600
TRUNCATION_ALARM = 0.02
TURNS_LIMIT = 1000
DEFAULT_WORKERS = min(os.cpu_count() or 1, 8)
RESULTS_DIR = Path("results/eval")
PARTIAL_PATH = RESULTS_DIR / "online.partial.json"
FINAL_PATH = RESULTS_DIR / "online.json"

AGENT_FAMILIES = {
    "lr-unified": ("lr",  "unified"),
    "lr-per_bucket": ("lr",  "per_bucket"),
    "xgb-unified": ("gbt", "unified"),
    "xgb-per_bucket": ("gbt", "per_bucket"),
}

BASELINES = {
    "RandomPlayer": RandomPlayer,
    "WeightedRandomPlayer": WeightedRandomPlayer,
    "VictoryPointPlayer": VictoryPointPlayer,
    "AlphaBetaPlayer": AlphaBetaPlayer,
}


def stable_seed(key_str: str) -> int:
    digest = hashlib.sha256(f"{key_str}|catan_vf_v1".encode()).digest()
    return int.from_bytes(digest[:8], "big")


# Per-worker model cache, populated once by _init_worker. Keyed by family ("lr"/"gbt")
# so lr-unified + lr-per_bucket share the same loaded pipelines.
_MODEL_CACHE: dict = {}


def _init_worker(agent_names):
    families = {AGENT_FAMILIES[name][0] for name in agent_names}
    for family in families:
        family_dir = Path(f"results/{family}")
        unified = joblib.load(family_dir / "pipeline_unified.joblib")
        bucket_models = {}
        for low, high in VP_BUCKETS:
            p = family_dir / f"pipeline_vp_{low:02d}-{min(high, 15):02d}.joblib"
            if p.exists():
                bucket_models[(low, high)] = joblib.load(p)
        _MODEL_CACHE[family] = {"unified": unified, "bucket_models": bucket_models}


def loss_mode_within_1ply(trace, agent_color, winner_color) -> bool:
    # Return true iff opponent reached 15 VP in the window between agent's last END_TURN
    # and opponent's next END_TURN. Pure function over the per-tick trace.

    if winner_color is None or winner_color == agent_color:
        return False

    # Find the agent's last END_TURN tick (scan from the end).
    last_agent_end_turn_idx = None
    for i in range(len(trace) - 1, -1, -1):
        t = trace[i]
        if t["acting_color"] == agent_color and t["action_type"] == ActionType.END_TURN:
            last_agent_end_turn_idx = i
            break
    if last_agent_end_turn_idx is None:
        return False

    opponent_color = Color.BLUE if agent_color == Color.RED else Color.RED

    # Right edge: first opponent END_TURN after the agent's last END_TURN.
    # If the game ended mid-opponent-turn, use end of trace.
    right_edge = len(trace)
    for j in range(last_agent_end_turn_idx + 1, len(trace)):
        t = trace[j]
        if t["acting_color"] == opponent_color and t["action_type"] == ActionType.END_TURN:
            right_edge = j + 1
            break

    window = trace[last_agent_end_turn_idx + 1 : right_edge]
    opp_key = "p1_vp_after" if agent_color == Color.RED else "p0_vp_after"
    return any(t[opp_key] >= 15 for t in window)


def play_one_game(pairing_label, seat_assignment, replicate_idx, agent_spec, baseline_class):
    seed = stable_seed(f"{pairing_label}|{seat_assignment}|{replicate_idx}")
    # Re-seed global RNG every game so worker reuse stays byte-deterministic
    # (Game.__init__ mutates global RNG — Phase 1 pitfall).
    random.seed(seed)
    np.random.seed(seed % (2**32))

    family = agent_spec["family"]
    mode = agent_spec["mode"]
    cache = _MODEL_CACHE[family]

    if mode == "unified":
        pipe = cache["unified"]
        router = None
        agent_factory = lambda color: make_player(color, pipe, bucket_router=None)
    else:
        router = BucketRouter(cache["unified"], cache["bucket_models"])
        agent_factory = lambda color: make_player(color, None, bucket_router=router)

    agent_color = Color.RED if seat_assignment == "RED" else Color.BLUE
    opp_color = Color.BLUE if agent_color == Color.RED else Color.RED
    agent = agent_factory(agent_color)
    opp = baseline_class(opp_color)
    players = [agent, opp] if agent_color == Color.RED else [opp, agent]

    game = Game(players, seed=seed, vps_to_win=15)

    trace = []
    tick_idx = 0
    while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT:
        game.play_tick()
        records = game.state.action_records
        if not records:
            tick_idx += 1
            continue
        last = records[-1]
        ps = game.state.player_state
        trace.append({
            "tick_idx": tick_idx,
            "acting_color": last.action.color,
            "action_type": last.action.action_type,
            "max_vp_after": max(ps["P0_VICTORY_POINTS"], ps["P1_VICTORY_POINTS"]),
            "p0_vp_after": ps["P0_VICTORY_POINTS"],
            "p1_vp_after": ps["P1_VICTORY_POINTS"],
        })
        tick_idx += 1

    winner = game.winning_color()
    truncated = winner is None
    agent_won = (winner == agent_color)

    hit_1ply = False if truncated else loss_mode_within_1ply(trace, agent_color, winner)

    # Hidden-VP-card detection via state delta at game end
    hit_hidden = False
    if not truncated and not agent_won:
        opp_prefix = "P1" if agent_color == Color.RED else "P0"
        actual = game.state.player_state.get(f"{opp_prefix}_ACTUAL_VICTORY_POINTS", 0)
        visible = game.state.player_state.get(f"{opp_prefix}_VICTORY_POINTS", 0)
        if actual > visible:
            hit_hidden = True

    return {
        "pairing_label": pairing_label,
        "seat_assignment": seat_assignment,
        "replicate_idx": replicate_idx,
        "winner_color": winner.value if winner else None,
        "agent_color": agent_color.value,
        "truncated": truncated,
        "agent_won": bool(agent_won),
        "hit_loss_within_1ply": bool(hit_1ply),
        "hit_loss_hidden_vp": bool(hit_hidden),
        "unified_fallback_uses": router.unified_fallback_uses if router is not None else 0,
        "total_router_picks": router.total_picks if router is not None else 0,
    }


def wilson_ci(wins: int, n_resolved: int):
    if n_resolved == 0:
        return (0.0, 1.0)
    res = binomtest(k=wins, n=n_resolved)
    lo, hi = res.proportion_ci(method="wilson", confidence_level=0.95)
    return (float(lo), float(hi))


def aggregate_pairing(per_game_results):
    n_games = len(per_game_results)
    n_truncations = sum(1 for r in per_game_results if r["truncated"])
    n_resolved = n_games - n_truncations

    wins = sum(1 for r in per_game_results if r["agent_won"])
    
    truncation_rate = n_truncations / n_games if n_games else 0.0
    win_rate = wins / n_resolved if n_resolved else 0.0
    wilson_lo, wilson_hi = wilson_ci(wins, n_resolved)

    n_resolved_losses = sum(
        1 for r in per_game_results if not r["truncated"] and not r["agent_won"]
    )
    n_loss_within_1ply = sum(1 for r in per_game_results if r["hit_loss_within_1ply"])
    n_loss_hidden_vp = sum(1 for r in per_game_results if r["hit_loss_hidden_vp"])

    total_router_picks = sum(r["total_router_picks"] for r in per_game_results)
    unified_fallback_uses = sum(r["unified_fallback_uses"] for r in per_game_results)
    # None for unified agents (no router); 0.0 for per-bucket agents that never fell back.
    unified_fallback_rate = (
        (unified_fallback_uses / total_router_picks) if total_router_picks else None
    )

    return {
        "n_games": n_games,
        "n_resolved": n_resolved,
        "n_truncations": n_truncations,
        "truncation_rate": truncation_rate,
        "wins": wins,
        "win_rate": win_rate,
        "wilson_lo": wilson_lo,
        "wilson_hi": wilson_hi,
        "unified_fallback_rate": unified_fallback_rate,
        "loss_mode": {
            "n_resolved_losses": n_resolved_losses,
            "n_loss_within_1ply": n_loss_within_1ply,
            "n_loss_hidden_vp": n_loss_hidden_vp,
        },
    }


def build_agent_spec(agent_name: str):
    family, mode = AGENT_FAMILIES[agent_name]
    return {"family": family, "mode": mode}


def run_pairing(agent_name, baseline_name, n_per_pairing, executor):
    pairing_label = f"{agent_name}_vs_{baseline_name}"
    half = n_per_pairing // 2
    jobs = [("RED", i) for i in range(half)]
    jobs += [("BLUE", i) for i in range(n_per_pairing - half)]
    red_n = sum(1 for s, _ in jobs if s == "RED")
    blue_n = sum(1 for s, _ in jobs if s == "BLUE")
    assert abs(red_n - blue_n) <= 1, f"seat imbalance: RED={red_n} BLUE={blue_n}"

    agent_spec = build_agent_spec(agent_name)
    baseline_class = BASELINES[baseline_name]

    futures = {
        executor.submit(
            play_one_game, pairing_label, seat, rep_idx, agent_spec, baseline_class
        ): (seat, rep_idx)
        for (seat, rep_idx) in jobs
    }

    per_game = []
    # Progress bar for pairing
    with tqdm(total=len(jobs), desc=pairing_label, leave=False) as bar:
        for fut in as_completed(futures):
            try:
                per_game.append(fut.result())
            except Exception as e:
                seat, rep_idx = futures[fut]
                print(
                    f"[ERROR] {pairing_label} {seat} {rep_idx}: {e}",
                    file=sys.stderr,
                )
                raise
            bar.update(1)

    return aggregate_pairing(per_game)


def validate_models_present(agent_names):
    for name in agent_names:
        family, _ = AGENT_FAMILIES[name]
        path = Path(f"results/{family}") / "pipeline_unified.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"required unified pipeline missing for agent '{name}': {path} "
                f"(run train_{'logreg' if family == 'lr' else 'gbt'}.py first)"
            )


def write_partial(results_so_far):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PARTIAL_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results_so_far, indent=2))
    os.replace(tmp, PARTIAL_PATH)


def main():
    # Argumment commands
    parser = argparse.ArgumentParser(description="Phase 4 online evaluation runner.")
    parser.add_argument("--n", type=int, default=N_PER_PAIRING,
                        help=f"Games per pairing (default: {N_PER_PAIRING}).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Worker processes (default: {DEFAULT_WORKERS}).")
    parser.add_argument("--pilot", type=int, default=None,
                        help="If set, run only the first pairing with this many games.")
    parser.add_argument("--agents", type=str,
                        default=",".join(AGENT_FAMILIES.keys()),
                        help="Comma-separated subset of agent names.")
    args = parser.parse_args()

    agent_list = [a.strip() for a in args.agents.split(",") if a.strip()]
    for a in agent_list:
        if a not in AGENT_FAMILIES:
            raise ValueError(
                f"unknown agent '{a}'; valid: {list(AGENT_FAMILIES.keys())}"
            )

    validate_models_present(agent_list)

    # Insertion order defines the pairing-iteration order, first pairing is the cheap
    # one (lr-unified vs RandomPlayer) when both dicts are unmodified
    pairings = [(a, b) for a in agent_list for b in BASELINES]
    n_per_pairing = args.n

    if args.pilot is not None:
        pairings = pairings[:1]
        n_per_pairing = args.pilot
        print(f"[pilot] running {pairings[0][0]} vs {pairings[0][1]} for {n_per_pairing} games",
              file=sys.stderr)

    results = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(agent_list,),
    ) as executor:
        for agent_name, baseline_name in pairings:
            agg = run_pairing(agent_name, baseline_name, n_per_pairing, executor)
            results.setdefault(agent_name, {})[baseline_name] = agg
            write_partial(results)

            label = f"{agent_name}_vs_{baseline_name}"
            print(
                f"[{label}] {agg['wins']}/{agg['n_resolved']} wins, "
                f"{agg['n_truncations']} truncations, "
                f"win_rate={agg['win_rate']:.3f} "
                f"[{agg['wilson_lo']:.3f}, {agg['wilson_hi']:.3f}]",
                file=sys.stderr,
            )
            if agg["truncation_rate"] > TRUNCATION_ALARM:
                print(
                    f"[WARN] {label} truncation_rate={agg['truncation_rate']:.3f} "
                    f"exceeds alarm threshold {TRUNCATION_ALARM}",
                    file=sys.stderr,
                )

    if args.pilot is None:
        os.replace(PARTIAL_PATH, FINAL_PATH)
        print(f"[done] wrote {FINAL_PATH}", file=sys.stderr)
    else:
        print(f"[pilot done] partial results at {PARTIAL_PATH} (not promoted to final)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
