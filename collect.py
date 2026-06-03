import hashlib
import os
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from catanatron import Game, RandomPlayer
from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from features import create_sample_92
from schema import SCHEMA

BOT_CLASSES = (
    RandomPlayer,
    WeightedRandomPlayer,
    VictoryPointPlayer,
    AlphaBetaPlayer,
)
# All equal probability to be chose
BOT_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

GAME_ID_NAMESPACE = "catan_vf_v1"

TURNS_LIMIT = 1000



# game_id and seed helpers
def game_id_for(i: int):
    # Return the canonical game_id string for ordinal i
    return f"g{i:06d}"

# Return a uint64 seed that is stable across processes, platforms, and Python versions.
def stable_seed(game_id: str):
    digest = hashlib.sha256(
        f"{game_id}|{GAME_ID_NAMESPACE}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")

# Return a per-game random.Random instance seeded from stable_seed(game_id).
def per_game_rng(game_id: str):
    return random.Random(stable_seed(game_id))

# Return (p1_cls, p2_cls) drawn uniformly from BOT_CLASSES
def pick_pair(rng: random.Random):
    p1_cls = rng.choices(BOT_CLASSES, weights=BOT_WEIGHTS, k=1)[0]
    p2_cls = rng.choices(BOT_CLASSES, weights=BOT_WEIGHTS, k=1)[0]
    return p1_cls, p2_cls


# Return max visible VP across both players.
def compute_max_vp(player_state: dict):
    return max(player_state["P0_VICTORY_POINTS"], player_state["P1_VICTORY_POINTS"])

# Return the  string name for a bot class.
def _bot_name(cls: type):
    return cls.__name__

# Simulate one 2-player game and return (rows, summary).
def simulate_one_game(game_id: str):
    rng = per_game_rng(game_id)

    # Draw before Game construction
    p1_cls, p2_cls = pick_pair(rng)
    pov_color = rng.choice([Color.RED, Color.BLUE])

    seed = stable_seed(game_id)
    p1 = p1_cls(Color.RED)
    p2 = p2_cls(Color.BLUE)
    game = Game([p1, p2], seed=seed, vps_to_win=15)

    rows = []
    snapshot_idx = 0
    # Iterate via ticks (upper-bound on limts to avoid huge/outliers)
    while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT:
        # Capture pre-tick metadata.
        current_color = game.state.current_color()
        turn= game.state.num_turns

        game.play_tick()

        # Feature extraction from state AFTER the tick.
        sample = create_sample_92(game, pov_color)
        ps = game.state.player_state
        max_vp = compute_max_vp(ps)

        row = {
            "game_id": game_id,
            "snapshot_idx": snapshot_idx,
            "turn": turn,
            "current_color": current_color.value, 
            "pov_color": pov_color.value,
            "max_vp": int(max_vp),
            # Placeholder values — replaced after loop when winner is known.
            "label": 0,
            "game_complete": False,
            "p1_bot": _bot_name(p1_cls),
            "p2_bot": _bot_name(p2_cls),
            "seed": seed,
            **sample,  # 92 feature floats
        }
        rows.append(row)
        snapshot_idx += 1

    # Set true label and game_completed on every row (constant per game).
    winner = game.winning_color()
    game_completed = winner is not None
    label = 1 if winner == pov_color else 0
    for r in rows:
        r["label"] = label
        r["game_complete"] = game_completed

    summary = {
        "game_id": game_id,
        "game_complete": game_completed,
        "p1_bot": _bot_name(p1_cls),
        "p2_bot": _bot_name(p2_cls),
        "n_rows": len(rows),
    }
    return rows, summary


# Simulate game_id and write its Parquet file (always overwrites)
def simulate_and_write(game_id: str, raw_dir: Path | str):
    raw_dir = Path(raw_dir)
    final = raw_dir / f"{game_id}.parquet"

    rows, summary = simulate_one_game(game_id)
    tmp = final.with_suffix(".parquet.tmp")
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, final)

    return summary


# Run game collection in parallel using ProcessPoolExecutor
def collect_run(
    n: int,
    workers: int,
    raw_dir: Path,
    test_n: int
):

    target = test_n if test_n is not None else n
    all_ids = [game_id_for(i) for i in range(target)]
    raw_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    n_failed = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        max_tasks_per_child=1,
    ) as executor:
        futures = {
            executor.submit(simulate_and_write, g, str(raw_dir)): g
            for g in all_ids
        }
        # Progress bar
        with tqdm(total=len(all_ids), desc="collect") as bar:
            for fut in as_completed(futures):
                gid = futures[fut]
                try:
                    summaries.append(fut.result())
                except Exception as e:
                    n_failed += 1
                    print(
                        f"[ERROR] {gid}: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
                bar.update(1)

    # Truncation stats
    n_complete = sum(1 for s in summaries if s.get("game_complete"))
    n_done = len(summaries)
    truncation_rate = (n_done - n_complete) / n_done if n_done else 0.0

    # Per-pairing snapshot counts.
    per_pairing = {}
    for s in summaries:
        key = (s["p1_bot"], s["p2_bot"])
        per_pairing[key] = per_pairing.get(key, 0) + s.get("n_rows", 0)

    return {
        "n_total": len(all_ids),
        "n_run": n_done,
        "n_failed": n_failed,
        "truncation_rate": truncation_rate,
        "summaries": summaries,
        "per_pairing_snapshot_counts": per_pairing,
    }


# Direct-run entry point: `python collect.py [--n N] [--test-n N] [--workers W] [--raw-dir PATH]`
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Catanatron games into per-game parquet files.")
    # Target number of games (default: 15000).
    parser.add_argument("--n", type=int, default=15000)
    # Run only N games and exit (smoke-test mode).
    parser.add_argument("--test-n", type=int, default=None)
    # Parallel worker count.
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    # Output directory for per-game parquet files.
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    result = collect_run(n=args.n, workers=args.workers, raw_dir=args.raw_dir, test_n=args.test_n)
    print(f"n_total={result['n_total']}")
    print(f"n_run={result['n_run']}")
    print(f"n_failed={result['n_failed']}")
    print(f"truncation_rate={result['truncation_rate']:.4f}")
    if result.get("per_pairing_snapshot_counts"):
        print("per_pairing_snapshot_counts:")
        for k, v in sorted(result["per_pairing_snapshot_counts"].items()):
            print(f"  {k}: {v}")