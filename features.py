"""
Features (6 extractors → 92 named columns):
    player_features           17  keys  (P0_/P1_ prefixes)
    resource_hand_features    23  keys  (P0_*_IN_HAND / P1_*_IN_HAND)
    game_features_compat       8  keys  (BANK_*, IS_*)
    production_effective      10  keys  (EFFECTIVE_*)
    production_total          10  keys  (TOTAL_*)
    port_distance_features    24  keys  (P0_*_PORT* / P1_*_PORT*)
    ─────────────────────────────
    Total                     92
"""
from catanatron.game import Game
from catanatron.models.decks import freqdeck_count
from catanatron.models.enums import ActionType, RESOURCES
from catanatron.models.map import build_map
from catanatron.models.player import Color, SimplePlayer
from catanatron_gym.features import (
    build_production_features,
    player_features,
    port_distance_features,
    resource_hand_features,
)

# Alternative to catanatron_gym.game_features but renaming for the sake of consistency and 
def game_features_adjustment(game: Game, pov_color: Color) -> dict[str, float]:
    possible_actions = {a.action_type for a in game.playable_actions}
    features: dict[str, float] = {
        "BANK_DEV_CARDS": len(game.state.development_listdeck),
        "IS_MOVING_ROBBER": ActionType.MOVE_ROBBER in possible_actions,
        "IS_DISCARDING": ActionType.DISCARD_RESOURCE in possible_actions,
    }
    for resource in RESOURCES:
        features[f"BANK_{resource}"] = freqdeck_count(
            game.state.resource_freqdeck, resource
        )
    return features


PORT_DISTANCE_STANDIN = 0.0


# Extractor list for the different features we can pull from a game state
CUSTOM_EXTRACTORS = [
    player_features,
    resource_hand_features,
    game_features_compat,
    build_production_features(consider_robber=True),   # EFFECTIVE_* prefix
    build_production_features(consider_robber=False),  # TOTAL_* prefix
    port_distance_features,
]



# Based on game output the features to be used at the current state
def create_sample_92(game: Game, pov_color: Color):
    record: dict[str, float] = {}
    for ex in CUSTOM_EXTRACTORS:
        record.update(ex(game, pov_color))

    # Sanitize port_distance unreachable-port sentinel.
    # See RESEARCH.md Pitfall 4: inf paired with HAS_*_PORT=0 means
    # "no port reachable"; we collapse to 0.0 so float32 + StandardScaler work.
    inf = float("inf")
    for k, v in list(record.items()):
        if isinstance(v, float) and (v == inf or v == -inf):
            record[k] = PORT_DISTANCE_STANDIN
    return record



# Constructs a fresh BASE 2-player game, calls create_sample_92, and
# returns sorted(sample.keys())
def build_feature_ordering() -> list[str]:
    """Return the canonical alphabetically-sorted list of 92 feature names.

    Constructs a fresh BASE 2-player game, calls create_sample_92, and
    returns sorted(sample.keys()). The result is deterministic because the
    set of keys is fixed by the extractor implementations regardless of
    game state.

    Raises AssertionError if the count is not exactly 92.
    """
    game = Game(
        [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)],
        catan_map=build_map("BASE"),
    )
    sample = create_sample_92(game, Color.RED)
    names = sorted(sample.keys())
    if len(names) != 92:
        raise AssertionError(
            f"Expected 92 features, got {len(names)}. "
            f"Composition expected: 17+23+8+10+10+24=92."
        )
    return names
