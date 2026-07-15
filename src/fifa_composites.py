"""Derive the 6 standard FIFA composite stats (Pace/Shooting/Passing/Dribbling/
Defending/Physical) from the ~33 raw FIFA-index attributes, since this dataset
only scraped the raw attributes, not the pre-computed composites.

Weights approximate the publicly documented EA groupings. 'Composure' is
excluded (dropped upstream for missing data), its Dribbling weight folded
into BallControl.
"""

OUTFIELD_COMPOSITES = {
    "pace": {"Acceleration": 0.5, "SprintSpeed": 0.5},
    "shooting": {
        "Finishing": 0.45,
        "LongShots": 0.20,
        "ShotPower": 0.20,
        "Att.Position": 0.05,
        "Penalties": 0.05,
        "Volleys": 0.05,
    },
    "passing": {
        "ShortPass": 0.30,
        "Vision": 0.20,
        "Crossing": 0.20,
        "LongPass": 0.20,
        "FKAcc.": 0.10,
    },
    "dribbling": {
        "BallControl": 0.45,
        "Dribbling": 0.20,
        "Reactions": 0.20,
        "Agility": 0.10,
        "Balance": 0.05,
    },
    "defending": {
        "Marking": 0.30,
        "StandTackle": 0.25,
        "SlideTackle": 0.20,
        "Interceptions": 0.15,
        "Heading": 0.10,
    },
    "physical": {
        "Aggression": 0.40,
        "Strength": 0.30,
        "Stamina": 0.25,
        "Jumping": 0.05,
    },
}

GOALKEEPER_COMPOSITES = {
    "pace": {"GKDiving": 1.0},
    "shooting": {"GKKicking": 1.0},
    "passing": {"GKHandling": 1.0},
    "dribbling": {"GKReflexes": 1.0},
    "defending": {"GKPositioning": 1.0},
    "physical": {"Strength": 0.5, "Jumping": 0.5},
}

COMPOSITE_NAMES = ["pace", "shooting", "passing", "dribbling", "defending", "physical"]


def compute_composite(row, prefix, is_goalkeeper):
    """Compute the 6 composite stats for one player from a wide dataframe row.

    row: pandas Series (or dict-like) containing '{prefix}_{Attribute}' columns
    prefix: e.g. 'home_pl3'
    is_goalkeeper: bool
    Returns dict of composite_name -> float
    """
    weights = GOALKEEPER_COMPOSITES if is_goalkeeper else OUTFIELD_COMPOSITES
    out = {}
    for composite, attr_weights in weights.items():
        total = 0.0
        for attr, w in attr_weights.items():
            total += row[f"{prefix}_{attr}"] * w
        out[composite] = total
    return out
