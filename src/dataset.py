"""Build the per-match (11x7 home, 11x7 away, label, date) arrays from the
raw Kaggle CSVs. Player vectors are the 6 FIFA composite stats plus an
is_goalkeeper flag (pl1 in each lineup is always the goalkeeper).
"""
import numpy as np
import pandas as pd

from src.fifa_composites import COMPOSITE_NAMES, compute_composite

RAW_DIR = "data/raw"
PROCESSED_PATH = "data/processed/matches.npz"

N_PLAYERS = 11
N_FEATURES = len(COMPOSITE_NAMES) + 1  # + is_goalkeeper flag

RESULT_HOME_WIN = 0
RESULT_DRAW = 1
RESULT_AWAY_WIN = 2


def _team_matrix(row, side):
    """side: 'home' or 'away'. Returns (11, 7) array for one match."""
    mat = np.zeros((N_PLAYERS, N_FEATURES), dtype=np.float32)
    for i in range(1, N_PLAYERS + 1):
        prefix = f"{side}_pl{i}"
        is_gk = i == 1
        composites = compute_composite(row, prefix, is_gk)
        mat[i - 1, :-1] = [composites[name] for name in COMPOSITE_NAMES]
        mat[i - 1, -1] = float(is_gk)
    return mat


def build_dataset(raw_dir=RAW_DIR):
    lineup = pd.read_csv(f"{raw_dir}/Lineup_FifaPlayersAttributes.csv")
    matches = pd.read_csv(
        f"{raw_dir}/finalmerge_MATCHES.csv",
        usecols=[
            "id",
            "time_starting_at_date_time",
            "scores_home_score",
            "scores_away_score",
            "league_name",
        ],
    )
    df = lineup.merge(matches, on="id", how="inner")
    df = df.dropna(subset=["scores_home_score", "scores_away_score"])
    df["date"] = pd.to_datetime(df["time_starting_at_date_time"])
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    home_teams = np.zeros((n, N_PLAYERS, N_FEATURES), dtype=np.float32)
    away_teams = np.zeros((n, N_PLAYERS, N_FEATURES), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)

    for idx, row in df.iterrows():
        home_teams[idx] = _team_matrix(row, "home")
        away_teams[idx] = _team_matrix(row, "away")
        hs, aws = row["scores_home_score"], row["scores_away_score"]
        if hs > aws:
            labels[idx] = RESULT_HOME_WIN
        elif hs == aws:
            labels[idx] = RESULT_DRAW
        else:
            labels[idx] = RESULT_AWAY_WIN

    meta = df[
        ["id", "date", "league_name", "home_id", "away_id", "scores_home_score", "scores_away_score"]
    ].copy()

    # Scale composite stats (0-100 FIFA scale) to 0-1; leave is_goalkeeper flag as-is.
    home_teams[..., :-1] /= 100.0
    away_teams[..., :-1] /= 100.0

    # A handful of players have missing raw attributes upstream (~0.01% of cells);
    # drop the few matches affected rather than imputing.
    valid = ~(np.isnan(home_teams).any(axis=(1, 2)) | np.isnan(away_teams).any(axis=(1, 2)))
    if not valid.all():
        home_teams, away_teams, labels = home_teams[valid], away_teams[valid], labels[valid]
        meta = meta.loc[valid].reset_index(drop=True)

    return home_teams, away_teams, labels, meta


def save_dataset(path=PROCESSED_PATH):
    import os

    home, away, labels, meta = build_dataset()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        home=home,
        away=away,
        labels=labels,
        dates=meta["date"].values.astype("datetime64[ns]"),
        match_ids=meta["id"].values,
    )
    meta.to_csv(path.replace(".npz", "_meta.csv"), index=False)
    return home, away, labels, meta


def load_dataset(path=PROCESSED_PATH, rebuild=False):
    import os

    if rebuild or not os.path.exists(path):
        return save_dataset(path)
    data = np.load(path, allow_pickle=True)
    meta = pd.read_csv(path.replace(".npz", "_meta.csv"), parse_dates=["date"])
    return data["home"], data["away"], data["labels"], meta
