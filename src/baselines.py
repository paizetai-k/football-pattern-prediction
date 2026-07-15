"""Baselines to compare Phase 1 against, per the project plan:
- always predict home win
- simple Elo rating model (ratings -> ordered/multinomial logistic regression)
- bookmaker odds, overround-adjusted implied probabilities
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HOME_WIN, DRAW, AWAY_WIN = 0, 1, 2


def always_home_win_probs(n):
    probs = np.zeros((n, 3))
    probs[:, HOME_WIN] = 1.0
    return probs


def compute_elo_ratings(meta, labels, k=20, home_advantage=60, initial=1500):
    """Sequentially updates Elo ratings match-by-match in chronological order
    (meta must already be sorted by date). Returns array of pre-match
    elo_diff (home - away, home_advantage already added) for every match,
    usable as a leak-free feature.
    """
    teams = pd.unique(pd.concat([meta["home_id"], meta["away_id"]]))
    ratings = {t: float(initial) for t in teams}
    elo_diff = np.zeros(len(meta))

    for i, (_, row) in enumerate(meta.iterrows()):
        h, a = row["home_id"], row["away_id"]
        r_h, r_a = ratings[h], ratings[a]
        diff = r_h - r_a + home_advantage
        elo_diff[i] = diff

        expected_h = 1.0 / (1.0 + 10 ** (-diff / 400))
        result = labels[i]
        score_h = 1.0 if result == HOME_WIN else (0.5 if result == DRAW else 0.0)

        ratings[h] = r_h + k * (score_h - expected_h)
        ratings[a] = r_a + k * ((1 - score_h) - (1 - expected_h))

    return elo_diff


def elo_probs(meta, labels, train_mask, k=20, home_advantage=60, initial=1500):
    """Fits a multinomial logistic regression (elo_diff -> H/D/A) on the
    training portion, then returns probabilities for all matches. This is
    the 'Elo ordered logit' baseline referenced in the project plan.
    """
    elo_diff = compute_elo_ratings(meta, labels, k=k, home_advantage=home_advantage, initial=initial)
    X = elo_diff.reshape(-1, 1)

    clf = LogisticRegression(multi_class="multinomial", max_iter=1000)
    clf.fit(X[train_mask], np.asarray(labels)[train_mask])

    probs = np.zeros((len(labels), 3))
    class_order = clf.classes_
    raw_probs = clf.predict_proba(X)
    for i, c in enumerate(class_order):
        probs[:, c] = raw_probs[:, i]
    return probs


def bookmaker_probs(odds_df):
    """odds_df must have columns '3W__1_median', '3W__X_median', '3W__2_median'
    (decimal odds). Returns overround-adjusted implied probabilities, shape (N, 3).
    Rows with any missing odds get NaN probabilities.
    """
    home_odds = odds_df["3W__1_median"].to_numpy()
    draw_odds = odds_df["3W__X_median"].to_numpy()
    away_odds = odds_df["3W__2_median"].to_numpy()

    inv_home = 1.0 / home_odds
    inv_draw = 1.0 / draw_odds
    inv_away = 1.0 / away_odds
    overround = inv_home + inv_draw + inv_away

    probs = np.stack([inv_home, inv_draw, inv_away], axis=1) / overround[:, None]
    return probs
