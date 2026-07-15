"""Phase 1 end-to-end: load data, time-based split, train the pooled-MLP
model, and report RPS/log-loss against the always-home-win, Elo, and
bookmaker-odds baselines on the same held-out matches.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import baselines
from src.dataset import load_dataset
from src.metrics import accuracy_score, log_loss_score, ranked_probability_score
from src.model import Phase1Model

TEST_FRACTION = 0.2
SEED = 0


def time_based_split(meta, test_fraction=TEST_FRACTION):
    n = len(meta)
    n_test = int(n * test_fraction)
    n_train = n - n_test
    train_idx = np.arange(0, n_train)
    test_idx = np.arange(n_train, n)
    return train_idx, test_idx


def train_phase1_model(home, away, labels, train_idx, epochs=30, lr=1e-3, batch_size=64, seed=SEED):
    torch.manual_seed(seed)
    model = Phase1Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(
        torch.from_numpy(home[train_idx]),
        torch.from_numpy(away[train_idx]),
        torch.from_numpy(labels[train_idx]),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for h, a, y in loader:
            optimizer.zero_grad()
            logits = model(h, a)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        avg_loss = total_loss / len(train_idx)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:>3}/{epochs}  train loss {avg_loss:.4f}")
    return model


def predict_probs(model, home, away):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(home), torch.from_numpy(away))
        probs = torch.softmax(logits, dim=1).numpy()
    return probs


def report(name, probs, labels, mask=None):
    if mask is not None:
        probs, labels = probs[mask], np.asarray(labels)[mask]
    rps = ranked_probability_score(probs, labels)
    ll = log_loss_score(probs, labels)
    acc = accuracy_score(probs, labels)
    n = len(labels)
    print(f"{name:<28} n={n:<6} accuracy={acc:.3f}  log_loss={ll:.4f}  RPS={rps:.4f}")


def main():
    print("Loading dataset...")
    home, away, labels, meta = load_dataset()
    print(f"  {len(meta)} matches, {meta['date'].min().date()} to {meta['date'].max().date()}")

    train_idx, test_idx = time_based_split(meta)
    print(
        f"Time-based split: train={len(train_idx)} "
        f"({meta['date'].iloc[train_idx[0]].date()}-{meta['date'].iloc[train_idx[-1]].date()}), "
        f"test={len(test_idx)} "
        f"({meta['date'].iloc[test_idx[0]].date()}-{meta['date'].iloc[test_idx[-1]].date()})"
    )

    print("\nTraining Phase 1 model (shared MLP encoder + mean pool)...")
    model = train_phase1_model(home, away, labels, train_idx)
    model_probs = predict_probs(model, home, away)

    print("\n=== Held-out test set results ===")
    report("Phase 1 (pooled MLP)", model_probs, labels, mask=test_idx)
    report("Always home win", baselines.always_home_win_probs(len(labels)), labels, mask=test_idx)

    print("\nFitting Elo ordered-logit baseline...")
    train_mask = np.zeros(len(labels), dtype=bool)
    train_mask[train_idx] = True
    elo_p = baselines.elo_probs(meta, labels, train_mask)
    report("Elo ordered logit", elo_p, labels, mask=test_idx)

    print("\nLoading bookmaker odds baseline...")
    odds = pd.read_csv(
        "data/raw/5_odds.csv", usecols=["id", "3W__1_median", "3W__X_median", "3W__2_median"]
    )
    odds_merged = meta.merge(odds, on="id", how="left")
    odds_available = odds_merged[["3W__1_median", "3W__X_median", "3W__2_median"]].notna().all(axis=1)
    bookmaker_p = np.full((len(labels), 3), np.nan)
    bookmaker_p[odds_available.to_numpy()] = baselines.bookmaker_probs(
        odds_merged.loc[odds_available]
    )
    odds_test_mask = np.zeros(len(labels), dtype=bool)
    odds_test_mask[test_idx] = True
    odds_test_mask &= odds_available.to_numpy()
    report(f"Bookmaker odds (n_odds={odds_available.sum()})", bookmaker_p, labels, mask=odds_test_mask)


if __name__ == "__main__":
    main()
