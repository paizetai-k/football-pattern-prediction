"""RPS (ranked probability score) and log-loss for 3-class (H/D/A) predictions.

Classes are assumed ordered [home_win, draw, away_win] so the cumulative sums
in RPS reflect the natural ranking of outcomes.
"""
import numpy as np
from sklearn.metrics import log_loss as sk_log_loss


def ranked_probability_score(probs, labels, n_classes=3):
    """probs: (N, n_classes) predicted probabilities.
    labels: (N,) integer class indices.
    Returns mean RPS (lower is better, 0 = perfect).
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    n = len(labels)
    outcomes = np.zeros((n, n_classes))
    outcomes[np.arange(n), labels] = 1.0

    cum_probs = np.cumsum(probs, axis=1)
    cum_outcomes = np.cumsum(outcomes, axis=1)

    rps = np.sum((cum_probs - cum_outcomes) ** 2, axis=1) / (n_classes - 1)
    return float(np.mean(rps))


def log_loss_score(probs, labels, n_classes=3):
    return float(sk_log_loss(labels, probs, labels=list(range(n_classes))))


def accuracy_score(probs, labels):
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == np.asarray(labels)))
