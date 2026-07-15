"""Phase 1: shared player-encoder MLP + mean-pooling (no attention).

Each player's 7-dim vector (6 composite stats + is_goalkeeper flag) is passed
through a shared MLP, mean-pooled over the 11 players to get one team vector,
then both team vectors are concatenated with match context and classified.
"""
import torch
import torch.nn as nn


class PlayerEncoder(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=32, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, players):
        # players: (batch, 11, in_dim) -> (batch, 11, out_dim)
        return self.net(players)


class Phase1Model(nn.Module):
    def __init__(self, player_dim=7, player_hidden=32, player_out=16, context_dim=0, clf_hidden=32):
        super().__init__()
        self.encoder = PlayerEncoder(player_dim, player_hidden, player_out)
        clf_in = player_out * 2 + context_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, clf_hidden),
            nn.ReLU(),
            nn.Linear(clf_hidden, 3),  # home win / draw / away win
        )

    def forward(self, home_players, away_players, context=None):
        home_emb = self.encoder(home_players).mean(dim=1)  # (batch, out_dim)
        away_emb = self.encoder(away_players).mean(dim=1)
        parts = [home_emb, away_emb]
        if context is not None:
            parts.append(context)
        combined = torch.cat(parts, dim=1)
        return self.classifier(combined)  # logits, (batch, 3)
