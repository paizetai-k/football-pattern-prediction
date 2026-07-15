# Data

Source: [enricocattaneo/data-football-match-prediction](https://www.kaggle.com/datasets/enricocattaneo/data-football-match-prediction) on Kaggle.

`data/raw/` is gitignored (large, easy to regenerate). To fetch it:

```bash
pip install kagglehub
python3 -c "import kagglehub; print(kagglehub.dataset_download('enricocattaneo/data-football-match-prediction'))"
```

Then copy these three files from the printed path into `data/raw/`:

- `Data/From_Preparation/Lineup_FifaPlayersAttributes.csv` — per-match starting XI with FIFA attributes snapped to the closest date before the match
- `Data/From_Preparation/finalmerge_MATCHES.csv` — match results, teams, venue, stats
- `Data/Modeling_Final/5_odds.csv` — aggregated bookmaker odds (3-way market: `3W__1/X/2_median` etc.)

`data/processed/matches.npz` (also gitignored) is built automatically by `src/dataset.py` the first time `src/train.py` runs, and cached for subsequent runs.

Note: the raw player attributes are the ~33 individual FIFA-index stats (BallControl, Marking, Acceleration, ...), not the pre-computed 6-category composites (Pace/Shooting/Passing/Dribbling/Defending/Physical). `src/fifa_composites.py` derives the 6 composites from the raw attributes using the standard public EA weighting groups.
