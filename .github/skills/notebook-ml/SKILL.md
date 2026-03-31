---
name: notebook-ml
description: >
  ML modelling workflows for the March Machine Learning Mania (mm2026) competition in
  Jupyter notebooks. Use this skill whenever building, training, evaluating, or submitting
  predictive models for tournament win probabilities — even if the user says "build a model",
  "train a classifier", "generate predictions", "improve the model", "feature engineering",
  "cross-validate", or "make a submission". This skill covers the full pipeline from
  feature construction to Kaggle submission format. Works alongside jupyter-mcp and
  notebook-data-exploration.
---

# Notebook ML — Modelling for March Mania

This skill covers the modelling pipeline for predicting NCAA tournament win probabilities.
The target is **win probability for every possible matchup**, submitted as a CSV of
`(Season_TeamA_TeamB, Pred)` pairs where Pred is the probability TeamA beats TeamB.

> **Load alongside jupyter-mcp** for notebook operations. Run EDA first if you haven't
> explored the data yet (see notebook-data-exploration skill).

---

## Competition Structure

**Task:** Predict P(TeamA beats TeamB) for all possible matchup pairs in Stage 1 (historical
seasons) and Stage 2 (current tournament).

**Evaluation:** Log loss — well-calibrated probabilities matter as much as ranking order.
Never predict exactly 0 or 1; clip predictions to [0.025, 0.975].

**Submission ID format:** `{Season}_{TeamA}_{TeamB}` where TeamA < TeamB (always lower ID first).

**Target construction:** From `MNCAATourneyCompactResults.csv`, for each game create a row:
- `ID = {Season}_{min(WTeamID, LTeamID)}_{max(WTeamID, LTeamID)}`
- `Result = 1` if the lower-ID team won, else `0`

---

## Standard Pipeline

### 1. Feature Engineering

Build per-team-per-season features, then combine into matchup-level differences:

```python
import polars as pl
import numpy as np
from pathlib import Path

DATA_DIR = Path("../data/landing")
mens_dir = DATA_DIR / "mens"

# ── Regular season win rate ─────────────────────────────────────────────────
reg = pl.read_csv(mens_dir / "MRegularSeasonCompactResults.csv")

def team_season_stats(results: pl.DataFrame) -> pl.DataFrame:
    wins = (results.group_by(["Season", "WTeamID"])
            .agg(pl.len().alias("wins"),
                 pl.mean("WScore").alias("avg_pts_for"),
                 pl.mean("LScore").alias("avg_pts_against"))
            .rename({"WTeamID": "TeamID"}))

    losses = (results.group_by(["Season", "LTeamID"])
              .agg(pl.len().alias("losses"))
              .rename({"LTeamID": "TeamID"}))

    return (wins.join(losses, on=["Season", "TeamID"], how="outer_coalesce")
            .with_columns(pl.col("wins").fill_null(0), pl.col("losses").fill_null(0))
            .with_columns(
                (pl.col("wins") / (pl.col("wins") + pl.col("losses"))).alias("win_pct"),
                (pl.col("avg_pts_for") - pl.col("avg_pts_against")).alias("point_diff")
            ))

team_features = team_season_stats(reg)

# ── Seed features ────────────────────────────────────────────────────────────
import re

seeds = pl.read_csv(mens_dir / "MNCAATourneySeeds.csv")
seeds = seeds.with_columns(
    pl.col("Seed").str.extract(r'(\d+)', 1).cast(pl.Int32).alias("SeedNum")
)
```

### 2. Building the Training Set

Construct symmetric matchup rows from historical tournament results:

```python
tourney = pl.read_csv(mens_dir / "MNCAATourneyCompactResults.csv")

def build_matchup_df(results, team_feats, seeds_parsed):
    """Create one row per game: winner = lower TeamID."""
    df = results.with_columns([
        pl.when(pl.col("WTeamID") < pl.col("LTeamID"))
          .then(pl.col("WTeamID")).otherwise(pl.col("LTeamID")).alias("Team1"),
        pl.when(pl.col("WTeamID") < pl.col("LTeamID"))
          .then(pl.col("LTeamID")).otherwise(pl.col("WTeamID")).alias("Team2"),
        pl.when(pl.col("WTeamID") < pl.col("LTeamID"))
          .then(1).otherwise(0).alias("Result"),
    ]).select(["Season", "Team1", "Team2", "Result"])

    # Join features for both teams
    f1 = team_feats.rename({c: f"T1_{c}" for c in team_feats.columns if c not in ["Season", "TeamID"]})
    f2 = team_feats.rename({c: f"T2_{c}" for c in team_feats.columns if c not in ["Season", "TeamID"]})

    s1 = seeds_parsed.rename({"TeamID": "Team1", "SeedNum": "T1_Seed"}).drop("Seed", "Region", strict=False)
    s2 = seeds_parsed.rename({"TeamID": "Team2", "SeedNum": "T2_Seed"}).drop("Seed", "Region", strict=False)

    return (df
            .join(f1.rename({"TeamID": "Team1"}), on=["Season", "Team1"])
            .join(f2.rename({"TeamID": "Team2"}), on=["Season", "Team2"])
            .join(s1, on=["Season", "Team1"], how="left")
            .join(s2, on=["Season", "Team2"], how="left"))

train_df = build_matchup_df(tourney, team_features, seeds)
```

### 3. Feature Differencing

Most powerful features are **deltas** between teams, not raw values:

```python
feature_cols = ["win_pct", "point_diff", "avg_pts_for", "SeedNum"]

for col in feature_cols:
    if f"T1_{col}" in train_df.columns and f"T2_{col}" in train_df.columns:
        train_df = train_df.with_columns(
            (pl.col(f"T1_{col}") - pl.col(f"T2_{col}")).alias(f"diff_{col}")
        )
```

### 4. Model Training

Start with logistic regression as the calibrated baseline, then gradient boosting:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import xgboost as xgb

FEATURE_COLS = [c for c in train_df.columns if c.startswith("diff_")]
X = train_df[FEATURE_COLS].to_pandas()
y = train_df["Result"].to_pandas()

# ── Logistic Regression baseline ─────────────────────────────────────────────
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=1000))
])
lr_scores = cross_val_score(lr_pipe, X, y, cv=5, scoring="neg_log_loss")
print(f"LR CV log-loss: {-lr_scores.mean():.4f} ± {lr_scores.std():.4f}")

# ── XGBoost ───────────────────────────────────────────────────────────────────
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
xgb_scores = cross_val_score(xgb_model, X, y, cv=5, scoring="neg_log_loss")
print(f"XGB CV log-loss: {-xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
```

### 5. Temporal Validation

For tournament prediction, **never use future data to predict past seasons**.
Leave-one-season-out is more realistic than random k-fold:

```python
from sklearn.model_selection import LeaveOneGroupOut

seasons = train_df["Season"].to_pandas()
logo = LeaveOneGroupOut()
logo_scores = cross_val_score(lr_pipe, X, y, groups=seasons,
                               cv=logo, scoring="neg_log_loss")
print(f"LOGO CV log-loss: {-logo_scores.mean():.4f} ± {logo_scores.std():.4f}")

# Also look at performance per season to spot degradation
for season, score in sorted(zip(seasons.unique(), logo_scores)):
    print(f"  {season}: {-score:.4f}")
```

### 6. Generating Submission

```python
# Load sample submission to get all required matchup IDs
sub_template = pl.read_csv(DATA_DIR / "SampleSubmissionStage1.csv")

# Parse IDs: "2024_1101_1102" → Season, Team1, Team2
sub_df = sub_template.with_columns([
    pl.col("ID").str.split("_").list.get(0).cast(pl.Int32).alias("Season"),
    pl.col("ID").str.split("_").list.get(1).cast(pl.Int32).alias("Team1"),
    pl.col("ID").str.split("_").list.get(2).cast(pl.Int32).alias("Team2"),
])

# Join features for submission matchups
sub_features = build_matchup_df(
    sub_df.rename({"Team1": "WTeamID", "Team2": "LTeamID"})
          .with_columns(pl.lit(0).alias("WScore"), pl.lit(0).alias("LScore")),
    team_features, seeds
)

# Predict and clip
lr_pipe.fit(X, y)  # retrain on full data
X_sub = sub_features[FEATURE_COLS].to_pandas().fillna(0.5)
sub_preds = lr_pipe.predict_proba(X_sub)[:, 1]
sub_preds = np.clip(sub_preds, 0.025, 0.975)

submission = pd.DataFrame({"ID": sub_template["ID"].to_list(), "Pred": sub_preds})
submission.to_csv("../data/submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows")
print(submission.describe())
```

---

## Modelling Checklist

- [ ] Features are built on data available **before** each tournament game
- [ ] No data leakage: regular season features only, not tournament outcomes
- [ ] Symmetric matchup construction: always Team1 < Team2 by ID
- [ ] Predictions clipped to [0.025, 0.975] — never 0 or 1
- [ ] Validated with temporal (leave-one-season-out) CV, not just random k-fold
- [ ] Submission ID format matches sample submission exactly
- [ ] Sanity check: average prediction ≈ 0.5 (tournament is roughly symmetric)

---

## Feature Engineering Ideas (in priority order)

1. **Seed difference** — single strongest predictor; simple but hard to beat
2. **Regular season win %** — overall team strength signal
3. **Point differential** — quality of wins, not just W/L
4. **Massey ordinals** — third-party power rankings (multiple systems available)
5. **Recent form** — last N games of regular season
6. **Strength of schedule** — who did they beat?
7. **Detailed box stats** — FG%, 3P%, turnover rate from `DetailedResults`
8. **Coach/program history** — tournament experience (requires joining team metadata)

---

## Common Pitfalls

**Data leakage:** Don't use tournament-year data to build features for that year's tournament.
If using Massey ordinals, filter to `RankingDayNum < 133` (before tournament starts).

**Seed availability:** Seeds are only available for tournament teams (~64-68 per year).
Regular season teams without seeds need imputation or should be excluded.

**Symmetric submissions:** The sample submission always has Team1 < Team2. Make sure your
feature differencing is consistent (T1 features minus T2 features, where T1 has lower ID).

**Calibration matters more than AUC:** Log loss penalises confident wrong predictions
heavily. A well-calibrated model beats an overconfident one.

---

## Related Skills

- **jupyter-mcp** — notebook operations
- **notebook-data-exploration** — EDA and data profiling before modelling
