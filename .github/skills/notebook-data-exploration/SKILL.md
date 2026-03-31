---
name: notebook-data-exploration
description: >
  Workflows and patterns for exploratory data analysis (EDA) in Jupyter notebooks,
  specifically for the March Machine Learning Mania (mm2026) competition dataset.
  Use this skill whenever the user wants to explore data, understand a dataset, profile
  variables, visualize distributions, check for missing values, examine relationships,
  or do any kind of "getting to know the data" work — even if they just say "what does
  this data look like?", "explore the tournament data", or "show me the distributions".
  Works alongside the jupyter-mcp skill for notebook operations.
---

# Notebook Data Exploration

This skill covers EDA workflows for the mm2026 competition data. The goal is to build
genuine understanding of the data before modelling — not just generate boilerplate
describe() calls, but actually surface patterns that matter for predicting tournament
outcomes.

> **Load alongside jupyter-mcp** for notebook operations (connecting, executing cells, etc.)

---

## The mm2026 Data Landscape

The data lives in `../data/landing/` relative to any notebook:

```
data/landing/
├── mens/          M*.csv   — Men's tournament data (history from 1985)
├── womens/        W*.csv   — Women's tournament data (history from 1998)
├── SampleSubmission*.csv  — Target format for predictions
└── Cities.csv, Conferences.csv, etc.
```

### Key tables to know

| File | What it contains |
|------|-----------------|
| `M/WNCAATourneyCompactResults.csv` | Game outcomes: WTeamID, LTeamID, WScore, LScore, DayNum |
| `M/WNCAATourneySeeds.csv` | Seed strings per team per season (e.g. `W01`, `Z16a`) |
| `M/WRegularSeasonCompactResults.csv` | Regular season outcomes — feature engineering gold |
| `M/WTeams.csv` | TeamID → TeamName mapping |
| `M/WMasseyOrdinals.csv` | Third-party ranking systems (Ken Pomeroy, etc.) — very useful |
| `M/WNCAATourneyDetailedResults.csv` | Box score stats (FGM, FGA, Rebounds, etc.) |
| `SampleSubmissionStage1.csv` | All possible matchup pairs: `ID` format is `YYYY_TeamA_TeamB` |

---

## Standard EDA Workflow

### 1. Environment setup cell

```python
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path("../data/landing")
mens_dir = DATA_DIR / "mens"
womens_dir = DATA_DIR / "womens"

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
```

### 2. Load and profile a table

```python
df = pl.read_csv(mens_dir / "MNCAATourneyCompactResults.csv")

# Shape and schema
print(f"Shape: {df.shape}")
print(df.schema)

# Quick profile
print(df.describe())

# Null check
print(df.null_count())

# Sample rows
df.head(10)
```

### 3. Seed parsing (critical for this competition)

Seeds encode tournament region and seeding. Parse them early:

```python
import re

def parse_seed(seed_str: str) -> int:
    """Extract numeric seed from strings like 'W01', 'Z16a'."""
    return int(re.search(r'(\d+)', seed_str).group(1))

seeds = pl.read_csv(mens_dir / "MNCAATourneySeeds.csv")
seeds = seeds.with_columns(
    pl.col("Seed").map_elements(parse_seed, return_dtype=pl.Int32).alias("SeedNum"),
    pl.col("Seed").str.slice(0, 1).alias("Region")
)
```

### 4. Win rate and upset analysis

```python
results = pl.read_csv(mens_dir / "MNCAATourneyCompactResults.csv")
seeds_parsed = ...  # from above

# Join seeds to results
results_with_seeds = (
    results
    .join(seeds_parsed.rename({"TeamID": "WTeamID", "SeedNum": "WSeed"}),
          on=["Season", "WTeamID"])
    .join(seeds_parsed.rename({"TeamID": "LTeamID", "SeedNum": "LSeed"}),
          on=["Season", "LTeamID"])
    .with_columns((pl.col("WSeed") - pl.col("LSeed")).alias("SeedDiff"))
)

# Upset rate by seed matchup
upset_rate = (
    results_with_seeds
    .filter(pl.col("SeedDiff") > 0)   # higher seed (worse) won
    .group_by("WSeed", "LSeed")
    .agg(pl.len().alias("n_upsets"))
    .sort("n_upsets", descending=True)
)
```

### 5. Score margin distributions

```python
results = results.with_columns(
    (pl.col("WScore") - pl.col("LScore")).alias("Margin")
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
results["Margin"].to_pandas().hist(bins=40, ax=axes[0], edgecolor='black')
axes[0].set_title("Score Margin Distribution")

# By round
results.group_by("DayNum").agg(pl.mean("Margin")).sort("DayNum").to_pandas().plot(
    x="DayNum", y="Margin", ax=axes[1], marker='o'
)
axes[1].set_title("Avg Margin by Tournament Day")
plt.tight_layout()
```

### 6. Year-over-year trends

```python
season_stats = (
    results
    .group_by("Season")
    .agg(
        pl.len().alias("n_games"),
        pl.mean("Margin").alias("avg_margin"),
        pl.mean("WScore").alias("avg_winning_score"),
    )
    .sort("Season")
)

season_stats.to_pandas().plot(x="Season", y=["avg_margin", "avg_winning_score"],
                               figsize=(12, 4), subplots=True)
plt.suptitle("Tournament Trends Over Time")
```

---

## Common Patterns

### Checking data availability by season

```python
# Before engineering features, always know what years have data
for fname in mens_dir.glob("*.csv"):
    df = pl.read_csv(fname)
    if "Season" in df.columns:
        seasons = df["Season"].unique().sort()
        print(f"{fname.name}: {seasons[0]}–{seasons[-1]} ({len(seasons)} seasons)")
```

### Building the "regular season strength" feature baseline

```python
reg = pl.read_csv(mens_dir / "MRegularSeasonCompactResults.csv")

# Win percentage per team per season
wins = (
    reg.group_by(["Season", "WTeamID"])
    .agg(pl.len().alias("wins"))
    .rename({"WTeamID": "TeamID"})
)
losses = (
    reg.group_by(["Season", "LTeamID"])
    .agg(pl.len().alias("losses"))
    .rename({"LTeamID": "TeamID"})
)
team_stats = (
    wins.join(losses, on=["Season", "TeamID"], how="outer_coalesce")
    .with_columns(
        pl.col("wins").fill_null(0),
        pl.col("losses").fill_null(0),
    )
    .with_columns(
        (pl.col("wins") / (pl.col("wins") + pl.col("losses"))).alias("win_pct")
    )
)
```

### Correlating features with upset probability

```python
# Use execute_code for quick checks — not saved to notebook
# JupyterMCP-execute_code: results_with_seeds.select("SeedDiff", "Margin").corr()
```

---

## EDA Checklist

Before moving to feature engineering, confirm you've checked:

- [ ] Shape and null counts for each table you'll use
- [ ] Season coverage (men's goes back to 1985, women's to 1998)
- [ ] Seed distribution — are all 64/68 seeds present each year?
- [ ] Score distributions — any outliers?
- [ ] Class balance — tournament games are ~50/50 by definition (symmetric pairs)
- [ ] 2025 data availability and recency — validate carefully before trusting

---

## Polars Tips for This Dataset

**Prefer lazy evaluation for large joins:**
```python
result = (
    pl.scan_csv(mens_dir / "MRegularSeasonCompactResults.csv")
    .filter(pl.col("Season") >= 2010)
    .collect()
)
```

**String seed parsing in Polars expressions:**
```python
seeds.with_columns(
    pl.col("Seed").str.extract(r'(\d+)', 1).cast(pl.Int32).alias("SeedNum")
)
```

---

## Related Skills

- **jupyter-mcp** — notebook operations (connecting, cell execution, etc.)
- **notebook-ml** — next step: building models from the features you've explored
