# Copilot Instructions — March Machine Learning Mania 2026

## Environment

**Pixi is the only permitted environment manager for this project.** All dependency changes go through `pixi.toml` followed by `pixi install`.

### Setup & Usage
```bash
pixi install                       # install / sync dependencies
pixi shell                         # activate environment
pixi run ingest_data_into_landing  # download competition data from Kaggle
```

**Jupyter:**
- Jupyter Lab runs inside the Pixi environment.
- **Use the default `Python 3` kernel.** No special kernel registration is required.
- Launch command (for humans): `pixi run open-jupyter-lab` (or just `jupyter lab` from a pixi shell).

**Adding Dependencies:**
- Edit `[dependencies]` in `pixi.toml`.
- Run `pixi install`.
- **Never** use `pip install` or `conda install` directly.

## Architecture

**Data Flow:**
```
data/landing/   ← Kaggle download (extracted)
notebooks/      ← analysis and modelling
scripts/        ← reusable scripts
```

**Data Structure:**
- `data/landing/mens/` — Men's competition data (`M*.csv`)
- `data/landing/womens/` — Women's competition data (`W*.csv`)
- `data/landing/` — Shared files (`Cities.csv`, `Conferences.csv`, `SampleSubmission*.csv`)

**Note:** The `data/` directory is gitignored. Ensure `KAGGLE_API_TOKEN` is set in `.env` or your environment before running ingestion tasks.

## Key Conventions

**Data Science Stack:**
- **Prefer Polars over Pandas** for data manipulation.
- Use **scikit-learn** / **XGBoost** / **LightGBM** for modelling.

**Notebooks:**
- **Location:** `notebooks/`
- **Naming:** Prefix with initials, e.g., `jl.feature_engineering.ipynb`.
- **Scratchpad:** Use `notebooks/crappy/` for experiments.
- **Paths:** Always use relative paths: `../data/landing/`.

**Code Snippets:**
```python
# Standard Data Loading
import polars as pl
from pathlib import Path

DATA_DIR = Path("../data/landing")
mens_dir = DATA_DIR / "mens"
womens_dir = DATA_DIR / "womens"

# Example: Load compact results
m_results = pl.read_csv(mens_dir / "MNCAATourneyCompactResults.csv")
```

**Competition Specifics:**
- **Target:** Win probability for every possible matchup (Stage 1 & 2).
- **Seeds:** Parse strings like `W01`, `Z16a`.
  - Region: `W`, `Z` (first char)
  - Seed: `01`, `16` (digits)
  - Extract int: `int(re.search(r'(\d+)', seed_str).group(1))`
- **History:** Men's (since 1985), Women's (since 1998).
- **Validation:** 2025 data may be available; validate carefully against historical trends.
