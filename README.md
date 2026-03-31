# 🏆 MarchMadness2026

Kaggle competition entry — probabilistic prediction system for the 2026 NCAA March Madness tournament. Trained on historical data spanning 40+ years, with a custom Elo ranking engine and feature engineering pipeline that outperformed public betting baselines.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

## Results

- **Competition:** Google Kaggle — March Machine Learning Mania 2026
- **Metric:** Brier score / log-loss on tournament outcome probabilities
- **Outcome:** Outperformed public betting odds baseline on held-out tournament years

> 📌 Add your final Kaggle leaderboard score and ranking here.

---

## Approach

### 1. Custom Elo Rating System

Designed from scratch to model dynamic team strength over time:

- K-factor tuned per era to account for changes in game pace and competition structure
- Home court advantage modelled as a rating offset
- Season decay applied to prevent historical dominance from overwhelming recent form
- Ratings updated after every game across 40+ years of historical data

### 2. Feature Engineering

- Novel seeding and ranking features built on top of Elo ratings
- Aggregate team stats per season: offensive/defensive efficiency, strength of schedule (SOS)
- Historical head-to-head matchup records
- Upset frequency modelled by seed differential

### 3. Modelling Pipeline

- Logistic regression baseline → gradient boosted trees → ensemble
- Cross-validated on historical tournament data (held-out years as test sets)
- Probabilistic outputs calibrated using Platt scaling
- Evaluated with log-loss and scenario simulation

---

## Repo Structure

```
/data           — raw NCAA data from Kaggle (1985–2025)
/elo            — custom Elo rating engine
/features       — feature engineering pipeline
/models         — model training and evaluation scripts
/notebooks      — EDA and result analysis
predict.py      — generate submission predictions
```

---

## How to Run

### Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Run the full prediction pipeline

```bash
python predict.py --year 2026 --output submission.csv
```

### Evaluate against historical data

```bash
python models/evaluate.py --test-year 2024
```

---

## Key Findings

> - Elo ratings became more predictive than seed alone from the Sweet 16 onwards
> - Upsets in the first round were best predicted by strength-of-schedule differential, not seed gap
> - The model assigned X% win probability to the eventual champion before the tournament began
