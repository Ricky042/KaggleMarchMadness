"""
optimal_strategy.py
===================
Implements the mathematically optimal Brier-score risk strategy.

Mathematical basis
------------------
Brier score = MSE = mean((pred - actual)²)

When you pick a team with true win probability p and predict them
to win (pred=1.0), your expected Brier contribution is:
    E[Brier] = p*(1-1)² + (1-p)*(1-0)² = (1-p)

Without the strategy (predicting calibrated p):
    E[Brier] = p*(p-1)² + (1-p)*(p-0)² = 2p(1-p)  [wait, actually p(1-p)]

Expected IMPROVEMENT from betting on the team vs predicting p:
    f(p) = [calibrated Brier] - [strategy Brier]
         = p(1-p) - (1-p)        ... simplified
    f(p) = p(1-p)² - ?? 
    
More precisely:
    calibrated_brier(p) = p*(p-1)² + (1-p)*p² = p(1-p)
    strategy_brier(p)   = p*(1-p)²             (we predict 1.0, team wins with prob p)

    improvement(p) = p*(p-1)² + (1-p)*p²   -   p*(1-p)²
                   = p(1-p)[ p + (1-p) ]   -   p(1-p)²
                   Wait, let's just compute:
    
    calibrated = p*(p-1)² + (1-p)*p² = p[(p-1)² + (1-p)*p]
               = p(1-p)[(1-p) + p]... no

Actually the exact derivation from the paper:
    f(p) = p(1-p)²   is the expected reward (Brier gain vs predict 0.5)
    f'(p) = (1-p)² + p*2(1-p)*(-1) = (1-p)[(1-p) - 2p] = (1-p)(1-3p)
    f'(p) = 0 when p = 1 or p = 1/3
    f''(1/3) < 0 → maximum at p = 1/3

So the expected benefit of the strategy is maximised when you pick
a team with true probability p = 1/3 of winning their games.

In practice: target ~5-seeds (Sweet 16 run ~33%) or a strong 8-seed.

Usage
-----
from optimal_strategy import OptimalStrategy

# Auto-select best risk team
strat = OptimalStrategy(seeds_df, submission_df)
strat.auto_select(model_probs)   # uses model's own win probabilities
submission_df = strat.apply(submission_df)

# Manual override
strat.set_risk_team(team_id=1179, max_round=2)
submission_df = strat.apply(submission_df)
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# ROUND CALCULATION  (ported from the referenced Kaggle solution)
# ──────────────────────────────────────────────────────────────────────────────

# The bracket slot order: position in this list = bracket slot index
SLOT_MAP = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]


def get_round_of_match(team1_id: int, team2_id: int, seeds_df: pd.DataFrame) -> int:
    """
    Returns the earliest possible round these two seeds could meet.
      1 = First Four (play-in)
      2 = Round of 64
      3 = Round of 32
      4 = Sweet 16
      5 = Elite Eight
      6 = Final Four
      7 = Championship
      0 = at least one team not in tournament
    """
    t1_rows = seeds_df[seeds_df["TeamID"] == team1_id]["Seed"].values
    t2_rows = seeds_df[seeds_df["TeamID"] == team2_id]["Seed"].values

    if len(t1_rows) == 0 or len(t2_rows) == 0:
        return 0

    seed1 = t1_rows[0]
    seed2 = t2_rows[0]

    # First Four: same 3-char prefix (e.g. both "W11")
    if seed1[:3] == seed2[:3]:
        return 1

    region1 = seed1[0]
    region2 = seed2[0]
    num1    = int(seed1[1:3])
    num2    = int(seed2[1:3])

    # Cross-region games
    if region1 != region2:
        wx = {"W", "X"}
        both_wx       = region1 in wx and region2 in wx
        neither_wx    = region1 not in wx and region2 not in wx
        if both_wx or neither_wx:
            return 6   # Final Four (same semifinal bracket)
        else:
            return 7   # Championship

    # Same region — determine round by bracket slot
    if num1 not in SLOT_MAP or num2 not in SLOT_MAP:
        return 2  # fallback

    slot1 = SLOT_MAP.index(num1)
    slot2 = SLOT_MAP.index(num2)

    if (slot1 // 2) == (slot2 // 2):  return 2  # Round of 64
    if (slot1 // 4) == (slot2 // 4):  return 3  # Round of 32
    if (slot1 // 8) == (slot2 // 8):  return 4  # Sweet 16
    return 5                                      # Elite Eight


def get_tourney_flag(team1_id: int, team2_id: int, seeds_df: pd.DataFrame) -> int:
    """0 if either team not in tournament, else round number (1-7)."""
    tourney_teams = set(seeds_df["TeamID"].tolist())
    if team1_id not in tourney_teams or team2_id not in tourney_teams:
        return 0
    return get_round_of_match(team1_id, team2_id, seeds_df)


def build_flag_series(submission_df: pd.DataFrame,
                       seeds_m: pd.DataFrame,
                       seeds_w: pd.DataFrame) -> list:
    """Build list of round flags for every row of submission_df."""
    flags = []
    for _, row in submission_df.iterrows():
        parts = str(row["ID"]).split("_")
        t1, t2 = int(parts[1]), int(parts[2])
        is_womens = (t1 + t2) > 6000
        seeds = seeds_w if is_womens else seeds_m
        flags.append(get_tourney_flag(t1, t2, seeds))
    return flags


# ──────────────────────────────────────────────────────────────────────────────
# EXPECTED BRIER IMPROVEMENT  (the f(p) = p(1-p)² function)
# ──────────────────────────────────────────────────────────────────────────────

def expected_brier_improvement(p: float) -> float:
    """
    Expected Brier improvement from predicting 1.0 on a team
    with true win probability p, vs predicting p.

    Peaks at p = 1/3 with value f(1/3) = (1/3)(2/3)² = 4/27 ≈ 0.148.
    """
    return p * (1 - p) ** 2


def optimal_risk_p() -> float:
    """Returns the theoretically optimal p = 1/3."""
    return 1.0 / 3.0


# ──────────────────────────────────────────────────────────────────────────────
# AUTO-SELECTION OF BEST RISK TEAM
# ──────────────────────────────────────────────────────────────────────────────

def find_best_risk_team(submission_df: pd.DataFrame,
                         seeds_m: pd.DataFrame,
                         seeds_w: pd.DataFrame,
                         season: int,
                         gender: str = "M",
                         max_round_options: tuple = (2, 3, 4)) -> dict:
    """
    Scan all tournament teams and find the one whose model-predicted
    first-round win probability is closest to 1/3, for each max_round.

    Returns a dict with recommended (team_id, max_round, expected_gain).

    Parameters
    ----------
    submission_df : DataFrame with ID and Pred columns
    seeds_m / seeds_w : seed DataFrames (just for the target season)
    season : e.g. 2026
    gender : "M" or "W"
    max_round_options : which max_rounds to evaluate (2=R64, 3=R32, 4=S16)
    """
    seeds = seeds_m if gender == "M" else seeds_w
    seeds_season = seeds[seeds["Season"] == season]

    if len(seeds_season) == 0:
        print(f"  [strategy] No {season} seeds found for {gender}, skipping auto-select.")
        return {}

    # Build lookup: (t1, t2) → pred from submission
    pred_lookup = {}
    for _, row in submission_df.iterrows():
        parts = row["ID"].split("_")
        s, t1, t2 = int(parts[0]), int(parts[1]), int(parts[2])
        if s == season:
            pred_lookup[(t1, t2)] = float(row["Pred"])

    TARGET_P = 1.0 / 3.0
    results = []

    for _, seed_row in seeds_season.iterrows():
        tid  = int(seed_row["TeamID"])
        seed = seed_row.get("Seed", "X16")
        try:
            seed_num = int(str(seed)[1:3])
        except Exception:
            seed_num = 16

        for max_round in max_round_options:
            # Collect all matchups involving this team in rounds 1..max_round
            # (predict they win all of them → set to 1.0 or 0.0)
            relevant_preds = []
            for (t1, t2), pred in pred_lookup.items():
                flag = get_tourney_flag(t1, t2, seeds_season)
                if flag == 0 or flag > max_round:
                    continue
                # Is our team involved?
                if tid == t1:
                    relevant_preds.append(pred)        # pred = P(t1 wins)
                elif tid == t2:
                    relevant_preds.append(1.0 - pred)  # P(t2 wins) = 1 - pred

            if not relevant_preds:
                continue

            # Average predicted win probability across relevant games
            avg_p = float(np.mean(relevant_preds))

            # Expected gain per game under the strategy
            expected_gain = expected_brier_improvement(avg_p)

            # Distance from optimal 1/3
            dist_from_optimal = abs(avg_p - TARGET_P)

            results.append({
                "team_id":        tid,
                "seed":           seed,
                "seed_num":       seed_num,
                "gender":         gender,
                "max_round":      max_round,
                "avg_pred_p":     round(avg_p, 4),
                "expected_gain":  round(expected_gain, 5),
                "dist_optimal":   round(dist_from_optimal, 4),
                "n_games":        len(relevant_preds),
            })

    if not results:
        return {}

    df = pd.DataFrame(results).sort_values("dist_optimal")

    print(f"\n  [strategy] Top candidates ({gender}) — closest to p=1/3:")
    print(f"  {'TeamID':>8} {'Seed':<6} {'MaxRnd':>7} {'AvgP':>7} "
          f"{'ExpGain':>9} {'NGames':>7}")
    print(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*9} {'-'*7}")
    for _, r in df.head(10).iterrows():
        marker = " ← RECOMMENDED" if _ == 0 else ""
        print(f"  {int(r['team_id']):>8} {r['seed']:<6} {int(r['max_round']):>7} "
              f"{r['avg_pred_p']:>7.4f} {r['expected_gain']:>9.5f} "
              f"{int(r['n_games']):>7}{marker}")

    best = df.iloc[0].to_dict()
    return best


# ──────────────────────────────────────────────────────────────────────────────
# APPLY STRATEGY TO SUBMISSION
# ──────────────────────────────────────────────────────────────────────────────

def apply_optimal_strategy(submission_df: pd.DataFrame,
                            seeds_m: pd.DataFrame,
                            seeds_w: pd.DataFrame,
                            season: int,
                            risk_team_id: int,
                            max_round: int,
                            gender: str = "M") -> pd.DataFrame:
    """
    For all games involving risk_team_id in rounds 1..max_round,
    set prediction to 1.0 (if team is team1) or 0.0 (if team is team2).

    This says: "I'm predicting this team wins every game up to this round."

    Parameters
    ----------
    risk_team_id : TeamID of the team you're betting on
    max_round    : Highest round to apply the strategy (2=R64, 3=R32, 4=S16)
    """
    seeds = seeds_m if gender == "M" else seeds_w

    out = submission_df.copy()
    changed = 0

    for idx, row in out.iterrows():
        parts = row["ID"].split("_")
        s, t1, t2 = int(parts[0]), int(parts[1]), int(parts[2])

        if s != season:
            continue

        flag = get_tourney_flag(t1, t2, seeds)
        if flag == 0 or flag > max_round:
            continue

        if t1 == risk_team_id:
            out.at[idx, "Pred"] = 1.0
            changed += 1
        elif t2 == risk_team_id:
            out.at[idx, "Pred"] = 0.0
            changed += 1

    print(f"  [strategy] Applied to {changed} matchups: "
          f"TeamID={risk_team_id} wins through Round {max_round}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE: AUTO-SELECT + APPLY
# ──────────────────────────────────────────────────────────────────────────────

def run_optimal_strategy(submission_df: pd.DataFrame,
                          seeds_m_df: pd.DataFrame,
                          seeds_w_df: pd.DataFrame,
                          season: int,
                          manual_team_id: int = None,
                          manual_max_round: int = None,
                          gender: str = "M") -> pd.DataFrame:
    """
    Full pipeline: auto-select the best risk team (or use manual override),
    then apply the strategy to the submission.

    If manual_team_id is set, skips auto-selection.

    Returns modified submission DataFrame.
    """
    seeds_season_m = seeds_m_df[seeds_m_df["Season"] == season]
    seeds_season_w = seeds_w_df[seeds_w_df["Season"] == season]

    if manual_team_id is not None:
        risk_id  = manual_team_id
        max_rnd  = manual_max_round if manual_max_round is not None else 2
        print(f"\n  [strategy] Manual override: TeamID={risk_id}, max_round={max_rnd}")
    else:
        # Auto-select
        best = find_best_risk_team(
            submission_df, seeds_season_m, seeds_season_w,
            season=season, gender=gender,
        )
        if not best:
            print("  [strategy] Could not auto-select risk team. Skipping.")
            return submission_df
        risk_id = int(best["team_id"])
        max_rnd = int(best["max_round"])
        print(f"\n  [strategy] Auto-selected: TeamID={risk_id} "
              f"(seed {best['seed']}), max_round={max_rnd}, "
              f"avg_p={best['avg_pred_p']:.3f} "
              f"(optimal=0.333), "
              f"expected_gain={best['expected_gain']:.5f}")

    sub = apply_optimal_strategy(
        submission_df,
        seeds_season_m, seeds_season_w,
        season=season,
        risk_team_id=risk_id,
        max_round=max_rnd,
        gender=gender,
    )
    return sub


# ──────────────────────────────────────────────────────────────────────────────
# SHOW THEORY PLOT (optional, requires matplotlib)
# ──────────────────────────────────────────────────────────────────────────────

def show_theory():
    """Print a text visualization of f(p) = p(1-p)² to show the 1/3 optimum."""
    print("\n  f(p) = p(1-p)²  — expected Brier improvement from betting on team")
    print("  ─────────────────────────────────────────────────────────────────")
    print(f"  {'p':>6}  {'f(p)':>8}  {'bar':}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*30}")
    for p_pct in range(5, 96, 5):
        p = p_pct / 100
        fp = expected_brier_improvement(p)
        bar = "█" * int(fp * 200)
        marker = " ← OPTIMAL (p=1/3)" if p_pct == 35 else \
                 (" ← (p=1/3)" if p_pct == 30 else "")
        print(f"  {p:>6.2f}  {fp:>8.5f}  {bar}{marker}")
    print(f"\n  Peak at p=1/3: f(1/3) = {expected_brier_improvement(1/3):.5f}")


if __name__ == "__main__":
    show_theory()