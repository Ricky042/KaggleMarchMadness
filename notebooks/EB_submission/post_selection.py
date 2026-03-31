"""
post_selection.py
=================
Run AFTER Selection Sunday (March 15, 2026).

Identical to solution.py but:
  1. Verifies 2026 seeds are present
  2. Shows top predictions annotated with team names
  3. Applies optimal 1/3-probability strategy (optional)
  4. Blends with seed-matchup historical priors (optional)

Usage:
    python post_selection.py                     # recommended
    python post_selection.py --no_boost          # skip confidence boost
    python post_selection.py --no_blend          # skip seed prior blend
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from solution import (
    DEFAULT_DATA_DIR, TARGET_SEASON,
    run as solution_run,
    load_data, parse_seed_num,
)

try:
    from optimal_strategy import run_optimal_strategy
    _HAS_STRATEGY = True
except ImportError:
    _HAS_STRATEGY = False

warnings.filterwarnings("ignore")

# Historical seed win probabilities (1985-2025)
_SEED_WR = {
    (1,16):0.991,(2,15):0.938,(3,14):0.852,(4,13):0.790,
    (5,12):0.645,(6,11):0.612,(7,10):0.604,(8, 9):0.481,
    (1, 8):0.724,(1, 9):0.716,(1, 5):0.769,(2,10):0.571,
    (2, 7):0.597,(3,11):0.614,(3, 6):0.591,(4,12):0.645,
    (1, 4):0.755,(1, 2):0.575,(1, 3):0.669,
}

def seed_prior(s1, s2):
    if (s1,s2) in _SEED_WR: return _SEED_WR[(s1,s2)]
    if (s2,s1) in _SEED_WR: return 1.0 - _SEED_WR[(s2,s1)]
    return float(1.0 / (1.0 + np.exp(-0.22*(s2-s1))))


def run(data_dir=DEFAULT_DATA_DIR,
        output="final_submission.csv",
        blend_weight=0.80,
        apply_strategy=True,
        strategy_team=None,
        strategy_round=None,
        boost=True):

    print("=" * 65)
    print("  Post-Selection Sunday — Final Submission")
    print(f"  model_weight={blend_weight:.0%}  seed_prior={1-blend_weight:.0%}")
    print("=" * 65)

    # Check seeds exist
    dfs_check = load_data(data_dir)
    s26m = dfs_check["m_seeds"][dfs_check["m_seeds"]["Season"] == TARGET_SEASON]
    s26w = dfs_check["w_seeds"][dfs_check["w_seeds"]["Season"] == TARGET_SEASON]
    print(f"\n  2026 Men's seeds:   {len(s26m)} teams")
    print(f"  2026 Women's seeds: {len(s26w)} teams")

    if len(s26m) == 0 and len(s26w) == 0:
        print("\n⚠️  No 2026 seeds yet. Run solution.py instead.")
        print("   Re-download seed files from Kaggle after Selection Sunday.")
        return None

    # Run full pipeline
    sub, models, spline = solution_run(
        data_dir=data_dir,
        output="_tmp_pre_blend.csv",
        validate=False,
        boost=boost,
    )

    # Blend with seed priors for tournament matchups
    if blend_weight < 1.0 and len(s26m) > 0:
        print(f"\n[+] Blending model ({blend_weight:.0%}) with seed priors ({1-blend_weight:.0%})")
        seed_lu_m = (s26m.assign(sn=s26m["Seed"].apply(parse_seed_num))
                     .set_index("TeamID")["sn"].to_dict())
        seed_lu_w = (s26w.assign(sn=s26w["Seed"].apply(parse_seed_num))
                     .set_index("TeamID")["sn"].to_dict())
        blended = 0
        for idx, row in sub.iterrows():
            parts = row["ID"].split("_")
            season = int(parts[0])
            if season != TARGET_SEASON: continue
            t1, t2 = int(parts[1]), int(parts[2])
            is_men = t1 < 3000
            lu = seed_lu_m if is_men else seed_lu_w
            s1, s2 = lu.get(t1), lu.get(t2)
            if s1 and s2:
                prior = seed_prior(s1, s2)
                sub.at[idx, "Pred"] = (blend_weight * row["Pred"] +
                                        (1-blend_weight) * prior)
                blended += 1
        print(f"  Blended {blended:,} tournament matchups")

    # Optimal strategy
    if apply_strategy and _HAS_STRATEGY:
        print("\n[+] Applying optimal 1/3-probability strategy...")
        sub = run_optimal_strategy(
            sub, dfs_check["m_seeds"], dfs_check["w_seeds"],
            season=TARGET_SEASON,
            manual_team_id=strategy_team,
            manual_max_round=strategy_round,
        )
    elif apply_strategy:
        print("  [skip] optimal_strategy.py not found")

    sub["Pred"] = sub["Pred"].clip(0.01, 0.99).round(6)
    sub.to_csv(output, index=False)

    # Show top predictions with names
    all_names = pd.concat([
        dfs_check["m_teams"][["TeamID","TeamName"]],
        dfs_check["w_teams"][["TeamID","TeamName"]],
    ]).set_index("TeamID")["TeamName"].to_dict()

    sub26 = sub[sub["ID"].str.startswith(str(TARGET_SEASON))].copy()
    sub26["conf"] = (sub26["Pred"] - 0.5).abs()
    print(f"\n  Top 20 most confident 2026 predictions:")
    for _, r in sub26.nlargest(20,"conf").iterrows():
        parts = r["ID"].split("_")
        t1n = all_names.get(int(parts[1]), parts[1])
        t2n = all_names.get(int(parts[2]), parts[2])
        fav = t1n if r["Pred"]>0.5 else t2n
        p   = r["Pred"] if r["Pred"]>0.5 else 1-r["Pred"]
        print(f"    {t1n:<22} vs {t2n:<22} → {fav} ({p:.1%})")

    print(f"\n✅  Saved → {output}")
    print(f"    rows={len(sub):,}  "
          f"min={sub['Pred'].min():.4f}  max={sub['Pred'].max():.4f}")
    return sub


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",       default=DEFAULT_DATA_DIR)
    p.add_argument("--output",         default="final_submission.csv")
    p.add_argument("--blend_weight",   type=float, default=0.80,
                   help="Model weight in blend with seed prior")
    p.add_argument("--no_strategy",    action="store_true")
    p.add_argument("--no_blend",       action="store_true")
    p.add_argument("--no_boost",       action="store_true")
    p.add_argument("--strategy_team",  type=int, default=None)
    p.add_argument("--strategy_round", type=int, default=None)
    a = p.parse_args()

    run(data_dir=a.data_dir,
        output=a.output,
        blend_weight=1.0 if a.no_blend else a.blend_weight,
        apply_strategy=not a.no_strategy,
        strategy_team=a.strategy_team,
        strategy_round=a.strategy_round,
        boost=not a.no_boost)