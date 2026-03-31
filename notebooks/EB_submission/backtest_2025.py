"""
backtest_2025.py
================
Train on everything before 2025, evaluate on actual 2025 tournament results.
Tells you what leaderboard rank your model would have achieved.

Usage:
    python backtest_2025.py
    python backtest_2025.py --data_dir /custom/path
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from solution import (
    DEFAULT_DATA_DIR, TARGET_SEASON,
    load_data, prepare_data, compact_to_matchups,
    build_season_avgs, build_elo, build_glm_quality,
    build_seeds, build_massey, load_kenpom, load_odds,
    assemble, get_features, train_loso, fit_spline,
    apply_spline, boost_predictions,
)

warnings.filterwarnings("ignore")
BACKTEST_SEASON = 2025


def run(data_dir=DEFAULT_DATA_DIR, output="backtest_2025_results.csv"):
    print("=" * 65)
    print(f"  Backtest: Train < {BACKTEST_SEASON}, Evaluate = {BACKTEST_SEASON}")
    print("=" * 65)

    print(f"\n[1/7] Loading data")
    dfs = load_data(data_dir)

    # Verify 2025 games exist
    m25 = dfs["m_tourn_compact"][dfs["m_tourn_compact"]["Season"] == BACKTEST_SEASON]
    w25 = dfs["w_tourn_compact"][dfs["w_tourn_compact"]["Season"] == BACKTEST_SEASON]
    print(f"  2025 men's tournament games:   {len(m25)}")
    print(f"  2025 women's tournament games: {len(w25)}")

    print(f"\n[2/7] Preparing data (Season < {BACKTEST_SEASON} only)")
    def pre(df): return df[df["Season"] < BACKTEST_SEASON].copy()

    m_reg = prepare_data(pre(dfs["m_reg_detailed"]))
    w_reg = prepare_data(pre(dfs["w_reg_detailed"]))
    regular_data = pd.concat([m_reg, w_reg], ignore_index=True)

    tc_m = pre(dfs["m_tourn_compact"])
    tc_m["men_women"] = 1
    tc_w = pre(dfs["w_tourn_compact"])
    tc_w["men_women"] = 0
    tourney_data = pd.concat([
        compact_to_matchups(tc_m, 1),
        compact_to_matchups(tc_w, 0),
    ], ignore_index=True)

    print(f"\n[3/7] Building features")
    seeds_all  = pd.concat([dfs["m_seeds"], dfs["w_seeds"]], ignore_index=True)
    seeds_T1, seeds_T2 = build_seeds(seeds_all)
    ss_T1, ss_T2 = build_season_avgs(regular_data)
    elo_T1, elo_T2 = build_elo(regular_data)
    print("  GLM quality...")
    glm_T1, glm_T2 = build_glm_quality(regular_data, pre(seeds_all))
    massey_T1, massey_T2 = build_massey(pre(dfs["m_massey"]))

    # Use 2025 odds for prediction (key fix vs previous version)
    odds_df = load_odds(dfs["m_spellings"], dfs["m_teams"], seasons=[BACKTEST_SEASON])

    # KenPom not available for 2025 in embedded data, so skip
    kenpom_df = None

    print(f"\n[4/7] Assembling training data")
    train_df = assemble(tourney_data, seeds_T1, seeds_T2,
                         ss_T1, ss_T2, elo_T1, elo_T2,
                         glm_T1, glm_T2, massey_T1, massey_T2,
                         kenpom_df, odds_df)
    features = get_features(train_df)
    print(f"  Rows: {len(train_df):,}  Features: {len(features)}")

    print(f"\n[5/7] Training LOSO models")
    models, oof_pred, oof_tgt = train_loso(train_df, features)
    spline = fit_spline(oof_pred, oof_tgt)

    # ── Build 2025 prediction features ────────────────────────────
    print(f"\n[6/7] Building 2025 prediction features")

    # Use 2025 regular season stats for prediction
    m_reg_25 = prepare_data(
        dfs["m_reg_detailed"][dfs["m_reg_detailed"]["Season"] == BACKTEST_SEASON])
    w_reg_25 = prepare_data(
        dfs["w_reg_detailed"][dfs["w_reg_detailed"]["Season"] == BACKTEST_SEASON])
    reg_25 = pd.concat([m_reg_25, w_reg_25], ignore_index=True)
    ss_T1_25, ss_T2_25 = build_season_avgs(reg_25)

    reg_full = pd.concat([regular_data, reg_25], ignore_index=True)
    elo_T1_25, elo_T2_25 = build_elo(reg_full)

    massey_full = build_massey(dfs["m_massey"][dfs["m_massey"]["Season"] <= BACKTEST_SEASON])
    massey_T1_full, massey_T2_full = massey_full

    # All teams in 2025
    teams_m25 = dfs["m_teams"]
    teams_w25 = dfs["w_teams"]
    all_ids_m = sorted(teams_m25["TeamID"].tolist())
    all_ids_w = sorted(teams_w25["TeamID"].tolist())

    print(f"  Generating all possible 2025 matchup pairs...")
    rows = []
    for ids, mw in [(all_ids_m, 1), (all_ids_w, 0)]:
        for i, t1 in enumerate(ids):
            for t2 in ids[i+1:]:
                rows.append({"Season": BACKTEST_SEASON,
                              "T1_TeamID": t1, "T2_TeamID": t2,
                              "PointDiff": 0, "win": 0,
                              "men_women": mw,
                              "ID": f"{BACKTEST_SEASON}_{t1}_{t2}"})
    all_matchups = pd.DataFrame(rows)
    print(f"  Total matchups: {len(all_matchups):,}")

    sub_feats = assemble(all_matchups,
                          seeds_T1, seeds_T2,
                          ss_T1_25, ss_T2_25,
                          elo_T1_25, elo_T2_25,
                          glm_T1, glm_T2,
                          massey_T1_full, massey_T2_full,
                          kenpom_df, odds_df)

    from solution import predict_loso
    raw_margins = predict_loso(models, sub_feats, features)
    probs       = apply_spline(spline, raw_margins)
    all_matchups["Pred"] = probs
    all_matchups = boost_predictions(all_matchups)

    # ── Score vs actual 2025 results ──────────────────────────────
    print(f"\n[7/7] Scoring against actual 2025 results")

    def day_to_round(day, gender):
        if gender == "M":
            if day in (134,135): return "Play-In"
            if day in (136,137): return "Round 1"
            if day in (138,139): return "Round 2"
            if day in (143,144): return "Sweet 16"
            if day in (145,146): return "Elite 8"
            if day == 152:       return "Final Four"
            if day == 154:       return "Championship"
        return f"Day {day}"

    outcomes = {}
    for _, g in m25.iterrows():
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        t1, t2 = (w,l) if w<l else (l,w)
        outcomes[(t1,t2)] = {"label": 1 if w==t1 else 0,
                              "gender":"M", "day": int(g["DayNum"])}
    for _, g in w25.iterrows():
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        t1, t2 = (w,l) if w<l else (l,w)
        outcomes[(t1,t2)] = {"label": 1 if w==t1 else 0,
                              "gender":"W", "day": int(g["DayNum"])}

    results = []
    for _, row in all_matchups.iterrows():
        key = (int(row["T1_TeamID"]), int(row["T2_TeamID"]))
        if key not in outcomes: continue
        info = outcomes[key]
        pred = float(row["Pred"])
        label = info["label"]
        results.append({
            "ID": row["ID"], "Team1": key[0], "Team2": key[1],
            "Pred": round(pred,4), "Actual": label,
            "Correct": int((pred>0.5)==bool(label)),
            "Brier": round((pred-label)**2, 6),
            "Round": day_to_round(info["day"], info["gender"]),
            "Gender": info["gender"],
        })
    results_df = pd.DataFrame(results)

    # Print summary
    all_names = pd.concat([
        dfs["m_teams"][["TeamID","TeamName"]],
        dfs["w_teams"][["TeamID","TeamName"]],
    ]).set_index("TeamID")["TeamName"].to_dict()

    ROUND_ORDER = ["Play-In","Round 1","Round 2","Sweet 16",
                   "Elite 8","Final Four","Championship"]

    print("\n" + "="*65)
    print(f"  BACKTEST RESULTS — {BACKTEST_SEASON} NCAA Tournament")
    print("="*65)

    for gender in ["M","W"]:
        gdf = results_df[results_df["Gender"]==gender]
        if not len(gdf): continue
        label = "Men's" if gender=="M" else "Women's"
        print(f"\n── {label} ({len(gdf)} games) ─────────────────────────────────")
        print(f"  Brier    : {gdf['Brier'].mean():.5f}  (prev: 0.13138)")
        print(f"  Accuracy : {gdf['Correct'].mean():.1%}  ({gdf['Correct'].sum()}/{len(gdf)})")
        print(f"\n  {'Round':<14} {'N':>4} {'Brier':>8} {'Acc':>8}")
        ri = {r:i for i,r in enumerate(ROUND_ORDER)}
        for rnd in ROUND_ORDER:
            rdf = gdf[gdf["Round"]==rnd]
            if not len(rdf): continue
            print(f"  {rnd:<14} {len(rdf):>4} {rdf['Brier'].mean():>8.5f} "
                  f"{rdf['Correct'].mean():>7.1%}")

    ob = results_df["Brier"].mean()
    oa = results_df["Correct"].mean()
    lb_approx = [(0.095,"~Top 10"),(0.105,"~Top 50"),(0.110,"~Top 100"),
                  (0.115,"~Top 200"),(0.120,"~Top 300"),(0.125,"~Top 400"),
                  (0.130,"~Top 600"),(0.135,"~Top 800"),(0.140,"~Top 1000")]
    est_rank = "Outside top 1000"
    for threshold, rank in lb_approx:
        if ob <= threshold: est_rank = rank; break

    print(f"\n── Overall ─────────────────────────────────────────────────")
    print(f"  Games        : {len(results_df)}")
    print(f"  Brier        : {ob:.5f}")
    print(f"  Accuracy     : {oa:.1%}")
    print(f"  vs prev sub  : {0.13138-ob:+.5f}")
    print(f"  Est. rank    : {est_rank}")

    # Show all games
    print(f"\n── Game-by-game ────────────────────────────────────────────")
    print(f"  {'Rnd':<14} {'G':<4} {'Team1':<20} {'Team2':<20} "
          f"{'Pred':>6} {'Act':>7} {'':>4}")
    disp = results_df.copy()
    disp["ri"] = disp["Round"].map(ri).fillna(99)
    disp = disp.sort_values(["Gender","ri","Team1"])
    for _, r in disp.iterrows():
        t1n = all_names.get(r["Team1"], str(r["Team1"]))
        t2n = all_names.get(r["Team2"], str(r["Team2"]))
        note = " ← BIG MISS" if not r["Correct"] and abs(r["Pred"]-0.5)>0.25 else ""
        print(f"  {r['Round']:<14} {'M' if r['Gender']=='M' else 'W':<4} "
              f"{t1n:<20} {t2n:<20} {r['Pred']:>6.3f} "
              f"{'T1' if r['Actual']==1 else 'T2':>7} "
              f"{'✓' if r['Correct'] else '✗':>4}{note}")

    results_df.to_csv(output, index=False)
    print(f"\n✅  Results saved → {output}")
    return results_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--output",   default="backtest_2025_results.csv")
    a = p.parse_args()
    run(a.data_dir, a.output)