"""
March Machine Learning Mania 2026 — Championship-Level Solution
================================================================
Built on the approach of the 2025 competition winner, with improvements.

Core architecture (from winning solution):
  1. Predict POINT DIFFERENTIAL (not win/loss) — far more signal
  2. GLM team quality — Bradley-Terry regression, beats seeds by AUC
  3. Leave-one-season-out ensemble — 21+ models averaged
  4. Spline calibration of margin → probability
  5. Prediction boosting post-processing

Our additions:
  + KenPom adjusted efficiency ratings (2026, embedded)
  + Multi-season betting odds → implied probabilities
  + Overtime-adjusted box score stats
  + Extended feature set (29 → 40+ features)
  + LightGBM as second base model, averaged with XGBoost

Usage:
    python solution.py
    python solution.py --no_validate
    python solution.py --output sub.csv
    python solution.py --data_dir /my/data
    python solution.py --no_boost
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import brier_score_loss
import statsmodels.api as sm
import tqdm
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

DEFAULT_DATA_DIR = "../../data/landing"
TARGET_SEASON    = 2026
SEED             = 42
SPLINE_CLIP      = 25

# ──────────────────────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS  (tuned from winning solution)
# ──────────────────────────────────────────────────────────────────────────────

XGB_PARAMS = {
    "objective":         "reg:squarederror",
    "booster":           "gbtree",
    "eta":               0.0093,
    "subsample":         0.6,
    "colsample_bynode":  0.8,
    "num_parallel_tree": 2,
    "min_child_weight":  4,
    "max_depth":         4,
    "tree_method":       "hist",
    "grow_policy":       "lossguide",
    "max_bin":           38,
    "seed":              SEED,
}
XGB_ROUNDS = 704

LGB_PARAMS = {
    "objective":        "regression",
    "metric":           "mae",
    "learning_rate":    0.0093,
    "subsample":        0.6,
    "colsample_bytree": 0.8,
    "min_child_samples":4,
    "max_depth":        4,
    "num_leaves":       15,
    "n_estimators":     704,
    "random_state":     SEED,
    "verbose":          -1,
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. KENPOM 2026  (embedded)
# ──────────────────────────────────────────────────────────────────────────────
# (rank, name, seed, net_rtg, off_rtg, def_rtg, adj_t, luck, sos_net, ncsos)

KENPOM_2026 = [
    ( 1,"Duke",1,+38.88,128.0,89.1,65.4,+.049,+14.29,+10.10),
    ( 2,"Arizona",1,+37.60,127.6,90.0,69.8,+.023,+14.93,+3.21),
    ( 3,"Michigan",1,+37.58,126.6,89.0,71.0,+.045,+16.66,+12.49),
    ( 4,"Florida",1,+33.75,125.5,91.8,70.6,-.036,+15.99,+7.93),
    ( 5,"Houston",2,+33.37,124.8,91.5,63.3,-.006,+13.55,+0.88),
    ( 6,"Iowa St.",2,+32.37,123.8,91.4,66.6,-.012,+12.40,-3.56),
    ( 7,"Illinois",3,+32.09,131.2,99.1,65.6,-.050,+13.65,+3.77),
    ( 8,"Purdue",2,+31.18,131.6,100.5,64.4,-.006,+15.88,+6.28),
    ( 9,"Michigan St.",3,+28.30,123.0,94.7,66.1,+.005,+13.70,+3.65),
    (10,"Gonzaga",3,+28.19,122.1,93.9,68.6,+.072,+6.01,+8.32),
    (11,"Connecticut",2,+27.84,122.0,94.2,64.5,+.055,+12.01,+7.63),
    (12,"Vanderbilt",5,+27.48,126.8,99.3,68.9,+.018,+14.55,+1.20),
    (13,"Virginia",3,+26.69,122.6,95.9,65.8,+.056,+9.95,-3.24),
    (14,"Nebraska",4,+26.15,118.5,92.4,66.8,+.034,+11.61,-6.01),
    (15,"Arkansas",4,+26.01,127.7,101.6,71.0,+.051,+14.93,+4.44),
    (16,"Tennessee",6,+25.98,121.0,95.1,65.0,-.060,+14.74,+0.77),
    (17,"St. John's",5,+25.87,120.0,94.2,69.6,+.061,+11.50,+6.57),
    (18,"Alabama",4,+25.68,129.0,103.3,73.1,+.019,+16.72,+13.43),
    (19,"Louisville",6,+25.40,124.0,98.7,69.7,-.020,+12.55,+2.41),
    (20,"Texas Tech",5,+25.19,125.0,99.9,66.2,+.006,+15.62,+7.79),
    (21,"Kansas",4,+24.40,118.3,93.9,67.7,+.053,+16.92,+11.01),
    (22,"Wisconsin",5,+23.39,125.3,102.0,68.8,+.041,+13.94,-1.08),
    (23,"BYU",6,+23.21,125.4,102.2,70.0,-.017,+14.25,+4.90),
    (24,"Saint Mary's",7,+23.15,120.4,97.2,65.2,+.011,+5.09,+3.87),
    (25,"Iowa",9,+22.44,121.7,99.3,63.0,-.061,+11.39,-7.17),
    (26,"Ohio St.",8,+22.23,124.3,102.1,66.1,-.031,+13.68,-1.81),
    (27,"UCLA",7,+21.66,123.7,102.1,64.7,+.017,+12.24,-2.16),
    (28,"Kentucky",7,+21.45,120.5,99.1,68.4,-.019,+15.91,+1.35),
    (29,"Utah St.",9,+20.84,122.2,101.4,67.7,+.065,+7.23,+2.01),
    (30,"North Carolina",6,+20.82,121.4,100.6,68.0,+.057,+11.46,+1.33),
    (31,"Miami FL",7,+20.67,121.4,100.7,67.6,+.021,+8.00,-7.27),
    (32,"Georgia",8,+20.45,124.7,104.3,71.4,-.005,+10.76,-6.43),
    (33,"Villanova",8,+19.94,120.4,100.4,65.2,+.067,+10.36,+2.88),
    (34,"N.C. State",11,+19.49,123.2,103.7,69.1,-.043,+12.25,+4.19),
    (35,"Santa Clara",10,+19.48,123.7,104.2,69.2,+.015,+6.13,+3.94),
    (36,"Clemson",8,+19.21,116.5,97.3,64.3,+.011,+10.53,+0.64),
    (37,"Texas",11,+19.13,124.2,105.0,66.9,-.068,+13.95,-6.36),
    (39,"Texas A&M",10,+18.63,119.7,101.1,70.5,-.002,+11.13,-5.87),
    (41,"Saint Louis",9,+18.27,119.5,101.2,71.0,+.030,+1.01,-5.78),
    (42,"SMU",11,+18.07,122.9,104.8,68.6,-.043,+11.15,+1.25),
    (43,"TCU",9,+17.53,115.3,97.8,67.7,+.004,+10.99,-6.94),
    (46,"VCU",11,+17.13,119.9,102.7,68.5,-.007,+3.44,-1.26),
    (48,"South Florida",11,+16.37,117.3,101.0,71.5,-.026,+3.06,+4.95),
    (52,"Missouri",10,+15.36,119.5,104.1,66.2,+.041,+11.50,-7.75),
    (54,"UCF",10,+14.99,120.4,105.4,69.2,+.097,+11.88,-2.53),
    (64,"Akron",12,+12.79,118.8,106.1,70.3,+.018,-3.64,-3.13),
    (66,"McNeese",12,+12.54,114.3,101.8,66.2,+.084,-1.78,+9.16),
    (71,"Northern Iowa",12,+11.80,110.0,98.2,62.3,-.070,+1.28,-0.91),
    (87,"Hofstra",13,+9.53,114.6,105.1,64.7,-.052,-0.86,+3.09),
    (92,"High Point",12,+8.40,117.0,108.6,69.9,+.048,-9.21,-8.86),
    (93,"Miami OH",11,+8.27,116.8,108.5,70.0,+.099,-5.34,-9.88),
    (107,"Cal Baptist",13,+5.99,107.9,101.9,65.8,+.091,-1.94,-3.31),
    (109,"Hawaii",13,+5.90,107.1,101.2,69.7,+.038,-3.42,-11.21),
    (115,"North Dakota St.",14,+5.00,111.7,106.7,66.3,+.040,-5.92,-0.94),
    (140,"Wright St.",14,+2.04,112.1,110.0,67.2,+.009,-4.02,+0.88),
    (143,"Troy",13,+1.80,110.8,109.0,64.9,+.024,-3.14,+4.19),
    (148,"Idaho",15,+1.51,108.8,107.3,67.8,-.012,-1.67,+0.75),
    (151,"Penn",14,+1.33,107.3,106.0,69.1,+.068,-0.80,+0.45),
    (162,"Kennesaw St.",14,+0.76,110.7,110.0,71.2,+.009,-2.02,-5.16),
    (183,"Queens",15,-1.44,115.8,117.2,69.6,+.067,-5.65,+5.68),
    (186,"Tennessee St.",15,-1.81,109.1,110.9,70.2,+.070,-8.23,+5.51),
    (188,"UMBC",16,-1.94,108.8,110.7,66.3,+.028,-14.45,-6.98),
    (190,"Furman",15,-1.97,107.5,109.4,65.9,+.010,-6.27,-0.68),
    (193,"Siena",16,-2.14,107.1,109.3,64.7,+.005,-9.50,-7.77),
    (205,"Howard",16,-2.92,104.1,107.0,69.1,+.003,-14.03,-1.67),
    (216,"LIU",16,-3.96,105.6,109.6,67.8,+.104,-9.96,+2.18),
    (284,"Lehigh",16,-10.41,102.7,113.1,66.9,+.081,-8.63,-0.54),
    (288,"Prairie View",16,-10.69,101.2,111.9,71.0,+.013,-9.56,+8.93),
]

KENPOM_ALIASES = {
    "iowa st.":          "iowa st",
    "michigan st.":      "michigan st",
    "ohio st.":          "ohio st",
    "utah st.":          "utah st",
    "n.c. state":        "nc state",
    "st. john's":        "st john's",
    "saint mary's":      "saint mary's",
    "saint louis":       "saint louis",
    "north carolina":    "north carolina",
    "miami fl":          "miami fl",
    "north dakota st.":  "north dakota st",
    "wright st.":        "wright st",
    "kennesaw st.":      "kennesaw st",
    "tennessee st.":     "tennessee st",
    "cal baptist":       "cal baptist",
    "mcneese":           "mcneese st",
    "penn":              "pennsylvania",
    "queens":            "queens nc",
    "liu":               "liu",
    "prairie view":      "prairie view",
    "south florida":     "south florida",
    "texas a&m":         "texas a&m",
    "vcu":               "vcu",
    "smu":               "smu",
    "tcu":               "tcu",
    "ucf":               "ucf",
    "byu":               "byu",
    "akron":             "akron",
}


def load_kenpom(spellings_df, teams_df, season=TARGET_SEASON):
    """Build KenPom DataFrame with TeamIDs attached."""
    lookup = {}
    for _, r in spellings_df.iterrows():
        lookup[r["TeamNameSpelling"].lower().strip()] = int(r["TeamID"])
    for _, r in teams_df.iterrows():
        lookup.setdefault(r["TeamName"].lower().strip(), int(r["TeamID"]))

    rows = []
    for rank, name, seed, net, off, defe, tempo, luck, sos, ncsos in KENPOM_2026:
        raw = name.lower().strip()
        tid = lookup.get(raw) or lookup.get(KENPOM_ALIASES.get(raw, "___"))
        if tid is None:
            for k, v in lookup.items():
                if raw in k or k in raw:
                    tid = v
                    break
        if tid:
            rows.append({
                "Season": season, "TeamID": tid,
                "kp_rank":  rank,  "kp_seed":  seed,
                "kp_net":   net,   "kp_off":   off,
                "kp_def":   defe,  "kp_tempo": tempo,
                "kp_luck":  luck,  "kp_sos":   sos,
                "kp_ncsos": ncsos,
            })
    print(f"  [KenPom] Matched {len(rows)}/{len(KENPOM_2026)} teams")
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 2. BETTING ODDS  (multi-season embedded)
# ──────────────────────────────────────────────────────────────────────────────
# (season, name, to_win, f4, e8, s16)

RAW_ODDS = [
    (2025,"Duke Blue Devils","+275","-180","-320","-700"),
    (2025,"Auburn Tigers","+350","-140","-280","-650"),
    (2025,"Florida Gators","+500","+110","-200","-550"),
    (2025,"Houston Cougars","+550","+130","-175","-500"),
    (2025,"Tennessee Volunteers","14-1","+260","+115","-300"),
    (2025,"Michigan State Spartans","18-1","+310","+130","-275"),
    (2025,"Alabama Crimson Tide","18-1","+320","+140","-250"),
    (2025,"Iowa State Cyclones","20-1","+350","+150","-225"),
    (2025,"Wisconsin Badgers","28-1","+400","+160","-200"),
    (2025,"Texas Tech Red Raiders","30-1","+450","+180","-175"),
    (2025,"Gonzaga Bulldogs","200-1","+250","10-1","+400"),
    (2025,"Arkansas Razorbacks","250-1","+300","11-1","+450"),
    (2026,"Duke Blue Devils","+350","-165","-255","-600"),
    (2026,"Michigan Wolverines","+370","-130","-380","-900"),
    (2026,"Arizona Wildcats","+380","-120","-275","-700"),
    (2026,"Florida Gators","+750","+165","-160","-450"),
    (2026,"Houston Cougars","13-1","+250","+110","-280"),
    (2026,"Iowa State Cyclones","18-1","+245","-130","-400"),
    (2026,"Illinois Fighting Illini","22-1","+310","+120","-400"),
    (2026,"Purdue Boilermakers","25-1","+360","+105","-330"),
    (2026,"UConn Huskies","30-1","+550","+170","-200"),
    (2026,"Michigan State Spartans","55-1","+650","+200","-130"),
    (2026,"Gonzaga Bulldogs","60-1","+500","+145","-250"),
    (2026,"Arkansas Razorbacks","60-1","11-1","+450","-120"),
    (2026,"Kansas Jayhawks","60-1","13-1","+650","+125"),
    (2026,"Virginia Cavaliers","75-1","11-1","+310","-125"),
    (2026,"Vanderbilt Commodores","75-1","11-1","+350","-110"),
    (2026,"Alabama Crimson Tide","180-1","22-1","+750","-120"),
    (2026,"Tennessee Volunteers","130-1","13-1","+400","+130"),
    (2026,"St. John's Red Storm","75-1","+900","+475","-105"),
    (2026,"Wisconsin Badgers","100-1","15-1","+550","+120"),
    (2026,"Texas Tech Red Raiders","130-1","17-1","+600","+105"),
    (2026,"Nebraska Cornhuskers","110-1","11-1","+400","-105"),
    (2026,"Louisville Cardinals","150-1","12-1","+360","+150"),
    (2026,"North Carolina Tar Heels","250-1","60-1","20-1","+600"),
    (2026,"Saint Mary's Gaels","300-1","35-1","11-1","+360"),
    (2026,"Kentucky Wildcats","250-1","45-1","13-1","+550"),
    (2026,"UCLA Bruins","180-1","19-1","+550","+200"),
    (2026,"BYU Cougars","350-1","60-1","14-1","+425"),
]

ODDS_ALIASES = {
    "duke blue devils":          "duke",
    "michigan wolverines":       "michigan",
    "arizona wildcats":          "arizona",
    "florida gators":            "florida",
    "houston cougars":           "houston",
    "iowa state cyclones":       "iowa st",
    "illinois fighting illini":  "illinois",
    "purdue boilermakers":       "purdue",
    "uconn huskies":             "connecticut",
    "michigan state spartans":   "michigan st",
    "gonzaga bulldogs":          "gonzaga",
    "arkansas razorbacks":       "arkansas",
    "kansas jayhawks":           "kansas",
    "st. john's red storm":      "st john's",
    "virginia cavaliers":        "virginia",
    "vanderbilt commodores":     "vanderbilt",
    "wisconsin badgers":         "wisconsin",
    "nebraska cornhuskers":      "nebraska",
    "texas tech red raiders":    "texas tech",
    "tennessee volunteers":      "tennessee",
    "louisville cardinals":      "louisville",
    "ucla bruins":               "ucla",
    "alabama crimson tide":      "alabama",
    "kentucky wildcats":         "kentucky",
    "north carolina tar heels":  "north carolina",
    "saint mary's gaels":        "saint mary's",
    "auburn tigers":             "auburn",
    "byu cougars":               "byu",
}


def _parse_odds(s):
    if not s or str(s).upper() in ("TBD", ""):
        return np.nan
    s = str(s).strip().replace(",", "")
    if s.startswith("+"):
        v = float(s[1:])
        return 100.0 / (v + 100.0)
    if s.startswith("-") and s.count("-") == 1:
        v = float(s[1:])
        return v / (v + 100.0)
    if "-" in s:
        parts = s.split("-")
        try:
            n, d = float(parts[0]), float(parts[1])
            return d / (n + d)
        except Exception:
            return np.nan
    return np.nan


def load_odds(spellings_df, teams_df, seasons):
    """Build odds DataFrame with TeamIDs, filtered to requested seasons."""
    lookup = {}
    for _, r in spellings_df.iterrows():
        lookup[r["TeamNameSpelling"].lower().strip()] = int(r["TeamID"])
    for _, r in teams_df.iterrows():
        lookup.setdefault(r["TeamName"].lower().strip(), int(r["TeamID"]))

    rows = []
    for season, name, to_win, f4, e8, s16 in RAW_ODDS:
        if season not in seasons:
            continue
        raw = name.lower().strip()
        tid = lookup.get(raw) or lookup.get(ODDS_ALIASES.get(raw, "___"))
        if tid is None:
            for k, v in lookup.items():
                if raw in k or k in raw:
                    tid = v
                    break
        if tid:
            rows.append({
                "Season":        season,
                "TeamID":        tid,
                "odds_win":      _parse_odds(to_win),
                "odds_f4":       _parse_odds(f4),
                "odds_e8":       _parse_odds(e8),
                "odds_s16":      _parse_odds(s16),
            })

    df = pd.DataFrame(rows)
    if len(df):
        df["odds_composite"] = (
            0.40 * df["odds_win"].fillna(0) +
            0.30 * df["odds_f4"].fillna(0) +
            0.20 * df["odds_e8"].fillna(0) +
            0.10 * df["odds_s16"].fillna(0)
        )
    print(f"  [Odds] Loaded {len(df)} team-season rows "
          f"(seasons {sorted(set(seasons))})")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_data(data_dir):
    d = Path(data_dir)
    if not d.exists():
        raise FileNotFoundError(
            f"Data directory not found: {d.resolve()}\n"
            f"Set --data_dir to the folder containing the Kaggle CSV files.")
    files = {
        "m_teams":         "MTeams.csv",
        "m_seeds":         "MNCAATourneySeeds.csv",
        "m_reg_detailed":  "MRegularSeasonDetailedResults.csv",
        "m_tourn_compact": "MNCAATourneyCompactResults.csv",
        "m_reg_compact":   "MRegularSeasonCompactResults.csv",
        "m_massey":        "MMasseyOrdinals.csv",
        "m_spellings":     "MTeamSpellings.csv",
        "w_teams":         "WTeams.csv",
        "w_seeds":         "WNCAATourneySeeds.csv",
        "w_reg_detailed":  "WRegularSeasonDetailedResults.csv",
        "w_tourn_compact": "WNCAATourneyCompactResults.csv",
        "w_reg_compact":   "WRegularSeasonCompactResults.csv",
        "w_spellings":     "WTeamSpellings.csv",
        "sample_sub":      "SampleSubmissionStage2.csv",
    }
    dfs = {}
    for key, fname in files.items():
        fp = d / fname
        if fp.exists():
            dfs[key] = pd.read_csv(fp)
            print(f"  ✓ {fname:52s} {str(dfs[key].shape):>14}")
        else:
            print(f"  ✗ MISSING: {fname}")
    return dfs


# ──────────────────────────────────────────────────────────────────────────────
# 4. DATA PREPARATION  (OT adjustment + doubling)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data(df):
    """
    From winning solution:
      - Adjust all stats by overtime factor (normalize to 40-min game)
      - Double dataset by swapping T1/T2 so model sees both perspectives
    """
    cols_needed = [
        "Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT",
        "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA",
        "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
        "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA",
        "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
    ]
    df = df[[c for c in cols_needed if c in df.columns]].copy()

    # Overtime normalisation — only divide actual stats, never IDs
    adjot     = (40 + 5 * df["NumOT"]) / 40
    skip_cols = {"Season", "DayNum", "NumOT", "WTeamID", "LTeamID"}
    stat_cols = [c for c in df.columns if c not in skip_cols]
    for col in stat_cols:
        df[col] = df[col] / adjot

    # Create T1 (winner) and T2 (loser) versions, then stack
    df_w = df.copy()
    df_l = df.copy()
    df_w.columns = [c.replace("W", "T1_").replace("L", "T2_")
                    for c in df_w.columns]
    df_l.columns = [c.replace("L", "T1_").replace("W", "T2_")
                    for c in df_l.columns]

    out = pd.concat([df_w, df_l], ignore_index=True)
    out["PointDiff"] = out["T1_Score"] - out["T2_Score"]
    out["win"]       = (out["PointDiff"] > 0).astype(int)
    out["men_women"] = out["T1_TeamID"].astype(str).str.startswith("1").astype(int)
    return out


def compact_to_matchups(df, men_women):
    """
    Convert compact results (WTeamID/LTeamID) into doubled matchup rows
    with OT-adjusted PointDiff. Used for tournament data where we don't
    need full box scores.
    """
    rows = []
    for _, r in df.iterrows():
        adjot  = (40 + 5 * int(r["NumOT"])) / 40
        pd_val = (r["WScore"] - r["LScore"]) / adjot
        rows.append({
            "Season":     r["Season"],
            "T1_TeamID":  r["WTeamID"],
            "T2_TeamID":  r["LTeamID"],
            "PointDiff":  pd_val,
            "win":        1,
            "men_women":  men_women,
        })
        rows.append({
            "Season":     r["Season"],
            "T1_TeamID":  r["LTeamID"],
            "T2_TeamID":  r["WTeamID"],
            "PointDiff":  -pd_val,
            "win":        0,
            "men_women":  men_women,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 5. SEASON AVERAGES  (medium features)
# ──────────────────────────────────────────────────────────────────────────────

BOX_COLS = [
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3",
    "T1_FTM",   "T1_FTA", "T1_OR",  "T1_DR",   "T1_Ast",
    "T1_TO",    "T1_Stl", "T1_Blk", "T1_PF",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3",
    "T2_FTM",   "T2_FTA", "T2_OR",  "T2_DR",   "T2_Ast",
    "T2_TO",    "T2_Stl", "T2_Blk", "T2_PF",
    "PointDiff",
]


def build_season_avgs(regular_data):
    """
    Per-team per-season average box score stats.
    Returns two DataFrames: ss_T1 and ss_T2 for merging.
    """
    avail  = [c for c in BOX_COLS if c in regular_data.columns]
    ss     = (regular_data
              .groupby(["Season", "T1_TeamID"])[avail]
              .mean()
              .reset_index())

    # Build T1 version
    ss_T1 = ss.copy()
    new_cols_T1 = ["Season", "T1_TeamID"]
    for c in avail:
        new_cols_T1.append(
            "T1_avg_" + c.replace("T1_", "").replace("T2_", "opp_"))
    ss_T1.columns = new_cols_T1

    # Build T2 version
    ss_T2 = ss.copy()
    new_cols_T2 = ["Season", "T2_TeamID"]
    for c in avail:
        new_cols_T2.append(
            "T2_avg_" + c.replace("T1_", "").replace("T2_", "opp_"))
    ss_T2.columns = new_cols_T2

    return ss_T1, ss_T2


# ──────────────────────────────────────────────────────────────────────────────
# 6. ELO RATINGS
# ──────────────────────────────────────────────────────────────────────────────

def build_elo(regular_data, base=1000, width=400, k=100):
    """
    Per-season Elo (from winning solution).
    Resets each season — simpler but effective.
    """
    elos  = []
    wins  = regular_data[regular_data["win"] == 1].copy()

    for season, sg in wins.groupby("Season"):
        teams = set(sg["T1_TeamID"]) | set(sg["T2_TeamID"])
        elo   = {t: base for t in teams}

        for _, row in sg.iterrows():
            w, l   = row["T1_TeamID"], row["T2_TeamID"]
            ew, el = elo[w], elo[l]
            exp_w  = 1.0 / (1.0 + 10 ** ((el - ew) / width))
            delta  = k * (1 - exp_w)
            elo[w] = ew + delta
            elo[l] = el - delta

        for tid, ev in elo.items():
            elos.append({"Season": season, "TeamID": tid, "elo": ev})

    df = pd.DataFrame(elos)
    T1 = df.rename(columns={"TeamID": "T1_TeamID", "elo": "T1_elo"})
    T2 = df.rename(columns={"TeamID": "T2_TeamID", "elo": "T2_elo"})
    return T1, T2


# ──────────────────────────────────────────────────────────────────────────────
# 7. GLM TEAM QUALITY  (most powerful feature — Bradley-Terry regression)
# ──────────────────────────────────────────────────────────────────────────────

def build_glm_quality(regular_data, seeds_df):
    """
    For each season, fit a GLM:
        PointDiff ~ -1 + T1_TeamID + T2_TeamID
    The T1_TeamID coefficients become team quality scores.

    Only uses tourney teams + teams that beat a tourney team, which
    keeps coefficients meaningful (from winning solution).
    """
    regular_data = regular_data.copy()
    seeds_df     = seeds_df.copy()

    # Build season/team keys — cast to int first to avoid float mismatches
    regular_data["ST1"] = (regular_data["Season"].astype(int).astype(str) + "/" +
                            regular_data["T1_TeamID"].astype(int).astype(str))
    regular_data["ST2"] = (regular_data["Season"].astype(int).astype(str) +  "/" +
                            regular_data["T2_TeamID"].astype(int).astype(str))
    st = set(seeds_df["Season"].astype(int).astype(str) + "/" +
             seeds_df["TeamID"].astype(int).astype(str))

    # Add teams that beat a tourney team at least once
    beaters = set(regular_data.loc[
        (regular_data["PointDiff"] > 0) &
        (regular_data["ST2"].isin(st)), "ST1"])
    st = st | beaters

    dt = regular_data.loc[
        regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)
    ].copy()

    dt["T1_TeamID"] = dt["T1_TeamID"].astype(str)
    dt["T2_TeamID"] = dt["T2_TeamID"].astype(str)
    dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
    dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"

    glm_results = []
    seasons = sorted(dt["Season"].unique())
    print(f"  Computing GLM quality for {len(seasons)} seasons (~2-3 min)...")

    for season in tqdm.tqdm(seasons, unit="season"):
        for mw in [0, 1]:
            subset = dt.loc[(dt["Season"] == season) &
                             (dt["men_women"] == mw)]
            if len(subset) < 10:
                continue
            try:
                glm = sm.GLM.from_formula(
                    "PointDiff ~ -1 + T1_TeamID + T2_TeamID",
                    data=subset,
                    family=sm.families.Gaussian(),
                ).fit(disp=False)

                # glm.params is a Series — index=param_name, values=coefficient
                # statsmodels names categorical levels like "T1_TeamID[T.1181]"
                q = glm.params[
                    glm.params.index.str.startswith("T1_TeamID")
                ].reset_index()
                q.columns = ["param", "quality"]

                # Extract numeric team ID from "T1_TeamID[T.1181]" → 1181
                q["TeamID"] = (q["param"]
                               .str.extract(r"T1_TeamID\[T\.(\d+)\]", expand=False)
                               .fillna(
                                   q["param"].str.extract(r"(\d+)", expand=False)
                               )
                               .astype(int))
                q["Season"]    = season
                q["men_women"] = mw
                glm_results.append(q[["Season", "TeamID", "quality", "men_women"]])
            except Exception as e:
                # Print so we can see any remaining issues
                tqdm.tqdm.write(f"    GLM failed season={season} mw={mw}: {e}")

    if not glm_results:
        raise RuntimeError(
            "GLM quality: no seasons produced results. "
            "Check the error messages printed above.")

    glm_df = pd.concat(glm_results).reset_index(drop=True)
    T1 = (glm_df[["Season", "TeamID", "quality"]]
          .rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"}))
    T2 = (glm_df[["Season", "TeamID", "quality"]]
          .rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"}))
    return T1, T2


# ──────────────────────────────────────────────────────────────────────────────
# 8. MASSEY ORDINALS
# ──────────────────────────────────────────────────────────────────────────────

GOOD_MASSEY = ["POM", "SAG", "MOR", "NET", "RPI", "BIH", "DOL", "KPI", "MAS"]


def build_massey(massey_df):
    """Pre-tournament Massey rankings → per-team per-season features."""
    pre = massey_df[massey_df["RankingDayNum"] == 133].copy()

    pivot = pre.pivot_table(
        index=["Season", "TeamID"],
        columns="SystemName",
        values="OrdinalRank",
        aggfunc="mean",
    )
    pivot.columns.name = None

    pivot["massey_mean"] = pivot.mean(axis=1)
    pivot["massey_min"]  = pivot.min(axis=1)

    systems = [s for s in GOOD_MASSEY if s in pivot.columns]
    keep    = systems + ["massey_mean", "massey_min"]
    result  = pivot[keep].reset_index()

    for s in systems:
        result.rename(columns={s: f"massey_{s}"}, inplace=True)

    T1 = result.rename(columns={"TeamID": "T1_TeamID"})
    T1.columns = (["Season", "T1_TeamID"] +
                   [f"T1_{c}" for c in T1.columns
                    if c not in ("Season", "T1_TeamID")])

    T2 = result.rename(columns={"TeamID": "T2_TeamID"})
    T2.columns = (["Season", "T2_TeamID"] +
                   [f"T2_{c}" for c in T2.columns
                    if c not in ("Season", "T2_TeamID")])

    return T1, T2


# ──────────────────────────────────────────────────────────────────────────────
# 9. SEEDS
# ──────────────────────────────────────────────────────────────────────────────

def parse_seed_num(s):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else 16


def build_seeds(seeds_df):
    df = seeds_df.copy()
    df["seed"] = df["Seed"].apply(parse_seed_num)
    T1 = df[["Season", "TeamID", "seed"]].rename(
        columns={"TeamID": "T1_TeamID", "seed": "T1_seed"})
    T2 = df[["Season", "TeamID", "seed"]].rename(
        columns={"TeamID": "T2_TeamID", "seed": "T2_seed"})
    return T1, T2


# ──────────────────────────────────────────────────────────────────────────────
# 10. ASSEMBLE FEATURES ONTO MATCHUP ROWS
# ──────────────────────────────────────────────────────────────────────────────

def assemble(matchup_df, seeds_T1, seeds_T2,
             ss_T1, ss_T2, elo_T1, elo_T2,
             glm_T1, glm_T2, massey_T1, massey_T2,
             kenpom_df=None, odds_df=None):
    """Merge all feature tables onto a matchup DataFrame."""
    df = matchup_df.copy()

    df = df.merge(seeds_T1, on=["Season", "T1_TeamID"], how="left")
    df = df.merge(seeds_T2, on=["Season", "T2_TeamID"], how="left")
    df["Seed_diff"] = df["T2_seed"] - df["T1_seed"]

    df = df.merge(ss_T1, on=["Season", "T1_TeamID"], how="left")
    df = df.merge(ss_T2, on=["Season", "T2_TeamID"], how="left")

    df = df.merge(elo_T1, on=["Season", "T1_TeamID"], how="left")
    df = df.merge(elo_T2, on=["Season", "T2_TeamID"], how="left")
    df["elo_diff"] = df["T1_elo"] - df["T2_elo"]

    df = df.merge(glm_T1, on=["Season", "T1_TeamID"], how="left")
    df = df.merge(glm_T2, on=["Season", "T2_TeamID"], how="left")
    df["diff_quality"] = df["T1_quality"] - df["T2_quality"]

    df = df.merge(massey_T1, on=["Season", "T1_TeamID"], how="left")
    df = df.merge(massey_T2, on=["Season", "T2_TeamID"], how="left")

    # KenPom — only 2026 has embedded data; NaN fills for all other seasons
    if kenpom_df is not None and len(kenpom_df):
        kp1 = (kenpom_df
               .rename(columns={c: f"T1_{c}" for c in kenpom_df.columns
                                 if c not in ("Season", "TeamID")})
               .rename(columns={"TeamID": "T1_TeamID"}))
        kp2 = (kenpom_df
               .rename(columns={c: f"T2_{c}" for c in kenpom_df.columns
                                 if c not in ("Season", "TeamID")})
               .rename(columns={"TeamID": "T2_TeamID"}))
        df = df.merge(kp1, on=["Season", "T1_TeamID"], how="left")
        df = df.merge(kp2, on=["Season", "T2_TeamID"], how="left")
        df["kp_net_diff"]  = df["T1_kp_net"]  - df["T2_kp_net"]
        df["kp_off_diff"]  = df["T1_kp_off"]  - df["T2_kp_off"]
        df["kp_def_diff"]  = df["T1_kp_def"]  - df["T2_kp_def"]
        df["kp_rank_diff"] = df["T2_kp_rank"] - df["T1_kp_rank"]

    # Betting odds
    if odds_df is not None and len(odds_df):
        od1 = (odds_df
               .rename(columns={c: f"T1_{c}" for c in odds_df.columns
                                 if c not in ("Season", "TeamID")})
               .rename(columns={"TeamID": "T1_TeamID"}))
        od2 = (odds_df
               .rename(columns={c: f"T2_{c}" for c in odds_df.columns
                                 if c not in ("Season", "TeamID")})
               .rename(columns={"TeamID": "T2_TeamID"}))
        df = df.merge(od1, on=["Season", "T1_TeamID"], how="left")
        df = df.merge(od2, on=["Season", "T2_TeamID"], how="left")
        df["odds_composite_diff"] = (df["T1_odds_composite"] -
                                      df["T2_odds_composite"])

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 11. FEATURE LISTS
# ──────────────────────────────────────────────────────────────────────────────

BASE_FEATURES = [
    # Seed (easy)
    "men_women",
    "T1_seed", "T2_seed", "Seed_diff",
    # Box score averages (medium — curated subset from winning solution)
    "T1_avg_Score",    "T1_avg_FGA",  "T1_avg_OR", "T1_avg_DR",
    "T1_avg_Blk",      "T1_avg_PF",   "T1_avg_opp_FGA",
    "T1_avg_opp_Blk",  "T1_avg_opp_PF", "T1_avg_PointDiff",
    "T2_avg_Score",    "T2_avg_FGA",  "T2_avg_OR", "T2_avg_DR",
    "T2_avg_Blk",      "T2_avg_PF",   "T2_avg_opp_FGA",
    "T2_avg_opp_Blk",  "T2_avg_opp_PF", "T2_avg_PointDiff",
    # Elo (hard)
    "T1_elo", "T2_elo", "elo_diff",
    # GLM quality (hardest — most powerful)
    "T1_quality", "T2_quality", "diff_quality",
]

EXTRA_FEATURES = [
    # Massey ordinals
    "T1_massey_mean", "T2_massey_mean",
    "T1_massey_min",  "T2_massey_min",
    # KenPom (2026 only — NaN for training seasons, used at prediction time)
    "T1_kp_net",  "T2_kp_net",  "kp_net_diff",
    "T1_kp_off",  "T2_kp_off",  "kp_off_diff",
    "T1_kp_def",  "T2_kp_def",  "kp_def_diff",
    "T1_kp_rank", "T2_kp_rank", "kp_rank_diff",
    "T1_kp_luck", "T2_kp_luck",
    "T1_kp_sos",  "T2_kp_sos",
    # Odds
    "T1_odds_composite", "T2_odds_composite", "odds_composite_diff",
]

ALL_FEATURES = BASE_FEATURES + EXTRA_FEATURES


def get_features(df):
    """Return only features that actually exist as columns."""
    return [f for f in ALL_FEATURES if f in df.columns]


# ──────────────────────────────────────────────────────────────────────────────
# 12. SPLINE CALIBRATION  (predicted margin → win probability)
# ──────────────────────────────────────────────────────────────────────────────

def fit_spline(oof_preds, oof_targets, clip=SPLINE_CLIP):
    """
    Fit a degree-5 spline mapping predicted point differential to
    empirical win probability. Identical to winning solution's approach.
    """
    pairs = sorted(zip(oof_preds,
                        (np.array(oof_targets) > 0).astype(int)),
                   key=lambda x: x[0])
    pred_s, label_s = zip(*pairs)
    spline = UnivariateSpline(
        np.clip(pred_s, -clip, clip),
        label_s, k=5,
    )
    return spline


def apply_spline(spline, preds, clip=SPLINE_CLIP):
    return np.clip(spline(np.clip(preds, -clip, clip)), 0.01, 0.99)


# ──────────────────────────────────────────────────────────────────────────────
# 13. LEAVE-ONE-SEASON-OUT TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train_loso(tourney_df, features):
    """
    Leave-one-season-out: one XGBoost + one LightGBM per held-out season.
    Predictions averaged. Returns models dict + OOF preds for spline fitting.
    """
    seasons  = sorted(tourney_df["Season"].unique())
    models   = {}
    oof_pred = np.zeros(len(tourney_df))
    oof_tgt  = tourney_df["PointDiff"].values

    print(f"\n  Training {len(seasons)} LOSO models (XGB + LGB each)...")

    for oof_season in seasons:
        tr_mask = tourney_df["Season"] != oof_season
        vl_mask = tourney_df["Season"] == oof_season

        X_tr = tourney_df.loc[tr_mask, features].fillna(-999).values.astype(float)
        y_tr = tourney_df.loc[tr_mask, "PointDiff"].values
        X_vl = tourney_df.loc[vl_mask, features].fillna(-999).values.astype(float)
        y_vl = tourney_df.loc[vl_mask, "PointDiff"].values

        # XGBoost
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval   = xgb.DMatrix(X_vl)
        m_xgb  = xgb.train(XGB_PARAMS, dtrain, XGB_ROUNDS,
                             verbose_eval=False)
        p_xgb  = m_xgb.predict(dval)

        # LightGBM
        m_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
        m_lgb.fit(X_tr, y_tr)
        p_lgb = m_lgb.predict(X_vl)

        p_avg = (p_xgb + p_lgb) / 2
        oof_pred[vl_mask] = p_avg
        models[oof_season] = (m_xgb, m_lgb)

        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_vl, p_avg)
        bs  = brier_score_loss((y_vl > 0).astype(int),
                                apply_spline(fit_spline(p_avg, y_vl), p_avg)
                                if len(set(y_vl > 0)) > 1 else [0.5] * len(y_vl))
        print(f"    {oof_season}:  MAE={mae:.3f}  N={int(vl_mask.sum())}")

    return models, oof_pred, oof_tgt


def predict_loso(models, sub_feats, features):
    """Average predictions from all LOSO models."""
    X     = sub_feats[features].fillna(-999).values.astype(float)
    dtest = xgb.DMatrix(X)
    preds = []
    for season, (m_xgb, m_lgb) in models.items():
        p_xgb = m_xgb.predict(dtest)
        p_lgb = m_lgb.predict(X)
        preds.append((p_xgb + p_lgb) / 2)
    return np.array(preds).mean(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# 14. POST-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def boost_predictions(sub_df, boost_pct=0.10, threshold=0.85):
    """
    From winning solution: increase predictions below threshold by boost_pct.
    Sharpens confident predictions. +10% for preds < 85%.
    """
    mask = sub_df["Pred"] < threshold
    sub_df = sub_df.copy()
    sub_df.loc[mask, "Pred"] = sub_df.loc[mask, "Pred"] * (1 + boost_pct)
    print(f"  [boost] {mask.sum():,} predictions boosted by {boost_pct:.0%}")
    return sub_df


def apply_manual_overrides(sub_df, overrides):
    """
    Manually override specific game predictions.
    Use after Selection Sunday when you have strong views on specific matchups.
    Format: {"2026_XXXX_YYYY": 0.95, ...}
    """
    sub_df = sub_df.copy()
    for match_id, pred in overrides.items():
        mask = sub_df["ID"] == match_id
        if mask.any():
            old = sub_df.loc[mask, "Pred"].values[0]
            sub_df.loc[mask, "Pred"] = pred
            print(f"  [override] {match_id}: {old:.4f} → {pred:.4f}")
        else:
            print(f"  [override] WARNING: {match_id} not found")
    return sub_df


def kenpom_rank_to_prob(rank1, rank2, scale=25.0):
    """
    Convert KenPom rank difference to a win probability for team1.
    Lower rank = better team.
    scale=25 means a rank gap of 25 gives ~73% win prob to the better team.
    scale=15 means a rank gap of 25 gives ~85% win prob.
    """
    diff = rank2 - rank1   # positive = team1 is better (lower rank number)
    return float(1.0 / (1.0 + np.exp(-diff / scale)))


def apply_kenpom_r1_boost(sub_df, seeds_df, kenpom_df,
                           season, kenpom_weight=0.75,
                           min_rank_gap=20):
    """
    For Round 1 tournament matchups only, blend model predictions
    heavily toward KenPom-implied win probabilities.

    Logic:
      - Identifies natural Round 1 pairings: same region, seed_a + seed_b = 17
        (e.g. W01 vs W16, X02 vs X15, Y05 vs Y12, etc.)
      - Converts KenPom rank difference to a probability via logistic function
      - Blends: (kenpom_weight * kp_prob) + ((1-kenpom_weight) * model_prob)
      - Only applies when BOTH teams have KenPom data AND rank gap >= min_rank_gap

    Parameters
    ----------
    kenpom_weight : float
        How much to trust KenPom vs the model. 0.75 = 75% KenPom, 25% model.
        The winner used essentially 1.0 for his manual overrides on clear cases.
    min_rank_gap : int
        Only apply the boost when KenPom rank difference is at least this large.
        Avoids interfering with close matchups where KenPom is less decisive.
        Default 20: if teams are ranked 5 and 180, that's a gap of 175 → very
        confident boost. If they're 40 and 55, gap=15 → skip, trust the model.
    """
    # Need seeds for this season
    seeds_s = seeds_df[seeds_df["Season"] == season].copy()
    if len(seeds_s) == 0:
        print(f"  [kenpom_r1] No {season} seeds found — skipping")
        return sub_df

    # Need KenPom for this season
    kp_s = kenpom_df[kenpom_df["Season"] == season] if kenpom_df is not None else pd.DataFrame()
    if len(kp_s) == 0:
        print(f"  [kenpom_r1] No {season} KenPom data — skipping")
        return sub_df

    # Build lookups
    seed_lookup = {}   # TeamID → (region, seed_num)
    for _, r in seeds_s.iterrows():
        region   = str(r["Seed"])[0]
        seed_num = parse_seed_num(r["Seed"])
        seed_lookup[int(r["TeamID"])] = (region, seed_num)

    kp_lookup = kp_s.set_index("TeamID")["kp_rank"].to_dict()  # TeamID → rank

    sub_df = sub_df.copy()
    n_applied = 0
    applied_rows = []

    for idx, row in sub_df.iterrows():
        parts  = row["ID"].split("_")
        s      = int(parts[0])
        t1, t2 = int(parts[1]), int(parts[2])

        if s != season:
            continue

        # Both teams must be seeded
        if t1 not in seed_lookup or t2 not in seed_lookup:
            continue

        reg1, sn1 = seed_lookup[t1]
        reg2, sn2 = seed_lookup[t2]

        # Must be same region AND seeds must sum to 17 (natural R1 pairing)
        # e.g. 1+16=17, 2+15=17, 3+14=17, 4+13=17, 5+12=17, 6+11=17, 7+10=17, 8+9=17
        if reg1 != reg2:
            continue
        if sn1 + sn2 != 17:
            continue

        # Both teams need KenPom rankings
        if t1 not in kp_lookup or t2 not in kp_lookup:
            continue

        kp1 = kp_lookup[t1]
        kp2 = kp_lookup[t2]
        rank_gap = abs(kp1 - kp2)

        # Only apply when KenPom has a strong opinion
        if rank_gap < min_rank_gap:
            continue

        # KenPom-implied probability that t1 wins
        kp_prob   = kenpom_rank_to_prob(kp1, kp2)
        model_prob = float(row["Pred"])
        blended   = kenpom_weight * kp_prob + (1 - kenpom_weight) * model_prob

        applied_rows.append({
            "ID":         row["ID"],
            "t1_seed":    f"{reg1}{sn1:02d}",
            "t2_seed":    f"{reg2}{sn2:02d}",
            "kp1":        kp1,
            "kp2":        kp2,
            "rank_gap":   rank_gap,
            "model_pred": round(model_prob, 4),
            "kp_prob":    round(kp_prob, 4),
            "blended":    round(blended, 4),
        })

        sub_df.at[idx, "Pred"] = blended
        n_applied += 1

    # Print a summary table
    print(f"\n  [kenpom_r1] Applied to {n_applied} Round 1 matchups "
          f"(kenpom_weight={kenpom_weight:.0%}, min_rank_gap={min_rank_gap})")

    if applied_rows:
        print(f"\n  {'Matchup':<14} {'KP1':>5} {'KP2':>5} {'Gap':>5} "
              f"{'Model':>7} {'KenPom':>8} {'Final':>7}")
        print(f"  {'-'*14} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*8} {'-'*7}")
        for r in sorted(applied_rows, key=lambda x: x["rank_gap"], reverse=True):
            print(f"  {r['t1_seed']} vs {r['t2_seed']:<8} "
                  f"{r['kp1']:>5} {r['kp2']:>5} {r['rank_gap']:>5} "
                  f"{r['model_pred']:>7.4f} {r['kp_prob']:>8.4f} "
                  f"{r['blended']:>7.4f}")

    return sub_df


# ──────────────────────────────────────────────────────────────────────────────
# 15b. SWING PICKS  (targeted upset bets in R1 upset zone)
# ──────────────────────────────────────────────────────────────────────────────

def find_best_swing_picks(sub_df, seeds_df, kenpom_df, season,
                           n_picks=8, min_seed=5, max_seed=12,
                           min_kenpom_edge=0.05):
    """
    Find Round 1 upset-zone games (5v12, 6v11, 7v10, 8v9) where KenPom
    strongly disagrees with the model in EITHER direction:

      Type A — Back the underdog: KenPom gives dog MORE than model does
      Type B — Back the favourite hard: model under-confident on fav,
               KenPom says the fav should win much more convincingly

    Both are valid swings. Ranked by how strongly KenPom disagrees with model.
    """
    seeds_s = seeds_df[seeds_df["Season"] == season].copy()
    kp_s    = (kenpom_df[kenpom_df["Season"] == season].copy()
               if kenpom_df is not None else pd.DataFrame())

    if len(seeds_s) == 0 or len(kp_s) == 0:
        print(f"  [swings] No {season} seeds or KenPom data")
        return pd.DataFrame()

    seed_lookup = {}
    for _, r in seeds_s.iterrows():
        region   = str(r["Seed"])[0]
        seed_num = parse_seed_num(r["Seed"])
        seed_lookup[int(r["TeamID"])] = (region, seed_num)

    kp_rank_lu = kp_s.set_index("TeamID")["kp_rank"].to_dict()
    kp_net_lu  = kp_s.set_index("TeamID")["kp_net"].to_dict()

    candidates = []

    for _, row in sub_df.iterrows():
        parts  = row["ID"].split("_")
        s      = int(parts[0])
        t1, t2 = int(parts[1]), int(parts[2])

        if s != season:
            continue
        if t1 not in seed_lookup or t2 not in seed_lookup:
            continue

        reg1, sn1 = seed_lookup[t1]
        reg2, sn2 = seed_lookup[t2]

        if reg1 != reg2 or sn1 + sn2 != 17:
            continue
        if not (min_seed <= sn1 <= max_seed and min_seed <= sn2 <= max_seed):
            continue
        if t1 not in kp_rank_lu or t2 not in kp_rank_lu:
            continue

        kp1      = kp_rank_lu[t1]
        kp2      = kp_rank_lu[t2]
        model_p  = float(row["Pred"])
        kp_p     = kenpom_rank_to_prob(kp1, kp2)

        fav_team = t1 if sn1 < sn2 else t2
        dog_team = t2 if sn1 < sn2 else t1
        fav_seed = min(sn1, sn2)
        dog_seed = max(sn1, sn2)

        model_fav_p = model_p if fav_team == t1 else (1 - model_p)
        kp_fav_p    = kp_p    if fav_team == t1 else (1 - kp_p)
        model_dog_p = 1 - model_fav_p
        kp_dog_p    = 1 - kp_fav_p

        kp_net_gap = kp_net_lu.get(dog_team, 0) - kp_net_lu.get(fav_team, 0)

        def make_candidate(bet_team, bet_seed, model_bet_p, kp_bet_p, swing_type):
            edge     = kp_bet_p - model_bet_p
            exp_gain = kp_bet_p * (1 - kp_bet_p) ** 2
            return {
                "type":          swing_type,
                "ID":            row["ID"],
                "matchup":       f"{reg1}{fav_seed:02d} vs {reg1}{dog_seed:02d}",
                "bet_team":      bet_team,
                "bet_seed":      bet_seed,
                "fav_team":      fav_team,
                "dog_team":      dog_team,
                "fav_seed":      fav_seed,
                "dog_seed":      dog_seed,
                "kp_fav":        int(kp_rank_lu[fav_team]),
                "kp_dog":        int(kp_rank_lu[dog_team]),
                "kp_net_gap":    round(kp_net_gap, 1),
                "model_bet_p":   round(model_bet_p, 4),
                "kp_bet_p":      round(kp_bet_p, 4),
                "kp_edge":       round(edge, 4),
                "exp_gain":      round(exp_gain, 5),
                "dist_from_1_3": round(abs(kp_bet_p - 1/3), 4),
            }

        # Type A: back the underdog (KenPom gives dog more than model)
        edge_dog = kp_dog_p - model_dog_p
        if edge_dog >= min_kenpom_edge:
            candidates.append(make_candidate(
                dog_team, dog_seed, model_dog_p, kp_dog_p, "A: Back Dog"))

        # Type B: back the favourite harder (KenPom gives fav more than model)
        edge_fav = kp_fav_p - model_fav_p
        if edge_fav >= min_kenpom_edge:
            candidates.append(make_candidate(
                fav_team, fav_seed, model_fav_p, kp_fav_p, "B: Back Fav"))

    if not candidates:
        print("  [swings] No qualifying matchups found")
        return pd.DataFrame()

    df = pd.DataFrame(candidates)
    df["swing_score"] = (df["kp_edge"] * 2.0
                         + df["exp_gain"]
                         - df["dist_from_1_3"] * 0.3)
    df = df.sort_values("swing_score", ascending=False).reset_index(drop=True)
    return df.head(n_picks)


def print_swing_candidates(df, team_names=None):
    """Pretty-print swing candidate table for user to choose from."""
    if len(df) == 0:
        print("  No swing candidates found.")
        return

    print(f"\n  ── Swing Pick Candidates ──────────────────────────────────────────")
    print(f"  {'#':>2}  {'Type':<12} {'Matchup':<12} {'KP Fav':>7} {'KP Dog':>7} "
          f"{'Net Δ':>6} {'ModelBet':>9} {'KPBet':>7} {'Edge':>6} {'Bet on'}")
    print(f"  {'--':>2}  {'-'*12} {'-'*12} {'-'*7} {'-'*7} "
          f"{'-'*6} {'-'*9} {'-'*7} {'-'*6} {'-'*20}")

    for i, r in df.iterrows():
        bet_name = team_names.get(int(r["bet_team"]), str(r["bet_team"])) \
                   if team_names else str(r["bet_team"])
        print(f"  {i:>2}  {r['type']:<12} {r['matchup']:<12} "
              f"{r['kp_fav']:>7} {r['kp_dog']:>7} "
              f"{r['kp_net_gap']:>+6.1f} {r['model_bet_p']:>9.4f} "
              f"{r['kp_bet_p']:>7.4f} {r['kp_edge']:>+6.4f}  {bet_name}")


def apply_swing_picks(sub_df, swing_df, picks, confidence=0.82):
    """
    Apply chosen swing picks to submission.

    picks : list of row indices from swing_df to bet on.
    confidence : how confidently to predict the chosen team wins.
        0.33 = optimal expected Brier (safest improvement)
        0.70 = moderate swing
        0.82 = aggressive (winner's style)
        0.99 = all-in

    Works for both Type A (back dog) and Type B (back fav harder).
    """
    sub_df = sub_df.copy()
    applied = []

    for i in picks:
        if i >= len(swing_df):
            print(f"  [swing] Index {i} out of range, skipping")
            continue

        pick     = swing_df.iloc[i]
        match_id = pick["ID"]
        bet_team = int(pick["bet_team"])

        mask = sub_df["ID"] == match_id
        if not mask.any():
            continue

        parts    = match_id.split("_")
        t1, t2   = int(parts[1]), int(parts[2])
        old_pred = float(sub_df.loc[mask, "Pred"].values[0])

        # If bet_team is t1 → pred = confidence, else pred = 1-confidence
        new_pred = confidence if bet_team == t1 else (1.0 - confidence)
        sub_df.loc[mask, "Pred"] = new_pred

        applied.append({
            "matchup":  pick["matchup"],
            "type":     pick["type"],
            "bet_seed": pick["bet_seed"],
            "old":      round(old_pred, 4),
            "kp":       pick["kp_bet_p"],
            "new":      round(new_pred, 4),
        })

    if applied:
        print(f"\n  [swings] {len(applied)} swing pick(s) applied "
              f"(confidence={confidence:.0%}):")
        print(f"  {'Matchup':<12} {'Type':<12} {'Seed':>4}  {'Old':>7} {'KP':>7} {'→ New':>8}")
        print(f"  {'-'*12} {'-'*12} {'-'*4}  {'-'*7} {'-'*7} {'-'*8}")
        for r in applied:
            print(f"  {r['matchup']:<12} {r['type']:<12} {r['bet_seed']:>4}  "
                  f"{r['old']:>7.4f} {r['kp']:>7.4f} {r['new']:>8.4f}")

    return sub_df


# ──────────────────────────────────────────────────────────────────────────────
# 15. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run(data_dir=DEFAULT_DATA_DIR,
        output="submission.csv",
        validate=True,
        boost=True,
        kenpom_r1=True,
        kenpom_weight=0.75,
        kenpom_min_gap=20,
        swing_picks=None,
        swing_confidence=0.82,
        manual_overrides=None):

    print("=" * 65)
    print("  March Machine Learning Mania 2026 — Championship Solution")
    print("=" * 65)

    # ── 1. Load ────────────────────────────────────────────────────
    print(f"\n[1/8] Loading data from {Path(data_dir).resolve()}")
    dfs = load_data(data_dir)

    # ── 2. Prepare base data ───────────────────────────────────────
    print("\n[2/8] Preparing data (OT adjustment + T1/T2 doubling)")

    m_reg = prepare_data(dfs["m_reg_detailed"])
    w_reg = prepare_data(dfs["w_reg_detailed"])
    regular_data = pd.concat([m_reg, w_reg], ignore_index=True)

    tc_m = dfs["m_tourn_compact"].copy()
    tc_m["men_women"] = 1
    tc_w = dfs["w_tourn_compact"].copy()
    tc_w["men_women"] = 0
    tourney_data = pd.concat([
        compact_to_matchups(tc_m, 1),
        compact_to_matchups(tc_w, 0),
    ], ignore_index=True)

    print(f"  Regular season: {len(regular_data):,} rows")
    print(f"  Tournament:     {len(tourney_data):,} rows (doubled)")

    # ── 3. Build features ──────────────────────────────────────────
    print("\n[3/8] Building features")

    seeds_all        = pd.concat([dfs["m_seeds"], dfs["w_seeds"]],
                                   ignore_index=True)
    seeds_T1, seeds_T2 = build_seeds(seeds_all)

    print("  Season averages...")
    ss_T1, ss_T2 = build_season_avgs(regular_data)

    print("  Elo ratings...")
    elo_T1, elo_T2 = build_elo(regular_data)

    print("  GLM team quality...")
    glm_T1, glm_T2 = build_glm_quality(regular_data, seeds_all)

    print("  Massey ordinals...")
    massey_T1, massey_T2 = build_massey(dfs["m_massey"])

    print("  KenPom 2026...")
    kenpom_df = load_kenpom(dfs["m_spellings"], dfs["m_teams"])

    print("  Betting odds...")
    tourney_seasons = sorted(tourney_data["Season"].unique().tolist())
    odds_df = load_odds(
        dfs["m_spellings"], dfs["m_teams"],
        seasons=list(set(tourney_seasons + [TARGET_SEASON])),
    )

    # ── 4. Assemble training data ──────────────────────────────────
    print("\n[4/8] Assembling training dataset")
    train_df = assemble(
        tourney_data,
        seeds_T1, seeds_T2,
        ss_T1, ss_T2,
        elo_T1, elo_T2,
        glm_T1, glm_T2,
        massey_T1, massey_T2,
        kenpom_df, odds_df,
    )
    features = get_features(train_df)
    print(f"  Rows: {len(train_df):,}   Features: {len(features)}")

    # ── 5. Train LOSO ensemble ─────────────────────────────────────
    print("\n[5/8] Leave-one-season-out training")
    models, oof_pred, oof_tgt = train_loso(train_df, features)

    # ── 6. Fit spline + evaluate ───────────────────────────────────
    print("\n[6/8] Fitting spline calibration (margin → probability)")
    spline       = fit_spline(oof_pred, oof_tgt)
    spline_probs = apply_spline(spline, oof_pred)
    oof_brier    = brier_score_loss((oof_tgt > 0).astype(int), spline_probs)
    print(f"  OOF Brier (all seasons): {oof_brier:.5f}")
    print(f"  Winner 2025 target:      ~0.113")

    if validate:
        print("\n  Per-season breakdown:")
        for s in sorted(train_df["Season"].unique()):
            mask = train_df["Season"] == s
            p    = apply_spline(spline, oof_pred[mask])
            y    = (oof_tgt[mask] > 0).astype(int)
            print(f"    {s}:  Brier={brier_score_loss(y, p):.5f}  "
                  f"N={int(mask.sum())}")

    # ── 7. Build submission features ──────────────────────────────
    print("\n[7/8] Building submission features")
    sub_raw = dfs["sample_sub"].copy()
    parts   = sub_raw["ID"].str.split("_", expand=True)
    sub_raw["Season"]    = parts[0].astype(int)
    sub_raw["T1_TeamID"] = parts[1].astype(int)
    sub_raw["T2_TeamID"] = parts[2].astype(int)
    sub_raw["men_women"] = (sub_raw["T1_TeamID"].astype(str)
                            .str.startswith("1").astype(int))
    sub_raw["PointDiff"] = 0
    sub_raw["win"]       = 0

    sub_feats = assemble(
        sub_raw,
        seeds_T1, seeds_T2,
        ss_T1, ss_T2,
        elo_T1, elo_T2,
        glm_T1, glm_T2,
        massey_T1, massey_T2,
        kenpom_df, odds_df,
    )

    # ── 8. Predict and save ────────────────────────────────────────
    print("\n[8/8] Generating predictions")
    raw_margins = predict_loso(models, sub_feats, features)
    probs       = apply_spline(spline, raw_margins)

    sub = pd.DataFrame({"ID": sub_raw["ID"], "Pred": probs})

    # KenPom Round 1 boost — applied first
    if kenpom_r1:
        print("\n  Applying KenPom Round 1 weighting...")
        sub = apply_kenpom_r1_boost(
            sub, seeds_all, kenpom_df,
            season=TARGET_SEASON,
            kenpom_weight=kenpom_weight,
            min_rank_gap=kenpom_min_gap,
        )

    # Swing picks — find best candidates and apply chosen ones
    swing_candidates = find_best_swing_picks(
        sub, seeds_all, kenpom_df, season=TARGET_SEASON)

    if len(swing_candidates):
        # Build team name lookup for display
        all_names = pd.concat([
            dfs["m_teams"][["TeamID", "TeamName"]],
            dfs["w_teams"][["TeamID", "TeamName"]],
        ], ignore_index=True).set_index("TeamID")["TeamName"].to_dict()

        print_swing_candidates(swing_candidates, all_names)

        if swing_picks is not None and len(swing_picks) > 0:
            sub = apply_swing_picks(sub, swing_candidates,
                                     swing_picks, swing_confidence)
        else:
            print(f"\n  [swings] No swing picks applied this run.")
            print(f"  To apply them, re-run with e.g.:")
            print(f"    --swing_picks 0 1   (bet on top 2 candidates)")

    if boost:
        sub = boost_predictions(sub)

    if manual_overrides:
        sub = apply_manual_overrides(sub, manual_overrides)

    sub["Pred"] = sub["Pred"].clip(0.01, 0.99).round(6)
    assert len(sub) == len(sub_raw), "Row count mismatch!"
    sub.to_csv(output, index=False)

    print(f"\n✅  Saved → {output}")
    print(f"    rows={len(sub):,}  "
          f"min={sub['Pred'].min():.4f}  "
          f"max={sub['Pred'].max():.4f}  "
          f"mean={sub['Pred'].mean():.4f}")
    print(f"\n  OOF Brier: {oof_brier:.5f}")

    return sub, models, spline


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",         default=DEFAULT_DATA_DIR,
                   help="Folder containing Kaggle CSV files")
    p.add_argument("--output",           default="submission.csv")
    p.add_argument("--no_validate",      action="store_true",
                   help="Skip per-season Brier breakdown")
    p.add_argument("--no_boost",         action="store_true",
                   help="Skip general prediction boosting")
    p.add_argument("--no_kenpom_r1",     action="store_true",
                   help="Skip KenPom Round 1 weighting for heavy favourites")
    p.add_argument("--kenpom_weight",    type=float, default=0.75,
                   help="KenPom weight for R1 favourite blending (0-1)")
    p.add_argument("--kenpom_min_gap",   type=int, default=20,
                   help="Min KenPom rank gap to apply R1 favourite boost")
    p.add_argument("--swing_picks",      type=int, nargs="*", default=None,
                   help="Indices of swing pick candidates to bet on (e.g. --swing_picks 0 1 2). "
                        "Run once without this flag to see the candidates table first.")
    p.add_argument("--swing_confidence", type=float, default=0.82,
                   help="Prediction confidence for swing picks (0.33=optimal expected, "
                        "0.82=aggressive, 0.99=all-in)")
    a = p.parse_args()

    run(data_dir=a.data_dir,
        output=a.output,
        validate=not a.no_validate,
        boost=not a.no_boost,
        kenpom_r1=not a.no_kenpom_r1,
        kenpom_weight=a.kenpom_weight,
        kenpom_min_gap=a.kenpom_min_gap,
        swing_picks=a.swing_picks,
        swing_confidence=a.swing_confidence)