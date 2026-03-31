"""
kenpom.py
=========
Embeds 2026 KenPom data directly and provides a function to attach
KenPom features to the FeatureStore for any season where we have data.

KenPom features are among the strongest predictors in this competition
because they are fully tempo-adjusted and opponent-adjusted — far better
than raw box score calculations.

Columns embedded:
  kenpom_rank     — overall KenPom national rank
  kenpom_net_rtg  — adjusted net efficiency (pts/100 poss above avg)
  kenpom_off_rtg  — adjusted offensive efficiency
  kenpom_def_rtg  — adjusted defensive efficiency
  kenpom_adj_t    — adjusted tempo (possessions per 40 min)
  kenpom_luck     — luck rating (actual wins vs expected wins)
  kenpom_sos_net  — strength of schedule: opponent net rating
  kenpom_sos_off  — strength of schedule: opponent offensive rating
  kenpom_sos_def  — strength of schedule: opponent defensive rating
  kenpom_ncsos    — non-conference SOS net rating
  kenpom_seed     — tournament seed (0 if not in tournament)
"""

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 2026 KENPOM DATA  (embedded directly from document)
# Format: (rank, team_name, seed, conf, W, L,
#          net_rtg, off_rtg, def_rtg, adj_t, luck,
#          sos_net, sos_off, sos_def, ncsos_net)
# seed=0 means not in tournament
# ──────────────────────────────────────────────────────────────────────────────

KENPOM_2026_RAW = [
    #  rk  name                    seed  conf   W   L    net     off    def    tempo   luck    sos_net  sos_off  sos_def  ncsos
    (  1, "Duke",                    1, "ACC",  32,  2, +38.88, 128.0,  89.1,  65.4, +.049,  +14.29, 117.0, 102.7, +10.10),
    (  2, "Arizona",                 1, "B12",  32,  2, +37.60, 127.6,  90.0,  69.8, +.023,  +14.93, 117.6, 102.7,  +3.21),
    (  3, "Michigan",                1, "B10",  31,  3, +37.58, 126.6,  89.0,  71.0, +.045,  +16.66, 119.2, 102.5, +12.49),
    (  4, "Florida",                 1, "SEC",  26,  7, +33.75, 125.5,  91.8,  70.6, -.036,  +15.99, 119.4, 103.4,  +7.93),
    (  5, "Houston",                 2, "B12",  28,  6, +33.37, 124.8,  91.5,  63.3, -.006,  +13.55, 116.9, 103.3,  +0.88),
    (  6, "Iowa St.",                2, "B12",  27,  7, +32.37, 123.8,  91.4,  66.6, -.012,  +12.40, 115.5, 103.1,  -3.56),
    (  7, "Illinois",                3, "B10",  24,  8, +32.09, 131.2,  99.1,  65.6, -.050,  +13.65, 117.1, 103.4,  +3.77),
    (  8, "Purdue",                  2, "B10",  27,  8, +31.18, 131.6, 100.5,  64.4, -.006,  +15.88, 118.3, 102.5,  +6.28),
    (  9, "Michigan St.",            3, "B10",  25,  7, +28.30, 123.0,  94.7,  66.1, +.005,  +13.70, 118.1, 104.4,  +3.65),
    ( 10, "Gonzaga",                 3, "WCC",  30,  3, +28.19, 122.1,  93.9,  68.6, +.072,   +6.01, 113.0, 107.0,  +8.32),
    ( 11, "Connecticut",             2, "BE",   29,  5, +27.84, 122.0,  94.2,  64.5, +.055,  +12.01, 115.3, 103.3,  +7.63),
    ( 12, "Vanderbilt",              5, "SEC",  26,  8, +27.48, 126.8,  99.3,  68.9, +.018,  +14.55, 118.0, 103.4,  +1.20),
    ( 13, "Virginia",                3, "ACC",  29,  5, +26.69, 122.6,  95.9,  65.8, +.056,   +9.95, 115.4, 105.4,  -3.24),
    ( 14, "Nebraska",                4, "B10",  26,  6, +26.15, 118.5,  92.4,  66.8, +.034,  +11.61, 116.4, 104.8,  -6.01),
    ( 15, "Arkansas",                4, "SEC",  26,  8, +26.01, 127.7, 101.6,  71.0, +.051,  +14.93, 118.8, 103.9,  +4.44),
    ( 16, "Tennessee",               6, "SEC",  22, 11, +25.98, 121.0,  95.1,  65.0, -.060,  +14.74, 119.0, 104.3,  +0.77),
    ( 17, "St. John's",              5, "BE",   28,  6, +25.87, 120.0,  94.2,  69.6, +.061,  +11.50, 115.1, 103.6,  +6.57),
    ( 18, "Alabama",                 4, "SEC",  23,  9, +25.68, 129.0, 103.3,  73.1, +.019,  +16.72, 119.4, 102.7, +13.43),
    ( 19, "Louisville",              6, "ACC",  23, 10, +25.40, 124.0,  98.7,  69.7, -.020,  +12.55, 115.8, 103.3,  +2.41),
    ( 20, "Texas Tech",              5, "B12",  22, 10, +25.19, 125.0,  99.9,  66.2, +.006,  +15.62, 118.3, 102.7,  +7.79),
    ( 21, "Kansas",                  4, "B12",  23, 10, +24.40, 118.3,  93.9,  67.7, +.053,  +16.92, 118.0, 101.0, +11.01),
    ( 22, "Wisconsin",               5, "B10",  24, 10, +23.39, 125.3, 102.0,  68.8, +.041,  +13.94, 117.2, 103.3,  -1.08),
    ( 23, "BYU",                     6, "B12",  23, 11, +23.21, 125.4, 102.2,  70.0, -.017,  +14.25, 116.2, 102.0,  +4.90),
    ( 24, "Saint Mary's",            7, "WCC",  27,  5, +23.15, 120.4,  97.2,  65.2, +.011,   +5.09, 111.5, 106.4,  +3.87),
    ( 25, "Iowa",                    9, "B10",  21, 12, +22.44, 121.7,  99.3,  63.0, -.061,  +11.39, 115.6, 104.2,  -7.17),
    ( 26, "Ohio St.",                8, "B10",  21, 12, +22.23, 124.3, 102.1,  66.1, -.031,  +13.68, 116.7, 103.0,  -1.81),
    ( 27, "UCLA",                    7, "B10",  23, 11, +21.66, 123.7, 102.1,  64.7, +.017,  +12.24, 116.1, 103.9,  -2.16),
    ( 28, "Kentucky",                7, "SEC",  21, 13, +21.45, 120.5,  99.1,  68.4, -.019,  +15.91, 119.0, 103.1,  +1.35),
    ( 29, "Utah St.",                9, "MWC",  28,  6, +20.84, 122.2, 101.4,  67.7, +.065,   +7.23, 112.8, 105.6,  +2.01),
    ( 30, "North Carolina",          6, "ACC",  24,  8, +20.82, 121.4, 100.6,  68.0, +.057,  +11.46, 115.4, 103.9,  +1.33),
    ( 31, "Miami FL",                7, "ACC",  25,  8, +20.67, 121.4, 100.7,  67.6, +.021,   +8.00, 113.6, 105.6,  -7.27),
    ( 32, "Georgia",                 8, "SEC",  22, 10, +20.45, 124.7, 104.3,  71.4, -.005,  +10.76, 115.9, 105.2,  -6.43),
    ( 33, "Villanova",               8, "BE",   24,  8, +19.94, 120.4, 100.4,  65.2, +.067,  +10.36, 114.7, 104.4,  +2.88),
    ( 34, "N.C. State",             11, "ACC",  20, 14, +19.49, 123.2, 103.7,  69.1, -.043,  +12.25, 116.3, 104.0,  +4.19),
    ( 35, "Santa Clara",            10, "WCC",  26,  8, +19.48, 123.7, 104.2,  69.2, +.015,   +6.13, 111.9, 105.8,  +3.94),
    ( 36, "Clemson",                 8, "ACC",  24, 10, +19.21, 116.5,  97.3,  64.3, +.011,  +10.53, 115.2, 104.7,  +0.64),
    ( 37, "Texas",                  11, "SEC",  19, 14, +19.13, 124.2, 105.0,  66.9, -.068,  +13.95, 117.6, 103.6,  -6.36),
    ( 38, "Auburn",                  0, "SEC",  18, 16, +18.68, 124.6, 105.9,  67.2, -.045,  +15.91, 118.9, 103.0,  +8.39),
    ( 39, "Texas A&M",              10, "SEC",  21, 11, +18.63, 119.7, 101.1,  70.5, -.002,  +11.13, 116.9, 105.8,  -5.87),
    ( 40, "Oklahoma",                0, "SEC",  19, 15, +18.33, 124.2, 105.9,  66.3, -.070,  +13.27, 117.2, 103.9,  -5.55),
    ( 41, "Saint Louis",             9, "A10",  28,  5, +18.27, 119.5, 101.2,  71.0, +.030,   +1.01, 108.9, 107.8,  -5.78),
    ( 42, "SMU",                    11, "ACC",  20, 13, +18.07, 122.9, 104.8,  68.6, -.043,  +11.15, 115.7, 104.5,  +1.25),
    ( 43, "TCU",                     9, "B12",  22, 11, +17.53, 115.3,  97.8,  67.7, +.004,  +10.99, 115.2, 104.2,  -6.94),
    ( 44, "Cincinnati",              0, "B12",  18, 15, +17.45, 111.2,  93.7,  67.5, -.073,  +10.99, 114.8, 103.8,  -3.23),
    ( 45, "Indiana",                 0, "B10",  18, 14, +17.17, 120.3, 103.2,  66.1, -.041,  +11.81, 116.1, 104.3,  -4.96),
    ( 46, "VCU",                    11, "A10",  27,  7, +17.13, 119.9, 102.7,  68.5, -.007,   +3.44, 110.3, 106.9,  -1.26),
    ( 47, "San Diego St.",           0, "MWC",  22, 11, +16.46, 113.3,  96.9,  68.8, +.015,   +8.67, 114.3, 105.6,  +5.36),
    ( 48, "South Florida",          11, "Amer", 25,  8, +16.37, 117.3, 101.0,  71.5, -.026,   +3.06, 111.2, 108.2,  +4.95),
    ( 49, "Baylor",                  0, "B12",  16, 16, +15.96, 122.9, 107.0,  67.8, -.086,  +13.90, 115.8, 101.9,  +1.17),
    ( 50, "New Mexico",              0, "MWC",  23, 10, +15.88, 117.1, 101.2,  69.8, -.035,   +5.77, 111.9, 106.1,  +0.58),
    ( 51, "Seton Hall",              0, "BE",   21, 12, +15.69, 110.5,  94.8,  65.1, -.033,   +8.98, 113.9, 104.9,  -4.20),
    ( 52, "Missouri",               10, "SEC",  20, 12, +15.36, 119.5, 104.1,  66.2, +.041,  +11.50, 116.5, 105.0,  -7.75),
    ( 54, "UCF",                    10, "B12",  21, 11, +14.99, 120.4, 105.4,  69.2, +.097,  +11.88, 115.0, 103.1,  -2.53),
    ( 60, "Boise St.",               0, "MWC",  20, 12, +13.25, 117.1, 103.9,  65.7, +.028,   +8.53, 114.1, 105.6,  +7.73),
    ( 64, "Akron",                  12, "MAC",  29,  5, +12.79, 118.8, 106.1,  70.3, +.018,   -3.64, 108.5, 112.1,  -3.13),
    ( 66, "McNeese",                12, "Slnd", 28,  5, +12.54, 114.3, 101.8,  66.2, +.084,   -1.78, 107.6, 109.3,  +9.16),
    ( 71, "Northern Iowa",          12, "MVC",  23, 12, +11.80, 110.0,  98.2,  62.3, -.070,   +1.28, 109.5, 108.2,  -0.91),
    ( 87, "Hofstra",                13, "CAA",  24, 10,  +9.53, 114.6, 105.1,  64.7, -.052,   -0.86, 108.0, 108.9,  +3.09),
    ( 92, "High Point",             12, "BSth", 30,  4,  +8.40, 117.0, 108.6,  69.9, +.048,   -9.21, 103.7, 112.9,  -8.86),
    ( 93, "Miami OH",               11, "MAC",  31,  1,  +8.27, 116.8, 108.5,  70.0, +.099,   -5.34, 107.2, 112.6,  -9.88),
    (107, "Cal Baptist",            13, "WAC",  25,  8,  +5.99, 107.9, 101.9,  65.8, +.091,   -1.94, 107.1, 109.1,  -3.31),
    (109, "Hawaii",                 13, "BW",   24,  8,  +5.90, 107.1, 101.2,  69.7, +.038,   -3.42, 106.5, 109.9, -11.21),
    (115, "North Dakota St.",       14, "Sum",  27,  7,  +5.00, 111.7, 106.7,  66.3, +.040,   -5.92, 106.4, 112.3,  -0.94),
    (140, "Wright St.",             14, "Horz", 23, 11,  +2.04, 112.1, 110.0,  67.2, +.009,   -4.02, 109.4, 113.4,  +0.88),
    (143, "Troy",                   13, "SB",   22, 11,  +1.80, 110.8, 109.0,  64.9, +.024,   -3.14, 107.6, 110.7,  +4.19),
    (148, "Idaho",                  15, "BSky", 21, 14,  +1.51, 108.8, 107.3,  67.8, -.012,   -1.67, 109.4, 111.1,  +0.75),
    (151, "Penn",                   14, "Ivy",  18, 11,  +1.33, 107.3, 106.0,  69.1, +.068,   -0.80, 109.1, 109.9,  +0.45),
    (162, "Kennesaw St.",           14, "CUSA", 21, 13,  +0.76, 110.7, 110.0,  71.2, +.009,   -2.02, 108.1, 110.1,  -5.16),
    (183, "Queens",                 15, "ASun", 21, 13,  -1.44, 115.8, 117.2,  69.6, +.067,   -5.65, 108.4, 114.0,  +5.68),
    (186, "Tennessee St.",          15, "OVC",  23,  9,  -1.81, 109.1, 110.9,  70.2, +.070,   -8.23, 103.2, 111.5,  +5.51),
    (188, "UMBC",                   16, "AE",   24,  9,  -1.94, 108.8, 110.7,  66.3, +.028,  -14.45, 100.6, 115.1,  -6.98),
    (190, "Furman",                 15, "SC",   22, 12,  -1.97, 107.5, 109.4,  65.9, +.010,   -6.27, 108.1, 114.4,  -0.68),
    (193, "Siena",                  16, "MAAC", 23, 11,  -2.14, 107.1, 109.3,  64.7, +.005,   -9.50, 102.9, 112.4,  -7.77),
    (205, "Howard",                 16, "MEAC", 24, 10,  -2.92, 104.1, 107.0,  69.1, +.003,  -14.03, 100.8, 114.8,  -1.67),
    (216, "LIU",                    16, "NEC",  24, 10,  -3.96, 105.6, 109.6,  67.8, +.104,   -9.96, 103.6, 113.5,  +2.18),
    (284, "Lehigh",                 16, "PL",   18, 16, -10.41, 102.7, 113.1,  66.9, +.081,   -8.63, 104.9, 113.5,  -0.54),
    (288, "Prairie View",           16, "SWAC", 18, 17, -10.69, 101.2, 111.9,  71.0, +.013,   -9.56, 103.4, 113.0,  +8.93),
]

# Spelling aliases: KenPom name → MTeamSpellings-compatible name
KENPOM_ALIASES = {
    "duke":           "duke",
    "arizona":        "arizona",
    "michigan":       "michigan",
    "florida":        "florida",
    "houston":        "houston",
    "iowa st.":       "iowa st",
    "illinois":       "illinois",
    "purdue":         "purdue",
    "michigan st.":   "michigan st",
    "gonzaga":        "gonzaga",
    "connecticut":    "connecticut",
    "vanderbilt":     "vanderbilt",
    "virginia":       "virginia",
    "nebraska":       "nebraska",
    "arkansas":       "arkansas",
    "tennessee":      "tennessee",
    "st. john's":     "st john's",
    "alabama":        "alabama",
    "louisville":     "louisville",
    "texas tech":     "texas tech",
    "kansas":         "kansas",
    "wisconsin":      "wisconsin",
    "byu":            "byu",
    "saint mary's":   "saint mary's",
    "iowa":           "iowa",
    "ohio st.":       "ohio st",
    "ucla":           "ucla",
    "kentucky":       "kentucky",
    "utah st.":       "utah st",
    "north carolina": "north carolina",
    "miami fl":       "miami fl",
    "georgia":        "georgia",
    "villanova":      "villanova",
    "n.c. state":     "nc state",
    "santa clara":    "santa clara",
    "clemson":        "clemson",
    "texas":          "texas",
    "auburn":         "auburn",
    "texas a&m":      "texas a&m",
    "oklahoma":       "oklahoma",
    "saint louis":    "saint louis",
    "smu":            "smu",
    "tcu":            "tcu",
    "cincinnati":     "cincinnati",
    "indiana":        "indiana",
    "vcu":            "vcu",
    "san diego st.":  "san diego st",
    "south florida":  "south florida",
    "baylor":         "baylor",
    "new mexico":     "new mexico",
    "seton hall":     "seton hall",
    "missouri":       "missouri",
    "ucf":            "ucf",
    "boise st.":      "boise st",
    "akron":          "akron",
    "mcneese":        "mcneese st",
    "northern iowa":  "northern iowa",
    "hofstra":        "hofstra",
    "high point":     "high point",
    "miami oh":       "miami oh",
    "cal baptist":    "cal baptist",
    "hawaii":         "hawaii",
    "north dakota st.": "north dakota st",
    "wright st.":     "wright st",
    "troy":           "troy",
    "idaho":          "idaho",
    "penn":           "pennsylvania",
    "kennesaw st.":   "kennesaw st",
    "queens":         "queens nc",
    "tennessee st.":  "tennessee st",
    "umbc":           "umbc",
    "furman":         "furman",
    "siena":          "siena",
    "howard":         "howard",
    "liu":            "liu",
    "lehigh":         "lehigh",
    "prairie view":   "prairie view",
}


# ──────────────────────────────────────────────────────────────────────────────
# BUILD DATAFRAME + ATTACH TEAM IDs
# ──────────────────────────────────────────────────────────────────────────────

def build_kenpom_df(season: int = 2026) -> pd.DataFrame:
    """Build a DataFrame from the embedded KenPom data."""
    rows = []
    for (rank, name, seed, conf, wins, losses,
         net_rtg, off_rtg, def_rtg, adj_t, luck,
         sos_net, sos_off, sos_def, ncsos_net) in KENPOM_2026_RAW:
        rows.append({
            "Season":          season,
            "name_lower":      name.lower().strip(),
            "kenpom_rank":     rank,
            "kenpom_seed":     seed,
            "kenpom_wins":     wins,
            "kenpom_losses":   losses,
            "kenpom_win_pct":  wins / max(wins + losses, 1),
            "kenpom_net_rtg":  net_rtg,
            "kenpom_off_rtg":  off_rtg,
            "kenpom_def_rtg":  def_rtg,
            "kenpom_adj_t":    adj_t,
            "kenpom_luck":     luck,
            "kenpom_sos_net":  sos_net,
            "kenpom_sos_off":  sos_off,
            "kenpom_sos_def":  sos_def,
            "kenpom_ncsos":    ncsos_net,
        })
    return pd.DataFrame(rows)


def attach_kenpom_ids(kenpom_df: pd.DataFrame,
                       spellings_df: pd.DataFrame,
                       teams_df: pd.DataFrame) -> pd.DataFrame:
    """Map KenPom team names to TeamIDs via spellings file."""
    lookup = {}
    for _, r in spellings_df.iterrows():
        lookup[r["TeamNameSpelling"].lower().strip()] = int(r["TeamID"])
    for _, r in teams_df.iterrows():
        lookup.setdefault(r["TeamName"].lower().strip(), int(r["TeamID"]))

    matched, unmatched = [], []
    for _, row in kenpom_df.iterrows():
        raw = row["name_lower"]
        tid = lookup.get(raw)
        if tid is None:
            alias = KENPOM_ALIASES.get(raw)
            if alias:
                tid = lookup.get(alias.lower())
        if tid is None:
            for k, v in lookup.items():
                if raw in k or k in raw:
                    tid = v
                    break
        if tid is not None:
            r = row.drop("name_lower").to_dict()
            r["TeamID"] = tid
            matched.append(r)
        else:
            unmatched.append(raw)

    if unmatched:
        print(f"  [kenpom] Unmatched ({len(unmatched)}): {unmatched}")
    print(f"  [kenpom] Matched {len(matched)}/{len(kenpom_df)} teams "
          f"to TeamIDs (season {kenpom_df['Season'].iloc[0]})")
    return pd.DataFrame(matched) if matched else pd.DataFrame()


def get_kenpom_features(season: int,
                         spellings_df: pd.DataFrame,
                         teams_df: pd.DataFrame) -> pd.DataFrame:
    """One-call helper: build + attach IDs. Returns (Season, TeamID, kenpom_*) df."""
    kp_raw = build_kenpom_df(season)
    return attach_kenpom_ids(kp_raw, spellings_df, teams_df)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS (used by FeatureStore)
# ──────────────────────────────────────────────────────────────────────────────

KENPOM_FEAT_COLS = [
    "kenpom_rank", "kenpom_net_rtg", "kenpom_off_rtg", "kenpom_def_rtg",
    "kenpom_adj_t", "kenpom_luck", "kenpom_sos_net", "kenpom_sos_off",
    "kenpom_sos_def", "kenpom_ncsos", "kenpom_win_pct",
]