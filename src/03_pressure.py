"""Step 3: define PRESSURE as a number for every ball.
Pressure = LEVERAGE = how much the next ball can swing the match.
Leverage = WP(after a six) - WP(after a wicket), from a win-probability model."""
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss
from scipy.stats import spearmanr

O = Path("outputs")
b = pd.read_parquet(O / "balls.parquet")
b["balls_left"] = b["balls_left"].clip(lower=0)

FEATS = ["innings", "score_before", "wkts_before", "balls_left", "runs_needed",
         "rrr", "crr", "is_ipl"]

def make_X(df):
    return df[FEATS].copy()

def after_state(df, runs, wicket):
    """Match state AFTER a hypothetical ball (runs scored, wicket or not)."""
    s = df.copy()
    s["score_before"] = df["score_before"] + runs
    s["wkts_before"] = (df["wkts_before"] + wicket).clip(upper=10)
    s["balls_left"] = (df["balls_left"] - 1).clip(lower=0)
    legal_after = 120 - s["balls_left"]
    s["crr"] = np.where(legal_after > 0, s["score_before"] / (legal_after / 6), np.nan)
    s["runs_needed"] = np.where(df["innings"] == 2, df["target"] - s["score_before"], np.nan)
    s["rrr"] = np.where(df["innings"] == 2,
                        s["runs_needed"] / (s["balls_left"].clip(lower=1) / 6), np.nan)
    return s

# 1. win-probability model, tested on matches it has never seen
lab = b[b["bat_team_won"].notna()].copy()
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr, te = next(gss.split(lab, groups=lab["match_id"]))
clf = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08,
                                     max_iter=250, random_state=42)
clf.fit(make_X(lab.iloc[tr]), lab.iloc[tr]["bat_team_won"])
p_te = clf.predict_proba(make_X(lab.iloc[te]))[:, 1]
y_te = lab.iloc[te]["bat_team_won"]
print("WIN-PROBABILITY MODEL (tested on unseen matches)")
print("  AUC     :", round(roc_auc_score(y_te, p_te), 3))
print("  log-loss:", round(log_loss(y_te, p_te), 3), "(coin flip = 0.693)")
for inn in [1, 2]:
    mk = lab.iloc[te]["innings"].values == inn
    print(f"  AUC innings {inn}:", round(roc_auc_score(y_te[mk], p_te[mk]), 3))

clf.fit(make_X(lab), lab["bat_team_won"])   # refit on everything

# 2. leverage for every ball
b["wp_before"] = clf.predict_proba(make_X(b))[:, 1]
wp_six = clf.predict_proba(make_X(after_state(b, 6, 0)))[:, 1]
wp_wkt = clf.predict_proba(make_X(after_state(b, 0, 1)))[:, 1]
b["leverage"] = wp_six - wp_wkt

# 3. tiers (percentile within each innings)
b["lev_pct"] = b.groupby("innings")["leverage"].rank(pct=True)
b["pressure_tier"] = pd.cut(b["lev_pct"], [0, 0.5, 0.8, 1.0],
                            labels=["low", "medium", "high"], include_lowest=True)
b.to_parquet(O / "pressure.parquet", index=False)

# 4. PROOF 1: outcomes by tier
def summ(df):
    return pd.Series({
        "balls": len(df),
        "runs/ball": df["total_runs"].mean(),
        "dot_%": 100 * ((df["total_runs"] == 0) & (df["legal"] == 1)).mean(),
        "boundary_%": 100 * (df["is_four"] | df["is_six"]).mean(),
        "wicket_%": 100 * df["is_wicket"].mean(),
    })
print("\nPROOF 1: ball outcomes by pressure tier")
print(b.groupby(["innings", "pressure_tier"], observed=True).apply(summ).round(2).to_string())
print("\n% of balls that are HIGH pressure, by phase:")
print((100 * b.assign(h=b.pressure_tier == "high")
       .groupby(["innings", "phase"], observed=True)["h"].mean()).round(1).to_string())

# 5. PROOF 2: does leverage predict the real swing?
b = b.sort_values(["match_id", "innings", "innings_ball_num"])
b["wp_next"] = b.groupby(["match_id", "innings"])["wp_before"].shift(-1)
b["swing"] = (b["wp_next"] - b["wp_before"]).abs()
print("\nPROOF 2: average REAL win-probability swing, by tier")
print(b.groupby(["innings", "pressure_tier"], observed=True)["swing"].mean().round(4).to_string())
for inn in [1, 2]:
    d = b[(b.innings == inn) & b["swing"].notna()]
    print(f"Innings {inn}: rank correlation leverage vs real swing =",
          round(spearmanr(d["leverage"], d["swing"])[0], 3))
print("\nInnings 2: average leverage by required run rate")
d2 = b[b.innings == 2]
print(d2.groupby(pd.cut(d2["rrr"], [-1, 6, 8, 10, 12, 15, 100]),
                 observed=True)["leverage"].mean().round(3).to_string())
print("\nSaved outputs/pressure.parquet")