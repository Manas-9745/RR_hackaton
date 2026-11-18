"""Step 4+5: rate batters and bowlers under pressure, then check reliability."""
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

# ======================= CONFIG (YOUR DECISIONS) =======================
K_WICKET   = 3       # runs a wicket is worth (kept small: wickets are rare and noisy)
HIGH_PCT   = 0.80    # 'high pressure' = top 20% leverage; used for eligibility, consistency, raw stats
W_INN1     = 0.5     # weight of first-innings pressure balls (second innings = 1.0)
MIN_BAT    = 80      # min high-pressure balls faced
MIN_BOWL   = 80      # min high-pressure balls bowled
W_PRESS, W_DELTA, W_CONS = 0.7, 0.15, 0.15 # final score weights
MIN_MATCH_BALLS = 6  # min pressure balls in a match to count it for consistency
# ======================================================================

O = Path("outputs")
b = pd.read_parquet(O / "pressure.parquet")
b["balls_left"] = b["balls_left"].clip(lower=0)
b["spin"] = (b["pace_or_spin"] == "Spin").astype(int)
b["runs_charged"] = b["runs"] + b["extras_type"].isin(["wide", "Overstep"]).astype(int)
b["bat_out"] = ((b["is_wicket"] == 1) & (b["dismissed_player_id"] == b["striker_id"])).astype(int)

# ---- opponent quality (shrunk toward the league mean) ----
def shrunk_rate(df, key, col, n0=300):
    g = df.groupby(key)[col].agg(["sum", "count"])
    mu = df[col].mean()
    return (g["sum"] + n0 * mu) / (g["count"] + n0)
bf = b[b.bat_ball_faced == 1]
b["bowler_q"] = b["bowler_id"].map(shrunk_rate(b, "bowler_id", "runs_charged"))
b["batter_q"] = b["striker_id"].map(shrunk_rate(bf, "striker_id", "runs"))

BASE = ["innings", "score_before", "wkts_before", "balls_left", "runs_needed", "rrr",
        "crr", "is_ipl", "overid", "spin"]

def oof(df, feats, target, clf=False):
    """Out-of-fold expected value (each match predicted by a model that never saw it)."""
    pred = np.zeros(len(df))
    X, y = df[feats], df[target]
    for tr, te in GroupKFold(4).split(X, y, groups=df["match_id"]):
        m = (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(
            max_depth=5, learning_rate=0.08, max_iter=150, random_state=42)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred[te] = m.predict_proba(X.iloc[te])[:, 1] if clf else m.predict(X.iloc[te])
    return pred

print("Fitting expected-outcome models (a few minutes)...")
bat = b[b.bat_ball_faced == 1].copy()
fb = BASE + ["bowler_q"]
bat["E_runs"] = oof(bat, fb, "runs")
bat["E_out"] = oof(bat, fb, "bat_out", clf=True)
bowl = b.copy()
fw = BASE + ["batter_q"]
bowl["E_runs"] = oof(bowl, fw, "runs_charged")
bowl["E_wkt"] = oof(bowl, fw, "bowler_wkt", clf=True)
print("Done.")

# ---------------- core rating function ----------------
def eb_shrink(mean, n_eff, sigma2):
    """Empirical-Bayes shrinkage toward 0 (= performing exactly as expected)."""
    tau2 = max(mean.var() - (sigma2 / n_eff).mean(), 0.05 * mean.var())
    n0 = sigma2 / tau2
    return mean * n_eff / (n_eff + n0), n0

def rate(df, role, K=K_WICKET, hp=HIGH_PCT, w1=W_INN1, min_n=None, power=1, verbose=False):
    d = df.copy()
    if role == "bat":
        pid, name = "striker_id", "striker_name"
        d["val"] = (d["runs"] - d["E_runs"]) - K * (d["bat_out"] - d["E_out"])
    else:
        pid, name = "bowler_id", "bowler_name"
        d["val"] = -(d["runs_charged"] - d["E_runs"]) + K * (d["bowler_wkt"] - d["E_wkt"])
    d["high"] = d["lev_pct"] > hp
    # every ball counts in proportion to its pressure (leverage); 1st innings counts less
    d["w"] = d["leverage"].clip(lower=0) ** power * np.where(d["innings"] == 2, 1.0, w1)
    d["wv"] = d["w"] * d["val"]
    d["w2"] = d["w"] ** 2
    g = d.groupby(pid).agg(sw=("w", "sum"), swv=("wv", "sum"), sw2=("w2", "sum"),
                           n_all=("val", "size"))
    g["high_n"] = d[d.high].groupby(pid).size()
    N = d[d.lev_pct <= 0.5].groupby(pid).agg(n_n=("val", "size"), s_n=("val", "sum"))
    t = g.join(N, how="left").fillna(0)
    min_n = min_n or (MIN_BAT if role == "bat" else MIN_BOWL)
    t = t[(t.high_n >= min_n) & (t.n_n >= 30)].copy()
    t["n_eff"] = t["sw"] ** 2 / t["sw2"]
    t["raw_press"] = t["swv"] / t["sw"]
    wmean = d["wv"].sum() / d["w"].sum()
    sig_p = (d["w"] * (d["val"] - wmean) ** 2).sum() / d["w"].sum()
    t["press"], n0 = eb_shrink(t["raw_press"], t["n_eff"], sig_p)
    t["normal"], _ = eb_shrink(t["s_n"] / t["n_n"], t["n_n"], d.loc[d.lev_pct <= 0.5, "val"].var())
    t["delta"] = t["press"] - t["normal"]
    # consistency: share of matches (with enough high-pressure balls) beating expectation
    H = d[d.high]
    pm = H.groupby([pid, "match_id"]).agg(n=("val", "size"), v=("val", "sum"))
    pm = pm[pm.n >= MIN_MATCH_BALLS]
    pm["pos"] = (pm.v > 0).astype(int)
    c = pm.groupby(level=0).agg(m=("pos", "size"), p=("pos", "sum"))
    t["matches"] = c["m"]; t["cons"] = (c["p"] + 1) / (c["m"] + 2)
    t = t[t.matches.notna() & (t.matches >= 5)]
    z = lambda x: (x - x.mean()) / x.std()
    t["score"] = W_PRESS * z(t["press"]) + W_DELTA * z(t["delta"]) + W_CONS * z(t["cons"])
    nm = d.groupby(pid)[name].agg(lambda x: x.mode().iat[0])
    t["player"] = t.index.map(nm)
    t = t.sort_values("score", ascending=False)
    t["rank"] = np.arange(1, len(t) + 1)
    if verbose:
        print(f"  prior strength n0 = {n0:.0f} effective balls; eligible players = {len(t)}")
    return t, H

def raw_stats(H, t, role):
    """Plain cricket numbers on high-pressure balls, for the report tables."""
    pid = "striker_id" if role == "bat" else "bowler_id"
    g = H[H[pid].isin(t.index)].groupby(pid)
    if role == "bat":
        r = pd.DataFrame({"balls": g["bat_ball_faced"].sum(), "runs": g["runs"].sum(),
                          "outs": g["bat_out"].sum(),
                          "dot%": 100 * g.apply(lambda x: ((x.runs == 0) & (x.bat_ball_faced == 1)).mean()),
                          "bdry%": 100 * g.apply(lambda x: (x.is_four | x.is_six).mean())})
        r["SR"] = 100 * r["runs"] / r["balls"]
    else:
        r = pd.DataFrame({"balls": g["legal"].sum(), "runs": g["runs_charged"].sum(),
                          "wkts": g["bowler_wkt"].sum(),
                          "dot%": 100 * g.apply(lambda x: ((x.runs_charged == 0) & (x.legal == 1)).mean())})
        r["econ"] = 6 * r["runs"] / r["balls"]
    return r.round(1)

res = {}
for role, df in [("bat", bat), ("bowl", bowl)]:
    print(f"\n================ {'BATTERS' if role == 'bat' else 'BOWLERS'} ================")
    t, H = rate(df, role, verbose=True)
    t = t.join(raw_stats(H, t, role))
    res[role] = t
    cols = ["player", "rank", "score", "press", "normal", "delta", "cons", "matches"] + \
           (["balls", "SR", "dot%", "bdry%"] if role == "bat" else ["balls", "econ", "dot%", "wkts"])
    show = t[cols].copy()
    for c in ["score", "press", "normal", "delta"]:
        show[c] = show[c].round(3)
    show["cons"] = show["cons"].round(2)
    print(show.head(15).to_string(index=False))
    t.reset_index().to_csv(O / f"{role}_ratings.csv", index=False)

    # ---- reliability 1: split-half (odd vs even matches) ----
    ids = df["match_id"].unique()
    A, B = df[df.match_id.isin(ids[::2])], df[df.match_id.isin(ids[1::2])]
    ta, _ = rate(A, role, min_n=30); tb, _ = rate(B, role, min_n=30)
    j = ta[["press"]].join(tb[["press"]], lsuffix="_a", rsuffix="_b", how="inner")
    print(f"\n  Split-half reliability (odd vs even matches, {len(j)} players): "
          f"rank correlation = {spearmanr(j.press_a, j.press_b)[0]:.2f}")

    # ---- reliability 2: sensitivity of the top 10 ----
    base = set(t.head(10).index)
    print("  Sensitivity: how many of the base top 10 stay in the top 10?")
    for lab, kw in [("wicket worth 0 runs", dict(K=0)), ("wicket worth 6 runs", dict(K=6)),
                    ("wicket worth 10 runs", dict(K=10)),
                    ("pressure weight squared", dict(power=2)),
                    ("1st-innings weight 0", dict(w1=0.0)), ("1st-innings weight 1", dict(w1=1.0)),
                    ("eligibility 120 balls", dict(min_n=120))]:
        tv, _ = rate(df, role, **kw)
        print(f"    {lab:24s}: {len(base & set(tv.head(10).index))}/10")

print("\nSaved outputs/bat_ratings.csv and outputs/bowl_ratings.csv")