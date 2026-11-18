"""Step 7: pick the final 5 Indian + 5 overseas per role, then make report charts and tables.
Rule: pool = wide ratings (>=40 high-pressure balls, >=3 matches) minus Pakistan + exclusions.
OVERSEAS slots need >=80 pressure balls and >=5 matches; INDIAN slots use the widest pool."""
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = Path("outputs"); F = O / "figures"; F.mkdir(exist_ok=True, parents=True)
N_IND, N_OVS = 5, 5

def pick(role):
    w = pd.read_csv(O / f"wide_{role}_candidates.csv")
    is_ind = (w["country"] == "India") | ((w["country"] == "no T20I in data") & (w["n_ipl"] > 0))
    w["group"] = np.where(is_ind, "Indian", "Overseas")
    w["tier"] = np.where((w["balls"] >= 80) & (w["matches"] >= 5), "solid", "thin")
    ovs = w[(w.group == "Overseas") & (w.tier == "solid")].sort_values("score", ascending=False).head(N_OVS)
    ind = w[w.group == "Indian"].sort_values("score", ascending=False).head(N_IND)
    f = pd.concat([ovs, ind]).sort_values("score", ascending=False).reset_index(drop=True)
    f["final_rank"] = np.arange(1, len(f) + 1)
    return w, f

res = {}
for role, title in [("bat", "BATTERS"), ("bowl", "BOWLERS")]:
    w, f = pick(role)
    res[role] = (w, f)
    cols = ["final_rank", "player", "group", "tier", "press", "normal", "delta", "cons", "matches", "balls",
            "country", "last_ipl_team", "last_ipl_season"] + (["SR", "dot%", "bdry%"] if role == "bat" else ["econ", "dot%", "wkts"])
    print(f"\n======== FINAL 5 INDIAN + 5 OVERSEAS: {title} (ordered by model score) ========")
    print(f[cols].round(3).to_string(index=False))
    f[cols].round(3).to_csv(O / f"final10_{role}.csv", index=False)
    print("\nNext best Indian backups:", ", ".join(w[(w.group == 'Indian') & ~w.player.isin(f.player)].sort_values('score', ascending=False).player.head(4)))
    print("Next best overseas backups:", ", ".join(w[(w.group == 'Overseas') & (w.tier == 'solid') & ~w.player.isin(f.player)].sort_values('score', ascending=False).player.head(4)))

# ---------------- charts ----------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
PINK, BLUE, GREY = "#d4146f", "#254aa5", "#9aa5b1"

# Fig 1: does the pressure index mean something?
p = pd.read_parquet(O / "pressure.parquet", columns=["match_id", "innings", "innings_ball_num", "wp_before",
                    "leverage", "pressure_tier", "rrr"]).sort_values(["match_id", "innings", "innings_ball_num"])
p["swing"] = (p.groupby(["match_id", "innings"])["wp_before"].shift(-1) - p["wp_before"]).abs()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sw = p.groupby(["innings", "pressure_tier"], observed=True)["swing"].mean().unstack(0)
sw.plot.bar(ax=ax[0], color=[GREY, PINK], rot=0)
ax[0].set_title("Real win-probability swing per ball\n(high-pressure balls swing the match more)")
ax[0].set_ylabel("average |change in win probability|"); ax[0].set_xlabel("pressure tier"); ax[0].legend(title="innings")
c = p[p.innings == 2].copy()
c["b"] = pd.cut(c["rrr"], [-1, 6, 8, 10, 12, 15, 100], labels=["<6", "6-8", "8-10", "10-12", "12-15", "15+"])
c.groupby("b", observed=True)["leverage"].mean().plot.bar(ax=ax[1], color=BLUE, rot=0)
ax[1].set_title("Pressure (leverage) in a chase\nby required run rate"); ax[1].set_xlabel("required run rate"); ax[1].set_ylabel("average leverage")
fig.tight_layout(); fig.savefig(F / "fig1_pressure_validation.png", dpi=170); plt.close(fig)

# Fig 2/3: normal vs pressure performance
for role, name, fn in [("bat", "Batters", "fig2_batters_normal_vs_pressure.png"), ("bowl", "Bowlers", "fig3_bowlers_normal_vs_pressure.png")]:
    w, f = res[role]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(w["normal"], w["press"], s=14, color=GREY, alpha=.6, label="other players")
    ax.scatter(f["normal"], f["press"], s=48, color=PINK, label="final 10")
    for _, r in f.iterrows():
        ax.annotate(r["player"], (r["normal"], r["press"]), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    lim = [min(w["normal"].min(), w["press"].min()), max(w["normal"].max(), w["press"].max())]
    ax.plot(lim, lim, color=BLUE, lw=1, ls="--", label="same as usual level")
    ax.set_xlabel("rating on ordinary balls (above expected)"); ax.set_ylabel("rating on high-pressure balls (above expected)")
    ax.set_title(f"{name}: usual level vs under pressure\n(above the dashed line = better under pressure)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(F / fn, dpi=170); plt.close(fig)

# Fig 4: final scores
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a, role, name in [(ax[0], "bat", "Batters"), (ax[1], "bowl", "Bowlers")]:
    f = res[role][1].iloc[::-1]
    a.barh(f["player"], f["score"], color=np.where(f["group"] == "Indian", PINK, BLUE))
    a.set_title(f"{name}: final model score"); a.set_xlabel("score (pink = Indian, blue = overseas)")
fig.tight_layout(); fig.savefig(F / "fig4_final_scores.png", dpi=170); plt.close(fig)
print("\nSaved outputs/final10_bat.csv, final10_bowl.csv and 4 charts in outputs/figures/")