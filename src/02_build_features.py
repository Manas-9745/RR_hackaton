"""Step 2: build one clean ball-level table with match-state features."""
import pandas as pd, numpy as np
from pathlib import Path

D, O = Path("data"), Path("outputs")
O.mkdir(exist_ok=True)

b = pd.read_csv(D/"ball_by_ball.csv", low_memory=False)
m = pd.read_csv(D/"matches.csv")
c = pd.read_csv(D/"competitions.csv")

# 1. keep normal innings only (3 and 4 are super overs)
b = b[b.innings.isin([1, 2])].copy()
b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

# 2. per-ball basics
# extras1_runs is a CODE (6 = no extra), not a run count. Extras runs are
# approximated and were checked against winning margins.
EXTRA_RUNS = {"wide": 1, "Overstep": 1, "byes": 2, "legbyes": 1, "other": 1, "Penalty": 5}
b["extras_tot"] = b["extras_type"].map(EXTRA_RUNS).fillna(0) + b["extras_2_runs"].fillna(0)
b["total_runs"] = b["runs"] + b["extras_tot"]
b["legal"] = (~b["extras_type"].isin(["wide", "Overstep"])).astype(int)
b["bat_ball_faced"] = (b["extras_type"] != "wide").astype(int)
b["bowler_wkt"] = (b["is_wicket"] &
    ~b["dismissal_type"].isin(["RunOut", "Retire Out", "Obstracting the field",
                               "Obstructing the field"])).astype(int)
b["is_wicket"] = b["is_wicket"].astype(int)

# 3. match state BEFORE each ball
g = b.groupby(["match_id", "innings"])
b["score_before"] = g["total_runs"].cumsum() - b["total_runs"]
b["wkts_before"] = g["is_wicket"].cumsum() - b["is_wicket"]
b["legal_before"] = g["legal"].cumsum() - b["legal"]
b["balls_left"] = 120 - b["legal_before"]

# 4. chase target and required rate
inn1 = b[b.innings == 1].groupby("match_id")["total_runs"].sum().rename("inn1_total")
b = b.merge(inn1, on="match_id", how="left")
b = b.merge(m[["match_id", "comp_id", "match_date", "venue_id", "stage", "knock_out",
               "revised_target", "second_inn_revised_overs", "win_team_id"]],
            on="match_id", how="left")
b = b.merge(c[["comp_id", "comp_name", "match_format", "season"]], on="comp_id", how="left")

b["target"] = np.where(b.innings == 2, b["revised_target"].fillna(b["inn1_total"] + 1), np.nan)
b["runs_needed"] = b["target"] - b["score_before"]
b["rrr"] = np.where(b.innings == 2, b["runs_needed"] / (b["balls_left"].clip(lower=1) / 6), np.nan)
b["crr"] = np.where(b.legal_before > 0, b["score_before"] / (b["legal_before"] / 6), np.nan)

# 5. phase, outcome, league flag
b["phase"] = pd.cut(b["overid"], [0, 6, 15, 20], labels=["powerplay", "middle", "death"])
b["bat_team_won"] = (b["batting_team_id"] == b["win_team_id"]).astype(float)
b.loc[b["win_team_id"].isna(), "bat_team_won"] = np.nan
b["is_ipl"] = (b["match_format"] == "T20").astype(int)

b.to_parquet(O/"balls.parquet", index=False)

# sanity report (aggregates only)
print("rows:", len(b), "| matches:", b.match_id.nunique())
print(b.groupby("match_format").size())
print(b[b.innings == 1].groupby(["match_format", "match_id"])["total_runs"].sum()
        .groupby("match_format").describe().round(1))
print(b.groupby("phase", observed=True).agg(rpb=("total_runs", "mean"),
      wkt_per_ball=("is_wicket", "mean")).round(3))
print("Saved outputs/balls.parquet")