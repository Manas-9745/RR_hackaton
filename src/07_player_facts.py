"""Step 8: plain-fact profile for each of your final 20, to write the 'how this player
strengthens RR' lines. These are FACTS from the data; the role description is YOUR call."""
import pandas as pd, numpy as np
from pathlib import Path

O = Path("outputs")
HP = 0.80   # same 'high pressure' cut-off as step 4
b = pd.read_parquet(O / "pressure.parquet")
b["high"] = b["lev_pct"] > HP
b["bat_out"] = ((b["is_wicket"] == 1) & (b["dismissed_player_id"] == b["striker_id"])).astype(int)

def share(s):  # % split of high-pressure balls by phase
    v = s.value_counts(normalize=True) * 100
    return {k: round(v.get(k, 0)) for k in ["powerplay", "middle", "death"]}

rows = []
for role, f in [("bat", pd.read_csv(O / "final10_bat.csv")), ("bowl", pd.read_csv(O / "final10_bowl.csv"))]:
    key = "striker_name" if role == "bat" else "bowler_name"
    for _, r in f.iterrows():
        d = b[b[key] == r["player"]]
        h = d[d.high]
        ph = share(h["phase"].astype(str))
        row = {"role": role, "rank": r["final_rank"], "player": r["player"], "group": r["group"], "tier": r["tier"],
               "pressure_deliveries": len(h), "pp%": ph["powerplay"], "mid%": ph["middle"], "death%": ph["death"],
               "in_chase%": round(100 * (h["innings"] == 2).mean())}
        if role == "bat":
            bf = d[d.bat_ball_faced == 1]
            for t, lab in [("Pace", "SR_vs_pace"), ("Spin", "SR_vs_spin")]:
                x = bf[bf.pace_or_spin == t]
                row[lab] = round(100 * x["runs"].sum() / len(x)) if len(x) >= 20 else np.nan
            row["batting_balls_all"] = len(bf)
        else:
            row["type"] = d["pace_or_spin"].mode().iat[0] if len(d) else ""
            row["wkts_in_pressure"] = int(h["bowler_wkt"].sum())
            row["death_overs_bowled"] = round(((d.phase == "death") & (d.legal == 1)).sum() / 6, 1)
        rows.append(row)

out = pd.DataFrame(rows)
for role in ["bat", "bowl"]:
    t = out[out.role == role].dropna(axis=1, how="all").drop(columns="role")
    print(f"\n=== {'BATTERS' if role == 'bat' else 'BOWLERS'}: where and how they face pressure ===")
    print(t.to_string(index=False))
out.to_csv(O / "player_facts.csv", index=False)
print("\nSaved outputs/player_facts.csv")