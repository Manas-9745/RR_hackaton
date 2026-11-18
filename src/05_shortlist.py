"""Step 6: turn the ratings into an AUCTION-READY shortlist."""
import pandas as pd, numpy as np
from pathlib import Path

D, O, C = Path("data"), Path("outputs"), Path("config")
C.mkdir(exist_ok=True)
EXC = C / "exclusions.csv"
if not EXC.exists():
    EXC.write_text("player,reason\n")
REL = C / "released.csv"
if not REL.exists():
    REL.write_text("player\n")
exc = pd.read_csv(EXC)
exc_names = set(exc["player"].astype(str).str.strip().str.lower())
rel = pd.read_csv(REL)
rel_names = set(rel["player"].astype(str).str.strip().str.lower())
USE_RELEASED = len(rel_names) > 0

# ---------- player facts from the data pack ----------
pe = pd.read_csv(D / "playing_eleven.csv")
pe = pe[pe["is_played"].astype(str).str.lower() == "true"]
m = pd.read_csv(D / "matches.csv")[["match_id", "comp_id", "match_date"]]
comps = pd.read_csv(D / "competitions.csv")
teams = pd.read_csv(D / "teams.csv")
mp = pd.read_csv(D / "competition_team_player_mapping.csv")

x = (pe.merge(m, on="match_id").merge(comps[["comp_id", "comp_name", "match_format"]], on="comp_id")
       .merge(teams, on=["comp_id", "team_id"], how="left"))
x["match_date"] = pd.to_datetime(x["match_date"], errors="coerce", dayfirst=True)
x["ipl"] = x["comp_name"].str.startswith("IPL")
x["t20i"] = x["match_format"] == "T20I"

info = pd.DataFrame({"n_ipl": x.groupby("player_id")["ipl"].sum(),
                     "n_t20i": x.groupby("player_id")["t20i"].sum()})
nat = x[x.t20i].groupby("player_id")["team_name"].agg(lambda s: s.mode().iat[0])
info["country"] = nat
pak = set(x[x.t20i & (x.team_name == "Pakistan")]["player_id"])
last = x[x.ipl].sort_values("match_date").groupby("player_id").tail(1).set_index("player_id")
info["last_ipl_team"] = last["team_name"]
info["last_ipl_season"] = last["comp_name"].str.extract(r"(\d{4})")[0]
info["pakistan"] = info.index.isin(pak)
info["rr_2026"] = info["last_ipl_team"].eq("Rajasthan Royals") & info["last_ipl_season"].eq("2026")
info["no_t20i_in_pack"] = info["n_t20i"] == 0
pname = mp.groupby("player_id")["player_name"].agg(lambda s: s.mode().iat[0])
info["name_map"] = pname
info.to_csv(O / "players_info.csv")

def load(role):
    r = pd.read_csv(O / f"{'bat' if role == 'bat' else 'bowl'}_ratings.csv")
    r = r.rename(columns={"striker_id" if role == "bat" else "bowler_id": "player_id"})
    r = r.merge(info.reset_index(), on="player_id", how="left")
    r["country"] = r["country"].fillna("no T20I in data")
    low = r["player"].astype(str).str.strip().str.lower()
    under_contract_2026 = r["last_ipl_season"].eq("2026")
    retained_guess = under_contract_2026 & ~low.isin(rel_names) if USE_RELEASED else False
    r["excluded"] = r["pakistan"] | low.isin(exc_names) | retained_guess
    why = exc.set_index(exc["player"].astype(str).str.strip().str.lower())["reason"].to_dict()
    r["why"] = np.select([r["pakistan"], low.isin(exc_names), retained_guess],
                         ["Pakistan (not in IPL)", low.map(why), "IPL 2026 player not on released list"],
                         default=np.where((r["n_ipl"] > 0) & ~under_contract_2026,
                                          "CHECK: not in IPL 2026 (retired/unsold?)", ""))
    return r

show_cols = ["rank_full", "player", "score", "country", "last_ipl_team", "last_ipl_season",
             "n_ipl", "rr_2026", "matches", "why"]
for role in ["bat", "bowl"]:
    r = load(role).sort_values("score", ascending=False).reset_index(drop=True)
    r["rank_full"] = np.arange(1, len(r) + 1)
    title = "BATTERS" if role == "bat" else "BOWLERS"
    print(f"\n######## {title} ########")
    print("\nA) FULL-POOL top 15 (pure data ranking):")
    print(r[show_cols].head(15).to_string(index=False))
    b = r[~r.excluded].reset_index(drop=True)
    b["rank_buyable"] = np.arange(1, len(b) + 1)
    print("\nB) BUYABLE-POOL top 15 (Pakistan + your exclusions / released-list filter removed):")
    print(b[["rank_buyable"] + show_cols].head(15).to_string(index=False))
    u = b[b.no_t20i_in_pack].head(8)
    print("\nC) NO-T20I-IN-PACK WATCHLIST. WARNING: NOT a reliable 'uncapped' test (the pack\n"
          "   misses some T20Is and ignores Tests/ODIs). Verify caps and age on a public source.")
    print(u[["rank_buyable", "player", "score", "last_ipl_team", "n_ipl", "matches"]].to_string(index=False) if len(u) else "  none eligible")
    b.to_csv(O / f"final_{role}_candidates.csv", index=False)
print("\nSaved outputs/final_bat_candidates.csv, final_bowl_candidates.csv, players_info.csv")
print(f"Exclusions applied: {len(exc)} names | released-list mode: {USE_RELEASED} ({len(rel_names)} names)")