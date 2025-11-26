#!/usr/bin/env python3
"""
rank_cfb25.py

FBS-only rankings using CFBD with:
 - competition ranking (ties share same rank; next rank skips)
 - conference abbreviations
 - mascot lookups

Scoring:
 - Winner gets opponent_wins * weight_vs_opp
 - Loser loses opponent_losses (full penalty, no weighting)

Weights:
 - P4/Power FBS opponents (Big Ten, SEC, Big 12, ACC, Independents): 1.0
 - G6 FBS opponents (AAC, MW, MAC, Sun Belt, C-USA, Pac-12): 0.75 for wins
 - FCS opponents: 0.0 for wins (no positive points), full penalty for losses

Normalization:
 - normalized_score = raw_score / fbs_games_processed
 - FBS games are those where opponent_class != 'fcs' for that team

Includes debug printing for a single team (BYU by default).
"""
import os
import sys
import time
from collections import defaultdict, Counter

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

BASE = "https://api.collegefootballdata.com"
API_KEY = (
    os.getenv("CFB_API_KEY")
    or os.getenv("CFBD_API_KEY")
    or os.getenv("BEARER_TOKEN")
)
if not API_KEY:
    raise RuntimeError("Set the CFB_API_KEY environment variable before running.")

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Connection": "close"}
YEAR = 2025
MAX_WEEKS = 20

# team to debug in compute_scores
DEBUG_TEAM = None  # e.g., "BYU" or None to disable

# Conference display abbreviations
CONF_ABBREV = {
    "American Athletic Conference": "AAC",
    "Atlantic Coast Conference": "ACC",
    "Big Ten": "Big Ten",
    "Big 12": "Big 12",
    "Southeastern Conference": "SEC",
    "Pac-12": "Pac-12",
    "Mountain West": "MW",
    "Sun Belt": "Sun Belt",
    "Mid-American": "MAC",
    "Conference USA": "C-USA",
    "FBS Independents": "Independent",
    # short forms that may already appear
    "AAC": "AAC",
    "MW": "MW",
    "C-USA": "C-USA",
    "MAC": "MAC",
    "Independent": "Independent",
}

# Group-of-6 (“G6”) detection for FBS conferences
G6_TOKENS = {
    "american athletic",  # AAC
    "aac",
    "mountain west",                 # MW
    "mw",
    "sun belt",
    "mid-american",       # MAC
    "mac",
    "conference usa",                # C-USA
    "c-usa",
    "pac-12",                        # Pac-12 (now small, treated as G6)
    "pac 12",
}

G6_WIN_MULTIPLIER = 0.75   # wins vs G6 give 75% of normal points
FCS_WIN_MULTIPLIER = 0.0   # wins vs FCS give 0 positive points

def make_session(
    retries: int = 5,
    backoff_factor: float = 0.5,
    status_forcelist=(429, 500, 502, 503, 504),
):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session = make_session()

# --- API fetch helpers ------------------------------------------------------

def fetch_records(year: int = YEAR):
    url = f"{BASE}/records"
    r = session.get(url, headers=HEADERS, params={"year": year}, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_games_weekly(year: int = YEAR, season_type: str = "regular", max_weeks: int = MAX_WEEKS):
    all_games = []
    for wk in range(1, max_weeks + 1):
        params = {"year": year, "week": wk, "seasonType": season_type}
        url = f"{BASE}/games"
        attempt = 0
        while attempt < 5:
            attempt += 1
            try:
                r = session.get(url, headers=HEADERS, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and data:
                    all_games.extend(data)
                break
            except Exception as exc:
                wait = 1.0 * attempt
                print(f"Warning: fetch week {wk} attempt {attempt} failed: {exc}. Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
        time.sleep(0.15)
    # fallback full fetch if weekly returned too few games
    if len(all_games) < 1000:
        try:
            r = session.get(
                f"{BASE}/games",
                headers=HEADERS,
                params={"year": year, "seasonType": season_type},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and len(data) > len(all_games):
                all_games = data
        except Exception as e:
            print("Fallback full fetch failed (continuing with weekly data):", e, file=sys.stderr)
    # dedupe by id or composite key
    uniq = {}
    for g in all_games:
        gid = g.get("id")
        if gid is None:
            gid = f"{g.get('homeTeam')}|{g.get('awayTeam')}|{g.get('startDate')}"
        if gid not in uniq:
            uniq[gid] = g
    return list(uniq.values())

def fetch_teams_fbs(year: int = YEAR):
    """Fetch FBS teams metadata (includes mascots) for the given year."""
    url = f"{BASE}/teams/fbs"
    r = session.get(url, headers=HEADERS, params={"year": year}, timeout=30)
    r.raise_for_status()
    return r.json()

# --- Mapping builders -------------------------------------------------------

def build_mascot_map(teams_json):
    """Map team name -> mascot."""
    mascot_map = {}
    for t in teams_json:
        school = t.get("school") or t.get("team") or t.get("name")
        mascot = t.get("mascot")
        if school and mascot:
            mascot_map[school] = mascot
    return mascot_map

def build_records_map(records_json):
    """Map team -> dict(wins, losses, ties, games)."""
    rec_map = {}
    for rec in records_json:
        team = rec.get("team") or rec.get("school") or rec.get("name")
        total = rec.get("total") if isinstance(rec.get("total"), dict) else rec
        wins = int(total.get("wins", 0))
        losses = int(total.get("losses", 0))
        ties = int(total.get("ties", 0) or 0)
        rec_map[team] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "games": wins + losses + ties,
        }
    return rec_map

def build_completed_edges_and_maps(games_json):
    """
    Build:
      edges: list of dicts(winner, loser, home, away, home_points, away_points, game_id)
      class_map: team -> 'fbs'/'fcs'/...
      conf_map: team -> conference string
    """
    edges = []
    class_map = {}
    conf_map = {}
    for g in games_json:
        home_class = g.get("homeClassification")
        away_class = g.get("awayClassification")
        home_conf = g.get("homeConference")
        away_conf = g.get("awayConference")
        home = g.get("homeTeam")
        away = g.get("awayTeam")
        if home and home_class:
            class_map[home] = home_class.lower()
        if away and away_class:
            class_map[away] = away_class.lower()
        if home and home_conf:
            conf_map[home] = home_conf
        if away and away_conf:
            conf_map[away] = away_conf

        home_points = g.get("homePoints")
        away_points = g.get("awayPoints")
        completed = g.get("completed", None)
        if completed is not True:
            continue
        if home_points is None or away_points is None:
            continue
        try:
            hp = float(home_points)
            ap = float(away_points)
        except Exception:
            continue
        if hp == ap:
            continue  # ignore ties
        if hp > ap:
            winner, loser = home, away
        else:
            winner, loser = away, home
        edges.append(
            {
                "game_id": g.get("id"),
                "winner": winner,
                "loser": loser,
                "home": home,
                "away": away,
                "home_points": hp,
                "away_points": ap,
            }
        )
    return edges, class_map, conf_map

# --- Scoring helpers --------------------------------------------------------

def is_g6_conference(conf_name: str | None) -> bool:
    """Return True if conference should be treated as Group of 6."""
    if not conf_name:
        return False
    c = conf_name.lower()
    for tok in G6_TOKENS:
        if tok in c:
            return True
    return False

def win_weight_vs_opponent(loser: str, class_map, conf_map) -> float:
    """
    Determine multiplier for the WINNER's points based on the LOSER (opponent).
    - If opponent is FCS -> FCS_WIN_MULTIPLIER
    - Else if opponent is FBS G6 -> G6_WIN_MULTIPLIER
    - Else -> 1.0
    """
    opp_class = class_map.get(loser)
    opp_conf = conf_map.get(loser)

    if opp_class == "fcs":
        return FCS_WIN_MULTIPLIER
    if is_g6_conference(opp_conf):
        return G6_WIN_MULTIPLIER
    return 1.0  # Power conferences / Independents

def compute_scores(records_map, edges, class_map, conf_map, debug_team: str | None = None):
    """
    Scoring logic (no head-to-head exclusion):
      - Winner gets opponent_wins * win_weight_vs_opponent
      - Loser loses opponent_losses (full), no weighting
      - FCS games:
          * win vs FCS -> 0 positive points
          * loss vs FCS -> full negative points
      - Denominator (games_count) for each team t counts only games
        where opponent_class != 'fcs'  (i.e., per FBS game).
    """
    recs = dict(records_map)

    def ensure(t):
        if t not in recs:
            recs[t] = {"wins": 0, "losses": 0, "ties": 0, "games": 0}

    raw = defaultdict(float)
    games_count = defaultdict(int)

    if debug_team:
        print(f"\n=== DEBUG for team: {debug_team} ===\n")

    for e in edges:
        w = e["winner"]
        l = e["loser"]
        ensure(w)
        ensure(l)

        opp_wins_for_winner = recs[l]["wins"]
        opp_losses_for_loser = recs[w]["losses"]

        # Winner's weighted gain
        w_mult = win_weight_vs_opponent(l, class_map, conf_map)
        delta_w = w_mult * opp_wins_for_winner
        raw[w] += delta_w

        # Loser's full penalty
        delta_l = -opp_losses_for_loser
        raw[l] += delta_l

        # Denominator: only count games where opponent is not FCS
        opp_class_for_winner = class_map.get(l)
        opp_class_for_loser = class_map.get(w)
        incr_w = 0
        incr_l = 0
        if opp_class_for_winner != "fcs":
            games_count[w] += 1
            incr_w = 1
        if opp_class_for_loser != "fcs":
            games_count[l] += 1
            incr_l = 1

        # --- DEBUG PRINTS for specific team --------------------------------
        if debug_team and (w == debug_team or l == debug_team):
            print(f"Game {e['game_id']}: {e['home']} {e['home_points']} - {e['away']} {e['away_points']}")
            print(f"  Winner: {w}, Loser: {l}")
            # winner perspective
            print(f"  {w} (winner) vs {l}:")
            print(f"    {l} record: {recs[l]['wins']}-{recs[l]['losses']}")
            print(f"    {l} class: {class_map.get(l)}, conf: {conf_map.get(l)}")
            print(f"    win_weight = {w_mult:.3f}")
            print(f"    opp_wins_for_winner = {opp_wins_for_winner}, delta_w = +{delta_w:.3f}")
            # loser perspective
            print(f"  {l} (loser) vs {w}:")
            print(f"    {w} record: {recs[w]['wins']}-{recs[w]['losses']}")
            print(f"    {w} class: {class_map.get(w)}, conf: {conf_map.get(w)}")
            print(f"    opp_losses_for_loser = {opp_losses_for_loser}, delta_l = {delta_l:.3f}")
            print(f"  Denominator increments: {w}: +{incr_w}, {l}: +{incr_l}")
            print(f"  Current raw[{w}] = {raw[w]:.3f}, raw[{l}] = {raw[l]:.3f}")
            print(f"  Current games_count[{w}] = {games_count[w]}, games_count[{l}] = {games_count[l]}")
            print("-" * 70)

    normalized = {}
    for t, val in raw.items():
        gp = games_count.get(t, 0)
        if gp == 0:
            gp = recs.get(t, {}).get("games", 0)
        normalized[t] = val / gp if gp > 0 else 0.0

    if debug_team:
        print(f"\n=== FINAL DEBUG SUMMARY for {debug_team} ===")
        print(f"  raw[{debug_team}] = {raw.get(debug_team, 0.0):.3f}")
        print(f"  fbs_games_processed[{debug_team}] = {games_count.get(debug_team, 0)}")
        rec = recs.get(debug_team, {"wins": 0, "losses": 0})
        print(f"  record (from /records): {rec['wins']}-{rec['losses']}")
        print(f"  normalized_score[{debug_team}] = {normalized.get(debug_team, 0.0):.3f}")
        print("=====================================================\n")

    return raw, games_count, normalized

# --- DataFrame helpers ------------------------------------------------------

def apply_fbs_filter_and_attach_conf(df: pd.DataFrame, class_map, conf_map):
    df = df.copy()
    df["classification"] = df["team"].map(lambda t: class_map.get(t))
    df = df[df["classification"] == "fbs"].copy()
    df["conference"] = df["team"].map(lambda t: conf_map.get(t))
    df = df.drop(columns=["classification"], errors="ignore")
    return df

def abbrev_conf(conf_name: str | None):
    if not conf_name:
        return None
    if conf_name in CONF_ABBREV:
        return CONF_ABBREV[conf_name]
    if "american" in conf_name.lower():
        return "AAC"
    return conf_name

def compute_competition_ranks(sorted_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame sorted by normalized_score descending, compute competition ranks:
    rank(score) = 1 + number of teams with strictly greater score.
    """
    scores = sorted_df["normalized_score"].tolist()
    counts = Counter(scores)
    unique_scores = sorted(list(counts.keys()), reverse=True)
    rank_map = {}
    cumulative = 0
    for score in unique_scores:
        rank_map[score] = cumulative + 1
        cumulative += counts[score]
    out = sorted_df.copy()
    out["rank"] = out["normalized_score"].map(lambda s: rank_map.get(s, None))
    return out

# --- main -------------------------------------------------------------------

def main():
    print(f"Fetching {YEAR} records and games from {BASE} (weekly fetch)...")
    recs_json = fetch_records(YEAR)
    games_json = fetch_games_weekly(YEAR, season_type="regular", max_weeks=MAX_WEEKS)
    print(f"Records returned: {len(recs_json)}  — Games returned (combined weekly): {len(games_json)}")

    teams_json = fetch_teams_fbs(YEAR)
    mascot_map = build_mascot_map(teams_json)
    records_map = build_records_map(recs_json)
    edges, class_map, conf_map = build_completed_edges_and_maps(games_json)
    print(f"Completed games with scores: {len(edges)} (these are used for scoring)")

    raw_scores, games_count, normalized = compute_scores(
        records_map, edges, class_map, conf_map, debug_team=DEBUG_TEAM
    )

    teams = list(normalized.keys())
    df = pd.DataFrame(
        {
            "team": teams,
            "normalized_score": [normalized[t] for t in teams],
            "raw_score": [raw_scores[t] for t in teams],
            "fbs_games_processed": [games_count.get(t, 0) for t in teams],
            "wins": [records_map.get(t, {}).get("wins", 0) for t in teams],
            "losses": [records_map.get(t, {}).get("losses", 0) for t in teams],
        }
    )
    df = df.sort_values(by="normalized_score", ascending=False).reset_index(drop=True)
    df = apply_fbs_filter_and_attach_conf(df, class_map, conf_map)

    # attach mascot
    df["mascot"] = df["team"].map(lambda t: mascot_map.get(t))

    # abbreviate conferences
    df["conference"] = df["conference"].map(lambda c: abbrev_conf(c) if c else c)

    out_csv = f"results_{YEAR}_fbs_with_conf.csv"
    df.to_csv(out_csv, index=False)

    # compute competition ranks
    df_ranked = compute_competition_ranks(
        df.sort_values(by="normalized_score", ascending=False).reset_index(drop=True)
    )

    print(f"\nSaved filtered results to {out_csv}. Top 30 (FBS only):\n")
    top = df_ranked.head(30).reset_index(drop=True)
    for _, row in top.iterrows():
        rank = int(row["rank"])
        conf_display = row["conference"] or "Independent/Unknown"
        print(
            f"{rank:2}. {row['team']:22}  {conf_display:12}  "
            f"score={row['normalized_score']:.3f}  "
            f"(raw={row['raw_score']:.1f}, wins={row['wins']}, losses={row['losses']})"
        )

if __name__ == "__main__":
    main()
