#!/usr/bin/env python3
"""
rank_cfb25.py

Robust fetch + FBS-only rankings with:
 - competition ranking (ties share same rank; next rank skips accordingly)
 - conference printed without parentheses and with abbreviation mapping
 - saves CSV results_{YEAR}_fbs_with_conf.csv

Requirements:
  pip install requests pandas
Set env var CFB_API_KEY before running.
"""
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict, Counter
import pandas as pd
import time
import sys

BASE = "https://api.collegefootballdata.com"
API_KEY = os.getenv("CFB_API_KEY") or os.getenv("CFBD_API_KEY") or os.getenv("BEARER_TOKEN")
if not API_KEY:
    raise RuntimeError("Set the CFB_API_KEY environment variable before running.")

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Connection": "close"}
YEAR = 2025
MAX_WEEKS = 20

# Conference abbreviation mapping (add any others you want)
CONF_ABBREV = {
    "American Athletic Conference": "AAC",
    "Atlantic Coast Conference": "ACC",
    "Big Ten": "Big Ten",
    "Big 12": "Big 12",
    "Southeastern Conference": "SEC",
    "Pac-12": "Pac-12",
    "Mountain West": "MW",
    "Sun Belt": "Sun Belt",
    "Mid-American Conference": "MAC",
    "Conference USA": "C-USA",
    "FBS Independents": "Independent",
    # Add more mappings as desired
}

def make_session(retries=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session = make_session()

def fetch_records(year=YEAR):
    url = f"{BASE}/records"
    r = session.get(url, headers=HEADERS, params={"year": year}, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_games_weekly(year=YEAR, season_type="regular", max_weeks=MAX_WEEKS):
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
            r = session.get(f"{BASE}/games", headers=HEADERS, params={"year": year, "seasonType": season_type}, timeout=60)
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

def fetch_teams_fbs(year=YEAR):
    """Fetch FBS teams metadata (includes mascots) for the given year."""
    url = f"{BASE}/teams/fbs"
    r = session.get(url, headers=HEADERS, params={"year": year}, timeout=30)
    r.raise_for_status()
    return r.json()

def build_mascot_map(teams_json):
    """
    Build a mapping of team name -> mascot.

    We use the 'school' field, which should match the team names used
    in the records/games APIs (e.g., 'Georgia', 'Ohio State', etc.).
    """
    mascot_map = {}
    for t in teams_json:
        school = t.get("school") or t.get("team") or t.get("name")
        mascot = t.get("mascot")
        if school and mascot:
            mascot_map[school] = mascot
    return mascot_map


def build_records_map(records_json):
    rec_map = {}
    for rec in records_json:
        team = rec.get("team") or rec.get("school") or rec.get("name")
        total = rec.get("total") if isinstance(rec.get("total"), dict) else rec
        wins = int(total.get("wins", 0))
        losses = int(total.get("losses", 0))
        ties = int(total.get("ties", 0) or 0)
        rec_map[team] = {"wins": wins, "losses": losses, "ties": ties, "games": wins + losses + ties}
    return rec_map

def build_completed_edges_and_maps(games_json):
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
            hp = float(home_points); ap = float(away_points)
        except Exception:
            continue
        if hp == ap:
            continue
        if hp > ap:
            winner, loser = home, away
        else:
            winner, loser = away, home
        edges.append({
            "game_id": g.get("id"),
            "winner": winner,
            "loser": loser,
            "home": home,
            "away": away,
            "home_points": hp,
            "away_points": ap
        })
    return edges, class_map, conf_map

def compute_scores(records_map, edges, exclude_head_to_head=True):
    recs = dict(records_map)
    def ensure(t):
        if t not in recs:
            recs[t] = {"wins":0, "losses":0, "ties":0, "games":0}

    raw = defaultdict(float)
    games_count = defaultdict(int)
    for e in edges:
        w = e["winner"]; l = e["loser"]
        ensure(w); ensure(l)
        opp_wins_for_winner = recs[l]["wins"]
        opp_losses_for_loser = recs[w]["losses"]
        if exclude_head_to_head:
            adj_loser_wins = recs[l]["wins"]
            adj_loser_losses = max(0, recs[l]["losses"] - 1)
            adj_winner_wins = max(0, recs[w]["wins"] - 1)
            adj_winner_losses = recs[w]["losses"]
            opp_wins_for_winner = adj_loser_wins
            opp_losses_for_loser = adj_winner_losses
        raw[w] += opp_wins_for_winner
        raw[l] -= opp_losses_for_loser
        games_count[w] += 1
        games_count[l] += 1

    normalized = {}
    for t, val in raw.items():
        gp = games_count.get(t, recs.get(t, {}).get("games", 0))
        normalized[t] = val / gp if gp > 0 else 0.0

    return raw, games_count, normalized

def apply_fbs_filter_and_attach_conf(df, class_map, conf_map):
    df = df.copy()
    df["classification"] = df["team"].map(lambda t: class_map.get(t))
    df = df[df["classification"] == "fbs"].copy()
    df["conference"] = df["team"].map(lambda t: conf_map.get(t))
    df = df.drop(columns=["classification"], errors="ignore")
    return df

def abbrev_conf(conf_name):
    if not conf_name:
        return None
    # prefer exact mapping first
    if conf_name in CONF_ABBREV:
        return CONF_ABBREV[conf_name]
    # common case: sometimes API returns 'American Athletic Conference' or 'American'
    if "american" in conf_name.lower():
        return "AAC"
    # fallback: return original
    return conf_name

def compute_competition_ranks(sorted_df):
    """
    Given a DataFrame sorted by normalized_score descending, compute competition ranks:
    rank(score) = 1 + number of teams with strictly greater score.
    We'll produce a new 'rank' column.
    """
    scores = sorted_df["normalized_score"].tolist()
    counts = Counter(scores)  # counts per exact score
    unique_scores = sorted(list(counts.keys()), reverse=True)
    rank_map = {}
    cumulative = 0
    for score in unique_scores:
        rank_map[score] = cumulative + 1
        cumulative += counts[score]
    # map into dataframe
    sorted_df = sorted_df.copy()
    sorted_df["rank"] = sorted_df["normalized_score"].map(lambda s: rank_map.get(s, None))
    return sorted_df

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

    raw_scores, games_count, normalized = compute_scores(records_map, edges, exclude_head_to_head=True)

    teams = list(normalized.keys())
    df = pd.DataFrame({
        "team": teams,
        "normalized_score": [normalized[t] for t in teams],
        "raw_score": [raw_scores[t] for t in teams],
        "games_processed": [games_count.get(t, 0) for t in teams],
        "wins": [records_map.get(t, {}).get("wins", 0) for t in teams],
        "losses": [records_map.get(t, {}).get("losses", 0) for t in teams]
    })
    df = df.sort_values(by="normalized_score", ascending=False).reset_index(drop=True)
    df = apply_fbs_filter_and_attach_conf(df, class_map, conf_map)

    # attach mascot from teams metadata
    df["mascot"] = df["team"].map(lambda t: mascot_map.get(t))

    # abbreviate conferences for display/CSV
    df["conference"] = df["conference"].map(lambda c: abbrev_conf(c))
    out_csv = f"results_{YEAR}_fbs_with_conf.csv"
    df.to_csv(out_csv, index=False)

    # compute competition ranks
    df_ranked = compute_competition_ranks(df.sort_values(by="normalized_score", ascending=False).reset_index(drop=True))

    print(f"\nSaved filtered results to {out_csv}. Top 30 (FBS only):\n")
    top = df_ranked.head(30).reset_index(drop=True)
    for _, row in top.iterrows():
        rank = int(row["rank"])
        conf_display = row["conference"] or "Independent/Unknown"
        # no parentheses around conference now
        print(f"{rank:2}. {row['team']:22}  {conf_display:12}  score={row['normalized_score']:.3f}  (raw={row['raw_score']:.1f}, wins={row['wins']}, losses={row['losses']})")


if __name__ == "__main__":
    main()
