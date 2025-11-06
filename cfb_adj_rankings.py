#!/usr/bin/env python3
"""
rank_cfb25_adjusted.py

- Robust weekly fetch from CollegeFootballData (FBS-only).
- Computes base normalized score (raw_points / games_played).
- Computes opponent-average of base normalized scores (optionally weighted by opponent conference).
- Final score = base_normalized + alpha * opponent_avg_weighted
- Outputs CSV with base_normalized, opponent_avg, final_score, rank, etc.

Defaults:
  - alpha = 1.0 (you can reduce to e.g. 0.5)
  - conference weights: Power Five = 1.0, Group of Five = 0.85, Independents = 0.9

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
from statistics import mean

# -------------------------
# Config
# -------------------------
BASE = "https://api.collegefootballdata.com"
API_KEY = os.getenv("CFB_API_KEY") or os.getenv("CFBD_API_KEY") or os.getenv("BEARER_TOKEN")
if not API_KEY:
    raise RuntimeError("Set the CFB_API_KEY environment variable before running.")

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Connection": "close"}
YEAR = 2025
MAX_WEEKS = 20

# conference abbreviation mapping (display)
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
    # extend as needed
}

# conference grouping -> weights (used to weight opponent contributions)
# Power Five conferences get weight 1.0, G5 slightly lower, Independents mid.
CONF_GROUP_WEIGHTS = {
    "PowerFive": 1.00,
    "GroupOfFive": 0.85,
    "Independent": 0.90,
    "Other": 0.80,
}

# map conference display/long names to group
CONF_TO_GROUP = {
    "Southeastern Conference": "PowerFive",
    "Big Ten": "PowerFive",
    "Atlantic Coast Conference": "PowerFive",
    "Big 12": "PowerFive",
    "Pac-12": "Other",
    "American Athletic Conference": "GroupOfFive",
    "Sun Belt": "GroupOfFive",
    "Mid-American Conference": "GroupOfFive",
    "Conference USA": "GroupOfFive",
    "Mountain West": "GroupOfFive",
    "FBS Independents": "Independent",
    "Independent": "Independent",
    # add more if needed
}

# alpha controls how strongly opponents' average affects final score
ALPHA = 0.5

# -------------------------
# Networking helpers
# -------------------------
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

# -------------------------
# Fetching functions
# -------------------------
def fetch_records(year=YEAR):
    r = session.get(f"{BASE}/records", headers=HEADERS, params={"year": year}, timeout=30)
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
        time.sleep(0.12)
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

# -------------------------
# Parsing & score functions
# -------------------------
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
    """
    Return:
      edges: list of dicts with keys winner, loser, home, away, home_points, away_points, game_id
      class_map: team -> classification (lowercased)
      conf_map: team -> conference (string as provided)
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

def compute_base_scores(records_map, edges):
    """
    Compute raw and base normalized scores (no head-to-head exclusion).
    raw: sum over games: winner += opponent_wins ; loser -= opponent_losses
    base_normalized = raw / games_processed (if games_processed > 0)
    Also returns opponents_map: team -> set/list of opponents (for opponent average)
    """
    recs = dict(records_map)  # shallow copy for defaults
    def ensure(t):
        if t not in recs:
            recs[t] = {"wins":0, "losses":0, "ties":0, "games":0}

    raw = defaultdict(float)
    games_count = defaultdict(int)
    opponents = defaultdict(list)

    for e in edges:
        w = e["winner"]; l = e["loser"]
        ensure(w); ensure(l)
        opp_wins_for_winner = recs[l]["wins"]
        opp_losses_for_loser = recs[w]["losses"]
        raw[w] += opp_wins_for_winner
        raw[l] -= opp_losses_for_loser
        games_count[w] += 1
        games_count[l] += 1
        opponents[w].append(l)
        opponents[l].append(w)

    base_normalized = {}
    for t, val in raw.items():
        gp = games_count.get(t, recs.get(t, {}).get("games", 0))
        base_normalized[t] = val / gp if gp > 0 else 0.0

    return raw, games_count, base_normalized, opponents

def abbrev_conf(conf_name):
    if not conf_name:
        return None
    if conf_name in CONF_ABBREV:
        return CONF_ABBREV[conf_name]
    if "american" in conf_name.lower():
        return "AAC"
    # fallback: return original
    return conf_name

def conf_weight_from_confname(conf_name):
    if not conf_name:
        return CONF_GROUP_WEIGHTS.get("Other", 0.8)
    # normalize by group mapping
    group = CONF_TO_GROUP.get(conf_name)
    if group:
        return CONF_GROUP_WEIGHTS.get(group, CONF_GROUP_WEIGHTS.get("Other", 0.8))
    # try commons
    lower = conf_name.lower()
    if "big" in lower and ("ten" in lower or "12" in lower or "pac" in lower or "sec" in lower):
        return CONF_GROUP_WEIGHTS.get("PowerFive", 1.0)
    if "american" in lower:
        return CONF_GROUP_WEIGHTS.get("GroupOfFive", 0.85)
    if "independent" in lower:
        return CONF_GROUP_WEIGHTS.get("Independent", 0.9)
    return CONF_GROUP_WEIGHTS.get("Other", 0.8)

def compute_opponent_adjustment(base_normalized, opponents_map, conf_map, alpha=ALPHA, use_conf_weight=True):
    """
    For each team, compute opponent_avg_weighted = mean( base_normalized[opp] * weight(opp_conf) )
    Then final_score = base_normalized[team] + alpha * opponent_avg_weighted
    Returns dicts opponent_avg and final_score.
    """
    opponent_avg = {}
    final_score = {}
    for team, opps in opponents_map.items():
        if not opps:
            opponent_avg[team] = 0.0
            final_score[team] = base_normalized.get(team, 0.0)
            continue
        vals = []
        for opp in opps:
            opp_base = base_normalized.get(opp, 0.0)
            # weight by opponent conference if requested
            conf = conf_map.get(opp)  # original conf string (not abbreviated)
            if use_conf_weight:
                w = conf_weight_from_confname(conf)
            else:
                w = 1.0
            vals.append(opp_base * w)
        # average (unweighted by count)
        opp_mean = mean(vals) if vals else 0.0
        opponent_avg[team] = opp_mean
        final_score[team] = base_normalized.get(team, 0.0) + alpha * opp_mean
    return opponent_avg, final_score

def compute_competition_ranks_from_series(series):
    """
    series: pandas Series or list of final scores (descending order not required)
    Return dict score->rank and a per-team rank series computed as competition ranking.
    Competition ranking: ties share same rank; next rank = previous rank + count(tied)
    Implementation: map unique scores descending to ranks.
    """
    vals = list(series)
    counts = Counter(vals)
    unique_scores = sorted(list(counts.keys()), reverse=True)
    rank_map = {}
    cumulative = 0
    for score in unique_scores:
        rank_map[score] = cumulative + 1
        cumulative += counts[score]
    return rank_map

# -------------------------
# main
# -------------------------
def main():
    print(f"Fetching {YEAR} records and games from {BASE} (weekly fetch)...")
    recs_json = fetch_records(YEAR)
    games_json = fetch_games_weekly(YEAR, season_type="regular", max_weeks=MAX_WEEKS)
    print(f"Records returned: {len(recs_json)}  — Games returned (combined weekly): {len(games_json)}")

    records_map = build_records_map(recs_json)
    edges, class_map, conf_map = build_completed_edges_and_maps(games_json)
    print(f"Completed games with scores: {len(edges)} (these are used for scoring)")

    # compute base raw & normalized and opponent lists
    raw_scores, games_count, base_normalized, opponents_map = compute_base_scores(records_map, edges)

    # compute opponent adjustment & final scores
    opponent_avg_map, final_scores = compute_opponent_adjustment(base_normalized, opponents_map, conf_map, alpha=ALPHA, use_conf_weight=True)

    # build DataFrame
    teams = sorted(final_scores.keys(), key=lambda t: final_scores[t], reverse=True)
    df = pd.DataFrame({
        "team": teams,
        "conference": [abbrev_conf(conf_map.get(t)) for t in teams],
        "base_normalized": [base_normalized.get(t, 0.0) for t in teams],
        "opponent_avg": [opponent_avg_map.get(t, 0.0) for t in teams],
        "final_score": [final_scores.get(t, 0.0) for t in teams],
        "raw_score": [raw_scores.get(t, 0.0) for t in teams],
        "games_processed": [games_count.get(t, 0) for t in teams],
        "wins": [records_map.get(t, {}).get("wins", 0) for t in teams],
        "losses": [records_map.get(t, {}).get("losses", 0) for t in teams]
    })

    # filter to FBS only via class_map
    df = df[df["team"].map(lambda t: class_map.get(t) == "fbs")].copy()

    # compute competition ranks based on final_score
    rank_map = compute_competition_ranks_from_series(df["final_score"].tolist())
    df["rank"] = df["final_score"].map(lambda s: rank_map.get(s))

    # sort by final_score desc, then rank, then team
    df = df.sort_values(by=["final_score", "team"], ascending=[False, True]).reset_index(drop=True)

    # save CSV with useful columns
    out_csv = f"results_{YEAR}_fbs_with_conf_adjusted.csv"
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\nSaved adjusted results to {out_csv}. Top 30 (FBS only):\n")

    # print top 30: rank. Team  Conf  final_score  base=...
    top = df.head(30).reset_index(drop=True)
    for _, row in top.iterrows():
        rank = int(row["rank"])
        conf_display = row["conference"] or "Independent/Unknown"
        print(f"{rank:2}. {row['team']:22}  {conf_display:12}  final={row['final_score']:.3f}  base={row['base_normalized']:.3f}  opp_avg={row['opponent_avg']:.3f}  (raw={row['raw_score']:.1f}, gp={int(row['games_processed'])})")

if __name__ == "__main__":
    main()
