#!/usr/bin/env python3
# test_cfb_api_v2.py
import os
import requests
import json

BASE = "https://api.collegefootballdata.com"
API_KEY = os.getenv("CFB_API_KEY") or os.getenv("CFBD_API_KEY") or os.getenv("BEARER_TOKEN")
if not API_KEY:
    raise RuntimeError("Set CFB_API_KEY environment variable to your API key before running.")

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
YEAR = 2025

def safe_get(dct, *keys, default=None):
    """Return the first non-None key in dct from keys, else default."""
    if not isinstance(dct, dict):
        return default
    for k in keys:
        if k in dct and dct[k] is not None:
            return dct[k]
    return default

def fetch_records(year=YEAR):
    r = requests.get(f"{BASE}/records", headers=HEADERS, params={"year": year})
    r.raise_for_status()
    return r.json()

def fetch_games(year=YEAR, season_type="regular"):
    r = requests.get(f"{BASE}/games", headers=HEADERS, params={"year": year, "seasonType": season_type})
    r.raise_for_status()
    return r.json()

def pretty_print_sample_games(games_json, n=5):
    print(f"\n--- printing RAW JSON for first {n} games ---")
    for i, g in enumerate(games_json[:n]):
        print(f"\nGAME #{i+1} RAW:")
        print(json.dumps(g, indent=2))
    print("--- end raw sample ---\n")

def print_parsed_games(games_json, n=10):
    print(f"--- parsed view of first {min(n, len(games_json))} games ---")
    for i, g in enumerate(games_json[:n]):
        home = safe_get(g, "home_team", "homeTeam", "home", "home_school", "home_team_name", "home_display_name")
        away = safe_get(g, "away_team", "awayTeam", "away", "away_school", "away_team_name", "away_display_name")
        home_pts = safe_get(g, "home_points", "homePoints", "homeScore", "home_score", default=None)
        away_pts = safe_get(g, "away_points", "awayPoints", "awayScore", "away_score", default=None)
        winner = safe_get(g, "winner", "winningTeam", "winning_team", default=None)
        gid = safe_get(g, "id", "gameId", "game_id", default=f"idx_{i}")
        print(f"Game {gid}: {home or '?'} ({home_pts if home_pts is not None else '—'}) vs {away or '?'} ({away_pts if away_pts is not None else '—'})  winner: {winner or '—'}")
    print("--- end parsed sample ---")

def main():
    print("Fetching records...")
    recs = fetch_records(YEAR)
    print("Sample records (first 8):")
    for rec in recs[:8]:
        # print a compact view if possible
        team = safe_get(rec, "team", "school", "team_name", "name") or safe_get(rec, "id", "team_id")
        total = rec.get("total") if isinstance(rec.get("total"), dict) else rec
        wins = safe_get(total, "wins", "w", default=None)
        losses = safe_get(total, "losses", "l", default=None)
        print(f"{team}: {wins}-{losses}")

    print("\nFetching games...")
    games = fetch_games(YEAR, season_type="regular")
    print(f"Total games returned from endpoint: {len(games)}")

    # Print raw JSON for first few games to inspect keys
    pretty_print_sample_games(games, n=3)

    # Print parsed view with safe_get to show found fields
    print_parsed_games(games, n=12)

if __name__ == "__main__":
    main()
