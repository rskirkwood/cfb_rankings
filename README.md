# College Football Unbiased Rankings

A data-driven FBS ranking model based purely on performance, not polls.

This project generates an unbiased, win/loss-based ranking system for FBS teams using game results from the CollegeFootballData API.

Instead of relying on polls or preseason expectations, this model evaluates teams only by who they beat and who they lost to, producing a clean, interpretable ranking.

### Features

Results-only model (no human or preseason bias)

Points earned for beating teams with more wins

Points deducted for losing to teams with more losses

Normalized scores to compare teams fairly

FBS-only filtering

Conference tagging

Mascot extraction using /teams/fbs

Transparent CSV output for analysis or integration

## How It Works
1. Data Sources

The model pulls data from three CollegeFootballData endpoints:

/records — team win/loss summaries

/games — weekly game results

/teams/fbs — metadata including mascots and conference affiliation

2. Scoring

Each team’s score is computed as:

+X points for beating a team with X wins

-Y points for losing to a team with Y losses

Example:
Beat an 8–2 team: +8
Lose to a 2–8 team: -8

3. Normalization

To prevent teams with more games being overrated, scores are normalized:

normalized_score = raw_score / games_played

4. FBS Filtering

Non-FBS teams are removed for cleaner comparisons.

5. Output

The script writes:

results_<year>_fbs_with_conf.csv

including team name, conference, wins, losses, raw score, normalized score, games processed, and mascot.

## Output Columns
Column	Description
team	School name
conference	Short conference abbreviation
wins, losses	Season record
raw_score	Opponent wins minus opponent losses
normalized_score	Score normalized by games played
games_processed	Number of games included
mascot	Team mascot name
## Usage
1. Install dependencies

pip install pandas requests

2. Set your API key

export CFB_API_KEY="your_real_key_here"

(You can obtain a free API key at https://collegefootballdata.com/
.)

3. Run the ranking system

python rank_cfb25.py

This generates:

results_<year>_fbs_with_conf.csv

## Test API Connectivity

python test_cfb_api.py

This verifies your API key and confirms access to required endpoints.

## File Overview

rank_cfb25.py - Main script: API fetch, scoring, CSV output
cfb_adj_rankings.py - Helper functions
test_cfb_api.py - Basic API key and endpoint test
results_*.csv - Generated CSV output files

## Example CSV Entry

team,conference,wins,losses,raw_score,normalized_score,games_processed,mascot
Georgia,SEC,10,0,82,8.2,10,Bulldogs

## Optional Integrations

This ranking system can be incorporated into dashboards, websites, and analytics projects.
A separate portfolio repository uses the generated CSV to display a live Top 25 ranking. This integration is optional and not part of this repository.

## Future Improvements

Strength of schedule refinement

Margin of Victory scoring

Drive- or play-level efficiency

Weekly ranking movement indicators

Visualization tools and dashboards

Comparison with SP+, FPI, and Elo-style systems