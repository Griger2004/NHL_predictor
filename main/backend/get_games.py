import requests
import json
from unidecode import unidecode  # Import the unidecode library
from datetime import datetime, timedelta


# Get today's date in the required format (YYYY-MM-DD)
today = datetime.today().strftime('%Y-%m-%d')
yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

# URL for the NHL schedule
url = f"https://api-web.nhle.com/v1/schedule/{today}"
print(url)

# URL for the team stats
url_stats = "https://api.nhle.com/stats/rest/en/team/summary?sort=shotsForPerGame&cayenneExp=seasonId=20242025%20and%20gameTypeId=2"

# Fetch the data from the URL
response = requests.get(url)
data = response.json()

response_stats = requests.get(url_stats)
data_stats = response_stats.json()

# Load the name-to-code mapping from your JSON file (name_codes.json)
with open('name_codes.json') as f:
    name_codes = json.load(f)

# Initialize an empty list to store the game schedule
games = []

# Previous day teams that played.

def check_prev_day():
    url = f"https://api-web.nhle.com/v1/schedule/{yesterday}"

    prev_day_teams = []

    # Fetch the data from the URL
    response = requests.get(url)
    data = response.json()

    for week in data['gameWeek']:
        if week['date'] == "2025-03-12":
            for game in week['games']:

                #get team abbriviations
                home_abrv = game['homeTeam']['abbrev']
                away_abrv = game['awayTeam']['abbrev']
                
                prev_day_teams.append(home_abrv)
                prev_day_teams.append(away_abrv)

    return prev_day_teams

all_prev_day_teams = check_prev_day()

# Loop through the 'gameWeek' in the data
for week in data['gameWeek']:
    if week['date'] == "2025-03-15":
        for game in week['games']:

            # Combine 'placeName' and 'commonName' to get the full team name
            home_team = game['homeTeam']['placeName']['default'] + " " + game['homeTeam']['commonName']['default']
            away_team = game['awayTeam']['placeName']['default'] + " " + game['awayTeam']['commonName']['default']

            #get team abbriviations
            home_abrv = game['homeTeam']['abbrev']
            away_abrv = game['awayTeam']['abbrev']
            
            # Remove accents using unidecode
            home_team = unidecode(home_team)
            away_team = unidecode(away_team)

            home_code = name_codes['team_codes'][0].get(home_team)
            away_code = name_codes['team_codes'][0].get(away_team)

            # Get team stats

            both_team_stats = 0
            for team in data_stats["data"]:
                if both_team_stats == 2:
                    break
                if team["teamFullName"] == home_team:
                    both_team_stats += 1
                    home_goals_for = team["goalsForPerGame"]
                    home_goals_against = team["goalsAgainstPerGame"]
                    home_pp_percent = team["powerPlayPct"]
                    home_fow_percent = team["faceoffWinPct"]
                if team["teamFullName"] == away_team:
                    both_team_stats += 1
                    away_goals_for = team["goalsForPerGame"]
                    away_goals_against = team["goalsAgainstPerGame"]
                    away_pp_percent = team["powerPlayPct"]
                    away_fow_percent = team["faceoffWinPct"]

            # Append the formatted game data to the list
            games.append({
                "home": home_team,
                "away": away_team,
                "home_abbrev": home_abrv,
                "away_abbrev": away_abrv,
                "team_code": home_code,
                "opp_code": away_code,
                "rest_days_home": 0 if home_abrv in all_prev_day_teams else 1,
                "rest_days_away": 0 if away_abrv in all_prev_day_teams else 1,
                "home_AVG_gf/gp": home_goals_for,
                "away_AVG_gf/gp": away_goals_for,
                'home_AVG_ga/gp': home_goals_against,
                'away_AVG_ga/gp': away_goals_against,
                'home_AVG_pp%': home_pp_percent,
                'away_AVG_pp%': away_pp_percent,
                'home_AVG_fow%': home_fow_percent,
                'away_AVG_fow%': away_fow_percent
            })

# Save the formatted schedule to a JSON file
with open('schedule.json', 'w') as f:
    json.dump(games, f, indent=4)

print("Schedule saved to schedule.json")



