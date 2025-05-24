import requests
import json
from unidecode import unidecode
from datetime import datetime, timedelta
import os

# Get today's date in the required format (YYYY-MM-DD)
today = datetime.today().strftime('%Y-%m-%d')
yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

# URL for the NHL schedule
url = f"https://api-web.nhle.com/v1/schedule/{today}"
print(url)

# URL for the team stats
url_stats = "https://api.nhle.com/stats/rest/en/team/summary?sort=shotsForPerGame&cayenneExp=seasonId=20242025%20and%20gameTypeId=2"
print(url_stats)

# Fetch the data from the URL
response = requests.get(url)
data = response.json()

response_stats = requests.get(url_stats)
data_stats = response_stats.json()

# Load the name-to-code mapping from your JSON file (name_codes.json)
script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the current script directory
data_dir = os.path.join(script_dir, '../data')
name_codes_path = os.path.join(data_dir, 'name_codes.json')

with open(name_codes_path) as f:
    name_codes = json.load(f)

# Initialize an empty list to store the game schedule
games = []

# Get teams that played the previous day
def check_prev_day():
    url = f"https://api-web.nhle.com/v1/schedule/{yesterday}"
    prev_day_teams = []

    # Fetch the data from the URL
    response = requests.get(url)
    data = response.json()

    for week in data['gameWeek']:
        if week['date'] == yesterday:
            for game in week['games']:
                #get team abbreviations
                home_abrv = game['homeTeam']['abbrev']
                away_abrv = game['awayTeam']['abbrev']
                
                prev_day_teams.append(home_abrv)
                prev_day_teams.append(away_abrv)

    return prev_day_teams

all_prev_day_teams = check_prev_day()

# Create a dictionary to store team stats for easy lookup
team_stats_dict = {}
for team in data_stats["data"]:
    team_full_name = team["teamFullName"]
    team_stats_dict[team_full_name] = {
        "goalsForPerGame": team.get("goalsForPerGame", 0),
        "goalsAgainstPerGame": team.get("goalsAgainstPerGame", 0),
        "powerPlayPct": (team.get("powerPlayPct", 0) * 100),
        "faceoffWinPct": (team.get("faceoffWinPct", 0) * 100)
    }

# Loop through the 'gameWeek' in the data
for week in data['gameWeek']:
    if week['date'] == today:
        for game in week['games']:
            # Combine 'placeName' and 'commonName' to get the full team name
            home_team = game['homeTeam']['placeName']['default'] + " " + game['homeTeam']['commonName']['default']
            away_team = game['awayTeam']['placeName']['default'] + " " + game['awayTeam']['commonName']['default']

            # Special case for Utah Hockey Club
            if home_team == "Utah Utah Hockey Club":
                home_team = "Utah Hockey Club"
            elif away_team == "Utah Utah Hockey Club":
                away_team = "Utah Hockey Club"

            # Get team abbreviations
            home_abrv = game['homeTeam']['abbrev']
            away_abrv = game['awayTeam']['abbrev']
            
            # Remove accents using unidecode
            home_team = unidecode(home_team)
            away_team = unidecode(away_team)

            home_code = name_codes['team_codes'][0].get(home_team)
            away_code = name_codes['team_codes'][0].get(away_team)

            # Get team stats from the dictionary
            home_stats = team_stats_dict.get(home_team, {})
            away_stats = team_stats_dict.get(away_team, {})

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
                "home_AVG_gf/gp": home_stats.get("goalsForPerGame", 0),
                "away_AVG_gf/gp": away_stats.get("goalsForPerGame", 0),
                'home_AVG_ga/gp': home_stats.get("goalsAgainstPerGame", 0),
                'away_AVG_ga/gp': away_stats.get("goalsAgainstPerGame", 0),
                'home_AVG_pp%': home_stats.get("powerPlayPct", 0),
                'away_AVG_pp%': away_stats.get("powerPlayPct", 0),
                'home_AVG_fow%': home_stats.get("faceoffWinPct", 0),
                'away_AVG_fow%': away_stats.get("faceoffWinPct", 0)
            })

# Save the formatted schedule to a JSON file
schedule_path = os.path.join(data_dir, 'schedule.json')  # Fixed the path
with open(schedule_path, 'w') as f:
    json.dump(games, f, indent=4)

print("Schedule saved to schedule.json")