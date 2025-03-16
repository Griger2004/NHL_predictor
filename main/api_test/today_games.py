import json
import requests

with open('../backend/schedule.json', 'r') as f:
    games = json.load(f)

api_url = 'http://localhost:5000/predict'

required_features = [
    'team_code', 'opp_code',
    'rest_days_home', 'rest_days_away', 'home_AVG_gf/gp', 'away_AVG_gf/gp',
    'home_AVG_ga/gp', 'away_AVG_ga/gp', 'home_AVG_pp%', 'away_AVG_pp%',
    'home_AVG_fow%', 'away_AVG_fow%'
]

for game in games:
    prediction_data = {}
    
    for feature in required_features:
        if feature in game:
            prediction_data[feature] = game[feature]

    # Send request
    try:
        response = requests.post(api_url, json=prediction_data)
        result = response.json()
        
        # Print prediction with team names
        winner = game['home'] if result['prediction'] == 1 else game['away']
        print(f"{game['home']} vs {game['away']}: Prediction = {winner} wins, " 
              f"Probabilities = {result['probabilities']}")
    except Exception as e:
        print(f"{game['home']} vs {game['away']}: Error - {str(e)}")
