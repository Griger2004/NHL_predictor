from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model    
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Receive JSON
    try:
        
        input_df = pd.DataFrame([data])

        # Define feature columns (same as during training)
        feature_cols = [
            'team_code', 'opp_code', 'home_goalie_code', 'away_goalie_code',
            'rest_days_home', 'rest_days_away', 'home_AVG_gf/gp', 'away_AVG_gf/gp',
            'home_AVG_ga/gp', 'away_AVG_ga/gp', 'home_AVG_pp%', 'away_AVG_pp%',
            'home_AVG_fow%', 'away_AVG_fow%', 'home_AVG_sv%', 'away_AVG_sv%'
        ]

        # Ensure input matches feature columns
        input_data = input_df[feature_cols]

        # prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0].tolist()

        return jsonify({
            'prediction': int(prediction),  # 1 for home win, 0 for away win
            'probabilities': probability 
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app/start server
if __name__ == '__main__':
    app.run(debug=True)