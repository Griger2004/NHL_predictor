import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def load_data(file_2024):
    nhl_game_data_2024 = pd.read_csv(file_2024)
    return nhl_game_data_2024

def clean_data(df):
    df = df.copy()

    columns_to_check = [
        "date", "time", "venue", "rest days", "result", "gf/gp", "ga/gp", "net goals",
        "opponent", "shots for", "shots against", "shot diff", "pp opp/gp", "ts/gp",
        "pp%", "pk%", "fow%", "team"
    ]
    df = df.drop_duplicates(subset=columns_to_check, keep='last')

    df.loc[:, "date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')
    df.loc[:, "opp_code"] = df["opponent"].astype("category").cat.codes
    df.loc[:, "team_code"] = df["team"].astype("category").cat.codes
    df.loc[:, "goalie_code"] = df["goalie"].astype("category").cat.codes
    df.loc[:, "hour"] = df["time"].fillna("0:00").str.replace(":.+", "", regex=True).astype(int)
    df.loc[:, "pp%"] = df["pp%"].replace('--', np.nan).astype(float)
    df.loc[:, "pk%"] = df["pk%"].replace('--', np.nan).astype(float)
    df.loc[:, "sv%"] = df["sv%"].astype(float)

    team_code_dict = dict(zip(df['team'].astype('category').cat.categories, range(len(df['team'].astype('category').cat.categories))))
    goalie_code_dict = dict(zip(df['goalie'].astype('category').cat.categories, range(len(df['goalie'].astype('category').cat.categories))))

    # Update home_win logic
    df.loc[:, "home_win"] = (
        ((df["venue"] == "Away") & (df["net goals"] < 0)) | 
        ((df["venue"] == "Home") & (df["net goals"] > 0))
    ).astype(int)

    return df

def rolling_averages(group, cols, new_cols):
    rolling_stats = group[cols].expanding(min_periods=1).mean().shift(1)
    group[new_cols] = rolling_stats
    return group

def apply_rolling_avg(df):
    stat_cols = ["gf/gp", "ga/gp", "shots for", "shots against", "pp opp/gp", "ts/gp", "pp%", "fow%"]
    avg_cols = [f"AVG_{c}" for c in stat_cols]
    df = df.groupby("team").apply(lambda x: rolling_averages(x, stat_cols, avg_cols))
    return df

def calculate_goalie_avg(group, cols, new_cols):
    rolling_stats = group[cols].expanding(min_periods=1).mean().shift(1)
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

def apply_goalie_avg(df):
    stat_cols = ["sv%"]
    avg_cols = ["AVG_sv%"]
    df = df.groupby("goalie").apply(lambda x: calculate_goalie_avg(x, stat_cols, avg_cols))
    df = df.sort_values(by="date")
    df.index = range(df.shape[0])
    return df

def create_game_ids(df):
    df['team_pair'] = df.apply(lambda row: tuple(sorted([row['team'], row['opponent']])), axis=1)
    df['game_key'] = df['date'].astype(str) + '_' + df['time'].astype(str) + '_' + df['team_pair'].astype(str)
    df['game_id'] = pd.factorize(df['game_key'])[0] + 1
    df.drop(columns=['team_pair', 'game_key'], inplace=True)
    return df

def combine_home_away_games(df):
    combined_rows = []
    for _, group in df.groupby('game_id'):
        home_games = group[group['venue'] == 'Home']
        away_games = group[group['venue'] == 'Away']
        if len(home_games) == 1 and len(away_games) == 1:
            home_game = home_games.iloc[0]
            away_game = away_games.iloc[0]
            combined_row = {
                'game_id': home_game['game_id'],
                'date': home_game['date'],
                'time': home_game['time'],
                # 'venue_code': home_game['venue_code'],
                'opp_code': away_game['opp_code'],
                'team_code': away_game['team_code'],
                'home_team': home_game['team'],
                'away_team': away_game['team'],
                'home_goalie': home_game['goalie'],
                'away_goalie': away_game['goalie'],
                'home_goalie_code': home_game['goalie_code'],
                'away_goalie_code': away_game['goalie_code'],
                'hour': home_game['hour'],
                'rest_days_home': home_game['rest days'],
                'rest_days_away': away_game['rest days'],
                'home_AVG_gf/gp': home_game['AVG_gf/gp'],
                'home_AVG_ga/gp': home_game['AVG_ga/gp'],
                'home_AVG_shots for': home_game['AVG_shots for'],
                'home_AVG_shots against': home_game['AVG_shots against'],
                'home_AVG_pp opp/gp': home_game['AVG_pp opp/gp'],
                'home_AVG_ts/gp': home_game['AVG_ts/gp'],
                'home_AVG_pp%': home_game['AVG_pp%'],
                'home_AVG_fow%': home_game['AVG_fow%'],
                'home_AVG_sv%': home_game['AVG_sv%'],
                'away_AVG_gf/gp': away_game['AVG_gf/gp'],
                'away_AVG_ga/gp': away_game['AVG_ga/gp'],
                'away_AVG_shots for': away_game['AVG_shots for'],
                'away_AVG_shots against': away_game['AVG_shots against'],
                'away_AVG_pp opp/gp': away_game['AVG_pp opp/gp'],
                'away_AVG_ts/gp': away_game['AVG_ts/gp'],
                'away_AVG_pp%': away_game['AVG_pp%'],
                'away_AVG_fow%': away_game['AVG_fow%'],
                'away_AVG_sv%': away_game['AVG_sv%'],
                'home_win': home_game['home_win']  # home_win is based on the home team's outcome
            }
            combined_rows.append(combined_row)
    combined_df = pd.DataFrame(combined_rows)
    return combined_df

def visualize_correlations(df, feature_cols):
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.show()

def save_mode(rfc_model):
    import pickle

    # Save the trained model to a file
    with open('model.pkl', 'wb') as file:
        pickle.dump(rfc_model, file)

    print("Model saved successfully!")

def main():
    nhl_game_data_2024 = load_data('final_nhl_data_2024.csv')
    nhl_game_data_2024 = clean_data(nhl_game_data_2024)
    nhl_game_data_2024 = apply_rolling_avg(nhl_game_data_2024)
    nhl_game_data_2024 = apply_goalie_avg(nhl_game_data_2024)
    nhl_game_data_2024 = create_game_ids(nhl_game_data_2024)
    combined_nhl_game_data_2024 = combine_home_away_games(nhl_game_data_2024)

    feature_cols = [
        'team_code', 
        'opp_code', 
        'home_goalie_code', 
        'away_goalie_code',
        'rest_days_home', 
        'rest_days_away', 
        'home_AVG_gf/gp',
        'away_AVG_gf/gp',
        'home_AVG_ga/gp',
        'away_AVG_ga/gp',
        'home_AVG_pp%',
        'away_AVG_pp%',
        'home_AVG_fow%',
        'away_AVG_fow%',
        'home_AVG_sv%',
        'away_AVG_sv%'
    ]

    visualize_correlations(combined_nhl_game_data_2024, feature_cols)

    X = combined_nhl_game_data_2024[feature_cols]
    y = combined_nhl_game_data_2024['home_win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = RandomForestClassifier(max_depth=None, max_features=None, min_samples_leaf=10, min_samples_split=10, n_estimators=50, random_state=1, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred: No', 'Pred: Yes'], yticklabels=['True: No', 'True: Yes'])
    plt.show()

    # save_mode(model)

if __name__ == '__main__':
    main()
