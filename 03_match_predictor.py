
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==================================================
# SETUP
# ==================================================

print("=" * 60)
print("FOOTBALL MATCH PREDICTION SYSTEM - SCORE MODEL")
print("=" * 60)

os.makedirs("models", exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================

results = pd.read_csv(
    "data/international_results.csv"
)

results["home_score"] = pd.to_numeric(
    results["home_score"],
    errors="coerce"
)

results["away_score"] = pd.to_numeric(
    results["away_score"],
    errors="coerce"
)

results = results.dropna(
    subset=[
        "home_score",
        "away_score"
    ]
)

results = results.reset_index(
    drop=True
)

elo = pd.read_csv(
    "data/elo_ratings.csv"
)

fifa = pd.read_csv(
    "data/fifa_rankings.csv"
)

# ==================================================
# CLEAN TEAM NAMES
# ==================================================

results["home_team"] = (
    results["home_team"]
    .astype(str)
    .str.replace("\xa0", " ", regex=False)
    .str.strip()
)

results["away_team"] = (
    results["away_team"]
    .astype(str)
    .str.replace("\xa0", " ", regex=False)
    .str.strip()
)

elo["team"] = (
    elo["team"]
    .astype(str)
    .str.replace("\xa0", " ", regex=False)
    .str.strip()
)

fifa["team"] = (
    fifa["team"]
    .astype(str)
    .str.replace("\xa0", " ", regex=False)
    .str.strip()
)

# ==================================================
# DATE CLEANING
# ==================================================

results["date"] = pd.to_datetime(
    results["date"],
    format="%d-%m-%Y",
    errors="coerce"
)

elo["date"] = pd.to_datetime(
    elo["date"],
    format="mixed",
    errors="coerce"
)

# ==================================================
# FIFA RANKINGS
# ==================================================

latest_fifa = (
    fifa
    .sort_values(
        ["date", "semester"]
    )
    .groupby("team")
    .tail(1)
)

rank_dict = dict(
    zip(
        latest_fifa["team"],
        latest_fifa["rank"]
    )
)

max_rank = latest_fifa["rank"].max()

# ==================================================
# ELO RATINGS
# ==================================================

latest_elo = (
    elo
    .sort_values("date")
    .groupby("team")
    .tail(1)
)

elo_dict = dict(
    zip(
        latest_elo["team"],
        latest_elo["rating"]
    )
)

avg_elo = latest_elo["rating"].mean()

# ==================================================
# GOAL STRENGTH
# ==================================================

home_attack = (
    results
    .groupby("home_team")
    ["home_score"]
    .mean()
)

away_attack = (
    results
    .groupby("away_team")
    ["away_score"]
    .mean()
)

goal_strength = {}

teams = (
    set(home_attack.index)
    |
    set(away_attack.index)
)

for team in teams:

    h = home_attack.get(team, 1.0)
    a = away_attack.get(team, 1.0)

    goal_strength[team] = (
        h + a
    ) / 2

# ==================================================
# FEATURES
# ==================================================

results["home_elo"] = (
    results["home_team"]
    .map(elo_dict)
    .fillna(avg_elo)
)

results["away_elo"] = (
    results["away_team"]
    .map(elo_dict)
    .fillna(avg_elo)
)

results["elo_diff"] = (
    results["home_elo"]
    -
    results["away_elo"]
)

results["home_rank"] = (
    results["home_team"]
    .map(rank_dict)
    .fillna(max_rank)
)

results["away_rank"] = (
    results["away_team"]
    .map(rank_dict)
    .fillna(max_rank)
)

results["rank_diff"] = (
    results["away_rank"]
    -
    results["home_rank"]
)

results["home_goal_avg"] = (
    results["home_team"]
    .map(goal_strength)
)

results["away_goal_avg"] = (
    results["away_team"]
    .map(goal_strength)
)

# ==================================================
# TEAM ENCODER
# ==================================================

team_encoder = LabelEncoder()

all_teams = pd.concat([
    results["home_team"],
    results["away_team"]
])

team_encoder.fit(all_teams)

results["home_team_enc"] = (
    team_encoder.transform(
        results["home_team"]
    )
)

results["away_team_enc"] = (
    team_encoder.transform(
        results["away_team"]
    )
)

# ==================================================
# FEATURES MATRIX
# ==================================================

X = results[
    [
        "home_team_enc",
        "away_team_enc",

        "home_elo",
        "away_elo",
        "elo_diff",

        "home_rank",
        "away_rank",
        "rank_diff",

        "home_goal_avg",
        "away_goal_avg"
    ]
]

# ==================================================
# TARGETS
# ==================================================

y_home = results["home_score"]
y_away = results["away_score"]

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_home_train, y_home_test = train_test_split(
    X,
    y_home,
    test_size=0.20,
    random_state=42
)

_, _, y_away_train, y_away_test = train_test_split(
    X,
    y_away,
    test_size=0.20,
    random_state=42
)

# ==================================================
# HOME MODEL
# ==================================================

home_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

home_model.fit(
    X_train,
    y_home_train
)

# ==================================================
# AWAY MODEL
# ==================================================

away_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

away_model.fit(
    X_train,
    y_away_train
)

# ==================================================
# EVALUATION
# ==================================================

home_preds = home_model.predict(X_test)
away_preds = away_model.predict(X_test)

print("\nHOME GOAL MODEL")

print(
    "MAE:",
    round(
        mean_absolute_error(
            y_home_test,
            home_preds
        ),
        3
    )
)

print(
    "R2:",
    round(
        r2_score(
            y_home_test,
            home_preds
        ),
        3
    )
)

print("\nAWAY GOAL MODEL")

print(
    "MAE:",
    round(
        mean_absolute_error(
            y_away_test,
            away_preds
        ),
        3
    )
)

print(
    "R2:",
    round(
        r2_score(
            y_away_test,
            away_preds
        ),
        3
    )
)

# ==================================================
# SAVE
# ==================================================

joblib.dump(
    home_model,
    "models/home_goal_model.pkl"
)

joblib.dump(
    away_model,
    "models/away_goal_model.pkl"
)

joblib.dump(
    team_encoder,
    "models/team_encoder.pkl"
)

joblib.dump(
    goal_strength,
    "models/goal_strength.pkl"
)

joblib.dump(
    elo_dict,
    "models/elo_dict.pkl"
)

joblib.dump(
    rank_dict,
    "models/rank_dict.pkl"
)

print("\nMODELS SAVED")

print("""
home_goal_model.pkl
away_goal_model.pkl

team_encoder.pkl

goal_strength.pkl
elo_dict.pkl
rank_dict.pkl
""")

