import joblib
import pandas as pd

# ==================================================
# LOAD MODELS
# ==================================================

print("=" * 60)
print("FOOTBALL MATCH PREDICTION SYSTEM")
print("=" * 60)

home_model = joblib.load(
    "models/home_goal_model.pkl"
)

away_model = joblib.load(
    "models/away_goal_model.pkl"
)

team_encoder = joblib.load(
    "models/team_encoder.pkl"
)

goal_strength = joblib.load(
    "models/goal_strength.pkl"
)

elo_dict = joblib.load(
    "models/elo_dict.pkl"
)

rank_dict = joblib.load(
    "models/rank_dict.pkl"
)

# ==================================================
# AVERAGES
# ==================================================

avg_elo = (
    sum(elo_dict.values())
    / len(elo_dict)
)

max_rank = max(
    rank_dict.values()
)

# ==================================================
# INPUT
# ==================================================

home_team = input(
    "\nEnter Team 1: "
).strip()

away_team = input(
    "Enter Team 2: "
).strip()

# ==================================================
# VALIDATION
# ==================================================

all_teams = set(
    team_encoder.classes_
)

if home_team not in all_teams:

    print(
        f"\n{home_team} not found."
    )
    exit()

if away_team not in all_teams:

    print(
        f"\n{away_team} not found."
    )
    exit()

# ==================================================
# FEATURES
# ==================================================

home_team_enc = (
    team_encoder.transform(
        [home_team]
    )[0]
)

away_team_enc = (
    team_encoder.transform(
        [away_team]
    )[0]
)

home_elo = elo_dict.get(
    home_team,
    avg_elo
)

away_elo = elo_dict.get(
    away_team,
    avg_elo
)

elo_diff = (
    home_elo
    -
    away_elo
)

home_rank = rank_dict.get(
    home_team,
    max_rank
)

away_rank = rank_dict.get(
    away_team,
    max_rank
)

rank_diff = (
    away_rank
    -
    home_rank
)

home_goal_avg = (
    goal_strength.get(
        home_team,
        1.0
    )
)

away_goal_avg = (
    goal_strength.get(
        away_team,
        1.0
    )
)

# ==================================================
# FEATURE VECTOR
# ==================================================

X = pd.DataFrame([{

    "home_team_enc":
        home_team_enc,

    "away_team_enc":
        away_team_enc,

    "home_elo":
        home_elo,

    "away_elo":
        away_elo,

    "elo_diff":
        elo_diff,

    "home_rank":
        home_rank,

    "away_rank":
        away_rank,

    "rank_diff":
        rank_diff,

    "home_goal_avg":
        home_goal_avg,

    "away_goal_avg":
        away_goal_avg

}])

# ==================================================
# SCORE PREDICTION
# ==================================================

home_goals = (
    home_model.predict(X)[0]
)

away_goals = (
    away_model.predict(X)[0]
)

# ==================================================
# STRENGTH ADJUSTMENT
# ==================================================

strength_bonus = (
    elo_diff / 400
)

home_goals = (
    home_goals
    +
    max(0, strength_bonus * 0.35)
)

away_goals = (
    away_goals
    -
    max(0, strength_bonus * 0.20)
)

home_goals = max(
    0,
    round(home_goals)
)

away_goals = max(
    0,
    round(away_goals)
)

# ==================================================
# RESULT
# ==================================================

if home_goals > away_goals:

    winner = home_team

elif away_goals > home_goals:

    winner = away_team

else:

    winner = "Draw"

# ==================================================
# SIMPLE WIN PROBABILITY
# ==================================================

rating_gap = abs(
    elo_diff
)

home_prob = 33
draw_prob = 34
away_prob = 33

if elo_diff > 0:

    home_prob += min(
        35,
        rating_gap / 20
    )

    away_prob -= min(
        25,
        rating_gap / 25
    )

elif elo_diff < 0:

    away_prob += min(
        35,
        rating_gap / 20
    )

    home_prob -= min(
        25,
        rating_gap / 25
    )

total = (
    home_prob
    +
    draw_prob
    +
    away_prob
)

home_prob = round(
    home_prob / total * 100,
    1
)

draw_prob = round(
    draw_prob / total * 100,
    1
)

away_prob = round(
    away_prob / total * 100,
    1
)

# ==================================================
# OUTPUT
# ==================================================

print("\n")
print("=" * 60)
print("MATCH PREDICTION")
print("=" * 60)

print(
    f"\n{home_team} {home_goals} - {away_goals} {away_team}"
)

print(
    f"\nWinner: {winner}"
)

print(
    f"Goal Difference: {abs(home_goals-away_goals)}"
)

print("\nWin Probability")

print(
    f"{home_team}: {home_prob}%"
)

print(
    f"Draw: {draw_prob}%"
)

print(
    f"{away_team}: {away_prob}%"
)

print("\nELO")

print(
    f"{home_team}: {round(home_elo,2)}"
)

print(
    f"{away_team}: {round(away_elo,2)}"
)

print(
    f"ELO Difference: {round(elo_diff,2)}"
)

print("\nFIFA Ranking")

print(
    f"{home_team}: {home_rank}"
)

print(
    f"{away_team}: {away_rank}"
)