import os
import joblib
import pandas as pd
from itertools import combinations

# ==================================================
# SETUP
# ==================================================

print("=" * 60)
print("GROUP STAGE PREDICTOR")
print("=" * 60)

os.makedirs(
    "outputs",
    exist_ok=True
)

# ==================================================
# LOAD MODELS
# ==================================================

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
# DEFAULTS
# ==================================================

avg_elo = (
    sum(elo_dict.values())
    / len(elo_dict)
)

max_rank = max(
    rank_dict.values()
)

# ==================================================
# EDIT GROUPS HERE
# ==================================================

GROUPS = {

    "A": [
        "Mexico",
        "South Korea",
        "Czechoslovakia",
        "South Africa"
    ],

    "B": [
        "Switzerland",
        "Canada",
        "Qatar",
        "Bosnia and Herzegovina"
    ],
    "C": [
        "Scotland",
        "Morocco",
        "Brazil",
        "Haiti"
    ],
    "D": [
        "United States",
        "Australia",
        "Turkey",
        "Paraguay"
    ],
    "E": [
        "Germany",
        "Ivory Coast",
        "Ecuador",
        "Curaçao"
    ],
    "F":[
        "Sweden",
        "Japan",
        "Netherlands",
        "Tunisia"
    ],
    "G":[
        "New Zealand",
        "Iran",
        "Belgium",
        "Egypt"
    ],
    "H":[
        "Uruguay",
        "Saudi Arabia",
        "Spain",
        "Cape Verde"
    ],
    "I":[
        "France",
        "Senegal",
        "Iraq",
        "Norway"
    ],
    "J":[
        "Argentina",
        "Algeria",
        "Austria",
        "Jordan"
    ],
    "K":[
        "Portugal",
        "DR Congo",
        "Uzbekistan",
        "Colombia"
    ],
    "L":[
        "England",
        "Croatia",
        "Ghana",
        "Panama"
    ]

}

# ==================================================
# MATCH PREDICTION FUNCTION
# ==================================================

def predict_match(home_team, away_team):

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

    home_goals = (
        home_model.predict(X)[0]
    )

    away_goals = (
        away_model.predict(X)[0]
    )

    strength_bonus = (
        elo_diff / 400
    )

    home_goals = (
        home_goals
        +
        max(
            0,
            strength_bonus * 0.35
        )
    )

    away_goals = (
        away_goals
        -
        max(
            0,
            strength_bonus * 0.20
        )
    )

    home_goals = max(
        0,
        round(home_goals)
    )

    away_goals = max(
        0,
        round(away_goals)
    )

    return (
        home_goals,
        away_goals
    )

# ==================================================
# QUALIFIED TEAMS
# ==================================================

qualified_teams = []
third_place_teams = []

# ==================================================
# PROCESS EACH GROUP
# ==================================================

for group_name, teams in GROUPS.items():

    print("\n")
    print("=" * 60)
    print(f"GROUP {group_name}")
    print("=" * 60)

    table = {}

    for team in teams:

        table[team] = {

            "Team": team,
            "Played": 0,
            "Won": 0,
            "Draw": 0,
            "Lost": 0,
            "GF": 0,
            "GA": 0,
            "GD": 0,
            "Points": 0

        }

    fixtures = list(
        combinations(
            teams,
            2
        )
    )

    print("\nFixtures\n")

    # ==========================================
    # PLAY MATCHES
    # ==========================================

    for home, away in fixtures:

        home_goals, away_goals = (
            predict_match(
                home,
                away
            )
        )

        print(
            f"{home} {home_goals} - {away_goals} {away}"
        )

        table[home]["Played"] += 1
        table[away]["Played"] += 1

        table[home]["GF"] += home_goals
        table[home]["GA"] += away_goals

        table[away]["GF"] += away_goals
        table[away]["GA"] += home_goals

        if home_goals > away_goals:

            table[home]["Won"] += 1
            table[away]["Lost"] += 1

            table[home]["Points"] += 3

        elif away_goals > home_goals:

            table[away]["Won"] += 1
            table[home]["Lost"] += 1

            table[away]["Points"] += 3

        else:

            table[home]["Draw"] += 1
            table[away]["Draw"] += 1

            table[home]["Points"] += 1
            table[away]["Points"] += 1

    # ==========================================
    # GOAL DIFFERENCE
    # ==========================================

    for team in table:

        table[team]["GD"] = (

            table[team]["GF"]

            -

            table[team]["GA"]

        )

    # ==========================================
    # STANDINGS
    # ==========================================

    standings = pd.DataFrame(
        table.values()
    )

    standings = standings.sort_values(

        by=[
            "Points",
            "GD",
            "GF"
        ],

        ascending=False

    ).reset_index(
        drop=True
    )

    print("\nStandings\n")

    print(
        standings
    )

    # ==========================================
    # SAVE GROUP TABLE
    # ==========================================

    standings.to_csv(

        f"outputs/group_{group_name}.csv",

        index=False

    )

    # ==========================================
    # QUALIFIERS
    # ==========================================

    q1 = standings.iloc[0]["Team"]
    q2 = standings.iloc[1]["Team"]

    qualified_teams.append(
        q1
    )

    qualified_teams.append(
        q2
    )

    third = standings.iloc[2]

    third_place_teams.append({
        "Team": third["Team"],
        "Group": group_name,
        "Position": 3,
        "Points": int(third["Points"]),
        "GD": int(third["GD"]),
        "GF": int(third["GF"])
    })

    print("\nQualified")

    print(
        q1
    )

    print(
        q2
    )

# ==================================================
# BEST THIRD-PLACE TEAMS
# ==================================================

third_place_df = pd.DataFrame(
    third_place_teams
)

third_place_df = third_place_df.sort_values(
    by=[
        "Points",
        "GD",
        "GF",
        "Group"
    ],
    ascending=[
        False,
        False,
        False,
        True
    ]
).reset_index(
    drop=True
)

third_place_df["Third-place rank"] = (
    third_place_df.index + 1
)

third_place_df["Status"] = "Eliminated"
third_place_df.loc[
    third_place_df.index < 8,
    "Status"
] = "Qualified"

best_third_teams = (
    third_place_df
    .head(8)
    ["Team"]
    .tolist()
)

qualified_teams.extend(
    best_third_teams
)

third_place_df.to_csv(
    "outputs/third_place_rankings.csv",
    index=False
)

# ==================================================
# SAVE ALL 32 QUALIFIERS
# ==================================================

qualified_df = pd.DataFrame({
    "Team": qualified_teams,
    "Qualification": (
        ["Direct"] * 24
        +
        ["Best third place"] * 8
    )
})

qualified_df.to_csv(

    "outputs/qualified_teams.csv",

    index=False

)

# ==================================================
# FINAL OUTPUT
# ==================================================

print("\n")
print("=" * 60)
print("ALL 32 QUALIFIED TEAMS")
print("=" * 60)

for team in qualified_teams:

    print(team)

print("\nSaved")

print(
    "outputs/qualified_teams.csv"
)

print(
    "outputs/third_place_rankings.csv"
)
