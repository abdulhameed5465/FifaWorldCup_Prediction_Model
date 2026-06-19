
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# SETUP
# ==================================================

print("=" * 60)
print("FOOTBALL MATCH PREDICTION SYSTEM - EDA")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================

matches = pd.read_csv("data/worldcup_matches.csv")
elo = pd.read_csv("data/elo_ratings.csv")
fifa = pd.read_csv("data/fifa_rankings.csv")

datasets = {
    "World Cup Matches": matches,
    "ELO Ratings": elo,
    "FIFA Rankings": fifa
}

# ==================================================
# BASIC INFORMATION
# ==================================================

for name, df in datasets.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

# ==================================================
# SAVE MISSING VALUE REPORT
# ==================================================

report = []

for name, df in datasets.items():

    for col, count in df.isnull().sum().items():

        report.append([
            name,
            col,
            count
        ])

missing_df = pd.DataFrame(
    report,
    columns=[
        "Dataset",
        "Column",
        "Missing"
    ]
)

missing_df.to_csv(
    "outputs/missing_values_report.csv",
    index=False
)

# ==================================================
# DATASET SUMMARY
# ==================================================

summary = pd.DataFrame({

    "Dataset": [
        "World Cup Matches",
        "ELO Ratings",
        "FIFA Rankings"
    ],

    "Rows": [
        matches.shape[0],
        elo.shape[0],
        fifa.shape[0]
    ],

    "Columns": [
        matches.shape[1],
        elo.shape[1],
        fifa.shape[1]
    ]
})

summary.to_csv(
    "outputs/dataset_summary.csv",
    index=False
)

# ==================================================
# MATCHES BY YEAR
# ==================================================

plt.figure(figsize=(12, 5))

matches["Year"].value_counts() \
    .sort_index() \
    .plot(kind="bar")

plt.title(
    "World Cup Matches by Year"
)

plt.xlabel("Year")
plt.ylabel("Matches")

plt.tight_layout()

plt.savefig(
    "outputs/matches_by_year.png"
)

plt.close()

# ==================================================
# HOME GOALS DISTRIBUTION
# ==================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    matches["Home Team Goals"],
    bins=15,
    kde=True
)

plt.title(
    "Home Team Goals Distribution"
)

plt.xlabel("Goals")

plt.tight_layout()

plt.savefig(
    "outputs/home_goals_distribution.png"
)

plt.close()

# ==================================================
# AWAY GOALS DISTRIBUTION
# ==================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    matches["Away Team Goals"],
    bins=15,
    kde=True
)

plt.title(
    "Away Team Goals Distribution"
)

plt.xlabel("Goals")

plt.tight_layout()

plt.savefig(
    "outputs/away_goals_distribution.png"
)

plt.close()

# ==================================================
# ELO DISTRIBUTION
# ==================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    elo["rating"],
    bins=20,
    kde=True
)

plt.title(
    "ELO Rating Distribution"
)

plt.xlabel("Rating")

plt.tight_layout()

plt.savefig(
    "outputs/elo_distribution.png"
)

plt.close()

# ==================================================
# FIFA RANK DISTRIBUTION
# ==================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    fifa["rank"],
    bins=20,
    kde=True
)

plt.title(
    "FIFA Rank Distribution"
)

plt.xlabel("Rank")

plt.tight_layout()

plt.savefig(
    "outputs/fifa_rank_distribution.png"
)

plt.close()

# ==================================================
# TOP 10 FIFA TEAMS
# ==================================================

latest_fifa = (
    fifa
    .sort_values(
        ["date", "semester"]
    )
    .groupby("team")
    .tail(1)
)

top10 = (
    latest_fifa
    .sort_values("rank")
    .head(10)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top10,
    x="rank",
    y="team"
)

plt.title(
    "Top 10 FIFA Ranked Teams"
)

plt.tight_layout()

plt.savefig(
    "outputs/top10_fifa_teams.png"
)

plt.close()

# ==================================================
# TOP 10 ELO TEAMS
# ==================================================

elo["date"] = pd.to_datetime(
    elo["date"],
    format="mixed",
    errors="coerce"
)

latest_elo = (
    elo
    .sort_values("date")
    .groupby("team")
    .tail(1)
)

top10_elo = (
    latest_elo
    .sort_values(
        "rating",
        ascending=False
    )
    .head(10)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top10_elo,
    x="rating",
    y="team"
)

plt.title(
    "Top 10 ELO Rated Teams"
)

plt.tight_layout()

plt.savefig(
    "outputs/top10_elo_teams.png"
)

plt.close()

# ==================================================
# SAVE TOP TEAMS
# ==================================================

top10.to_csv(
    "outputs/top10_fifa_teams.csv",
    index=False
)

top10_elo.to_csv(
    "outputs/top10_elo_teams.csv",
    index=False
)

# ==================================================
# COMPLETE
# ==================================================

print("\nEDA COMPLETE")

print("""
Generated Files:

outputs/

dataset_summary.csv
missing_values_report.csv

matches_by_year.png

home_goals_distribution.png
away_goals_distribution.png

elo_distribution.png
fifa_rank_distribution.png

top10_fifa_teams.png
top10_elo_teams.png

top10_fifa_teams.csv
top10_elo_teams.csv
""")

