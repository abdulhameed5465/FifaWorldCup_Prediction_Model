
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==================================================
# SETUP
# ==================================================

print("=" * 60)
print("FOOTBALL MATCH PREDICTION SYSTEM - POWER RANKINGS")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================

fifa = pd.read_csv(
    "data/fifa_rankings.csv"
)

elo = pd.read_csv(
    "data/elo_ratings.csv"
)

# ==================================================
# CLEAN TEAM NAMES
# ==================================================

fifa["team"] = (
    fifa["team"]
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

# ==================================================
# FIFA LATEST RANKING PER TEAM
# ==================================================

latest_fifa = (
    fifa
    .sort_values(
        ["date", "semester"]
    )
    .groupby("team")
    .tail(1)
    .copy()
)

print("\nLatest FIFA Teams:")
print(len(latest_fifa))

# ==================================================
# ELO LATEST RATING PER TEAM
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
    .copy()
)

print("\nLatest ELO Teams:")
print(len(latest_elo))

# ==================================================
# KEEP REQUIRED COLUMNS
# ==================================================

latest_fifa = latest_fifa[
    [
        "team",
        "rank",
        "total.points"
    ]
]

latest_elo = latest_elo[
    [
        "team",
        "rating"
    ]
]

# ==================================================
# MERGE
# ==================================================

power = pd.merge(
    latest_fifa,
    latest_elo,
    on="team",
    how="inner"
)

print("\nMerged Teams:")
print(len(power))

# ==================================================
# REMOVE MISSING RATINGS
# ==================================================

power = power.dropna(
    subset=[
        "rating"
    ]
)

# ==================================================
# NORMALIZATION
# ==================================================

scaler = MinMaxScaler()

# FIFA rank:
# lower rank = better
# invert it

power["rank_score"] = (
    power["rank"].max()
    -
    power["rank"]
)

power["rank_score"] = scaler.fit_transform(
    power[["rank_score"]]
)

power["fifa_score"] = scaler.fit_transform(
    power[["total.points"]]
)

power["elo_score"] = scaler.fit_transform(
    power[["rating"]]
)

# ==================================================
# POWER SCORE
# ==================================================

power["power_score"] = (

    0.40 * power["elo_score"]

    +

    0.35 * power["fifa_score"]

    +

    0.25 * power["rank_score"]

)

# ==================================================
# FINAL RANKING
# ==================================================

power = power.sort_values(

    by="power_score",

    ascending=False

).reset_index(
    drop=True
)

power["power_rank"] = (
    power.index + 1
)

# ==================================================
# REORDER COLUMNS
# ==================================================

power = power[
    [
        "power_rank",
        "team",
        "rank",
        "total.points",
        "rating",
        "power_score"
    ]
]

# ==================================================
# SAVE
# ==================================================

power.to_csv(
    "outputs/power_rankings.csv",
    index=False
)

# ==================================================
# TOP 25
# ==================================================

print("\nTOP 25 TEAMS\n")

print(
    power.head(25)
)

# ==================================================
# SUMMARY
# ==================================================

print("\nSaved:")

print(
    "outputs/power_rankings.csv"
)

print("\nTop 10 Teams")

for i, row in power.head(10).iterrows():

    print(
        f"{row['power_rank']:>2}. "
        f"{row['team']}"
    )
