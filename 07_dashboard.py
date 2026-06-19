"""Executive Gradio dashboard for the FIFA Match Predictor 2026 project."""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


# ---------------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------------

power_rankings = pd.read_csv(OUTPUT_DIR / "power_rankings.csv")
dataset_summary = pd.read_csv(OUTPUT_DIR / "dataset_summary.csv")

GROUP_FILES = {
    letter: OUTPUT_DIR / f"group_{letter}.csv"
    for letter in "ABCDEFGHIJKL"
}
group_tables = {
    letter: pd.read_csv(path)
    for letter, path in GROUP_FILES.items()
    if path.exists()
}


def build_qualification_tables():
    """Build direct, third-place, and complete qualification tables."""
    direct_rows = []
    third_rows = []

    for group_letter, standings in group_tables.items():
        for position in (0, 1):
            row = standings.iloc[position]
            direct_rows.append({
                "Team": row["Team"],
                "Group": group_letter,
                "Position": position + 1,
                "Points": int(row["Points"]),
                "GD": int(row["GD"]),
                "GF": int(row["GF"]),
                "Qualification": "Direct",
            })

        third = standings.iloc[2]
        third_rows.append({
            "Team": third["Team"],
            "Group": group_letter,
            "Position": 3,
            "Points": int(third["Points"]),
            "GD": int(third["GD"]),
            "GF": int(third["GF"]),
        })

    direct = pd.DataFrame(direct_rows)
    third_rankings = (
        pd.DataFrame(third_rows)
        .sort_values(
            ["Points", "GD", "GF", "Group"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    third_rankings.insert(0, "Third-place rank", range(1, len(third_rankings) + 1))
    third_rankings["Status"] = [
        "Qualified" if index < 8 else "Eliminated"
        for index in range(len(third_rankings))
    ]

    best_thirds = third_rankings.head(8).copy()
    best_thirds["Qualification"] = "Best third place"
    all_qualifiers = pd.concat(
        [
            direct,
            best_thirds[
                ["Team", "Group", "Position", "Points", "GD", "GF", "Qualification"]
            ],
        ],
        ignore_index=True,
    )
    return direct, third_rankings, all_qualifiers


direct_qualifiers, third_place_rankings, qualified_teams = (
    build_qualification_tables()
)
BEST_THIRD_TEAMS = set(
    third_place_rankings.loc[
        third_place_rankings["Status"] == "Qualified", "Team"
    ]
)

home_model = joblib.load(MODEL_DIR / "home_goal_model.pkl")
away_model = joblib.load(MODEL_DIR / "away_goal_model.pkl")
team_encoder = joblib.load(MODEL_DIR / "team_encoder.pkl")
goal_strength = joblib.load(MODEL_DIR / "goal_strength.pkl")
elo_dict = joblib.load(MODEL_DIR / "elo_dict.pkl")
rank_dict = joblib.load(MODEL_DIR / "rank_dict.pkl")

TEAMS = sorted(team_encoder.classes_.tolist())
AVG_ELO = sum(elo_dict.values()) / len(elo_dict)
MAX_RANK = max(rank_dict.values())

EDA_IMAGES = [
    (OUTPUT_DIR / "matches_by_year.png", "World Cup matches by year"),
    (OUTPUT_DIR / "home_goals_distribution.png", "Home goals distribution"),
    (OUTPUT_DIR / "away_goals_distribution.png", "Away goals distribution"),
    (OUTPUT_DIR / "elo_distribution.png", "Elo rating distribution"),
    (OUTPUT_DIR / "fifa_rank_distribution.png", "FIFA rank distribution"),
    (OUTPUT_DIR / "top10_fifa_teams.png", "Top 10 FIFA-ranked teams"),
    (OUTPUT_DIR / "top10_elo_teams.png", "Top 10 Elo-rated teams"),
]
EDA_GALLERY = [
    (str(path), caption) for path, caption in EDA_IMAGES if path.exists()
]


# ---------------------------------------------------------------------------
# Prediction and presentation helpers
# ---------------------------------------------------------------------------

def predict_match(home_team: str, away_team: str):
    """Predict a score and return polished result cards plus supporting data."""
    if not home_team or not away_team:
        raise gr.Error("Select both teams to run a prediction.")
    if home_team == away_team:
        raise gr.Error("Please select two different teams.")

    home_enc = team_encoder.transform([home_team])[0]
    away_enc = team_encoder.transform([away_team])[0]

    home_elo = float(elo_dict.get(home_team, AVG_ELO))
    away_elo = float(elo_dict.get(away_team, AVG_ELO))
    elo_diff = home_elo - away_elo

    home_rank = int(rank_dict.get(home_team, MAX_RANK))
    away_rank = int(rank_dict.get(away_team, MAX_RANK))

    feature_frame = pd.DataFrame([{
        "home_team_enc": home_enc,
        "away_team_enc": away_enc,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
        "home_rank": home_rank,
        "away_rank": away_rank,
        "rank_diff": away_rank - home_rank,
        "home_goal_avg": goal_strength.get(home_team, 1.0),
        "away_goal_avg": goal_strength.get(away_team, 1.0),
    }])

    raw_home = float(home_model.predict(feature_frame)[0])
    raw_away = float(away_model.predict(feature_frame)[0])

    strength_bonus = elo_diff / 400
    raw_home += max(0, strength_bonus * 0.35)
    raw_away -= max(0, strength_bonus * 0.20)

    home_goals = max(0, round(raw_home))
    away_goals = max(0, round(raw_away))

    rating_gap = abs(elo_diff)
    home_prob, draw_prob, away_prob = 33.0, 34.0, 33.0
    if elo_diff > 0:
        home_prob += min(35, rating_gap / 20)
        away_prob -= min(25, rating_gap / 25)
    elif elo_diff < 0:
        away_prob += min(35, rating_gap / 20)
        home_prob -= min(25, rating_gap / 25)

    probability_total = home_prob + draw_prob + away_prob
    home_prob = round(home_prob / probability_total * 100, 1)
    draw_prob = round(draw_prob / probability_total * 100, 1)
    away_prob = round(100 - home_prob - draw_prob, 1)

    if home_goals > away_goals:
        verdict = f"{home_team} projected to win"
        verdict_class = "win"
    elif away_goals > home_goals:
        verdict = f"{away_team} projected to win"
        verdict_class = "win"
    else:
        verdict = "Draw projected"
        verdict_class = "draw"

    home_safe = html.escape(home_team)
    away_safe = html.escape(away_team)
    verdict_safe = html.escape(verdict)

    result_html = f"""
    <section class="prediction-result">
      <div class="result-kicker">MODEL FORECAST</div>
      <div class="scoreline">
        <div class="score-team score-team-left">
          <span class="team-name">{home_safe}</span>
          <span class="team-meta">HOME DESIGNATION</span>
        </div>
        <div class="score-number">{home_goals}</div>
        <div class="score-divider">:</div>
        <div class="score-number">{away_goals}</div>
        <div class="score-team">
          <span class="team-name">{away_safe}</span>
          <span class="team-meta">AWAY DESIGNATION</span>
        </div>
      </div>
      <div class="verdict {verdict_class}">{verdict_safe}</div>
      <div class="prob-grid">
        <div><strong>{home_prob}%</strong><span>{home_safe}</span></div>
        <div><strong>{draw_prob}%</strong><span>Draw</span></div>
        <div><strong>{away_prob}%</strong><span>{away_safe}</span></div>
      </div>
      <div class="prob-track">
        <span style="width:{home_prob}%" class="prob-home"></span>
        <span style="width:{draw_prob}%" class="prob-draw"></span>
        <span style="width:{away_prob}%" class="prob-away"></span>
      </div>
    </section>
    """

    comparison = pd.DataFrame({
        "Metric": ["Elo rating", "FIFA rank", "Historical goal strength"],
        home_team: [
            round(home_elo, 1),
            home_rank,
            round(float(goal_strength.get(home_team, 1.0)), 2),
        ],
        away_team: [
            round(away_elo, 1),
            away_rank,
            round(float(goal_strength.get(away_team, 1.0)), 2),
        ],
    })

    probability_plot = make_probability_plot(
        home_team, away_team, home_prob, draw_prob, away_prob
    )
    return result_html, probability_plot, comparison


def make_probability_plot(
    home_team: str,
    away_team: str,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8.5, 3.1))
    fig.patch.set_facecolor("#08111f")
    ax.set_facecolor("#08111f")

    labels = [home_team, "Draw", away_team]
    values = [home_prob, draw_prob, away_prob]
    colors = ["#22d3ee", "#d7a83f", "#64748b"]
    bars = ax.barh(labels, values, color=colors, height=0.5)

    ax.set_xlim(0, max(100, max(values) + 12))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="y", colors="#dce7f5", labelsize=10, length=0)
    for bar, value in zip(bars, values):
        ax.text(
            value + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            color="#f8fafc",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title(
        "Outcome probability profile",
        loc="left",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    return fig


def make_power_chart(limit: int = 12):
    subset = power_rankings.head(limit).sort_values("power_score")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.patch.set_facecolor("#08111f")
    ax.set_facecolor("#08111f")

    colors = [
        "#d7a83f" if rank <= 3 else "#22d3ee"
        for rank in subset["power_rank"]
    ]
    bars = ax.barh(subset["team"], subset["power_score"], color=colors, height=0.62)
    ax.set_xlim(max(0, subset["power_score"].min() - 0.04), 1.01)
    ax.xaxis.set_visible(False)
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="y", colors="#dce7f5", labelsize=10, length=0)
    for bar, value in zip(bars, subset["power_score"]):
        ax.text(
            value - 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="right",
            va="center",
            color="#04111d",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_title(
        f"Top {limit} composite power scores",
        loc="left",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    fig.tight_layout()
    return fig


def update_power_view(limit: int):
    limit = int(limit)
    table = power_rankings.head(limit).copy()
    table["power_score"] = table["power_score"].round(4)
    return make_power_chart(limit), table


def show_group(group_letter: str):
    table = group_tables[group_letter].copy()
    winner = html.escape(str(table.iloc[0]["Team"]))
    runner_up = html.escape(str(table.iloc[1]["Team"]))
    third_team = str(table.iloc[2]["Team"])
    third_status = (
        "Best-third qualifier"
        if third_team in BEST_THIRD_TEAMS
        else "Eliminated"
    )
    summary = f"""
    <div class="group-summary">
      <div><span>GROUP {group_letter} WINNER</span><strong>{winner}</strong></div>
      <div><span>RUNNER-UP</span><strong>{runner_up}</strong></div>
      <div><span>THIRD PLACE · {third_status.upper()}</span><strong>{html.escape(third_team)}</strong></div>
    </div>
    """
    return summary, table


def kpi_card(icon: str, value: str, label: str, note: str) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-note">{note}</div>
    </div>
    """


def qualifier_html(teams: pd.DataFrame, qualifier_type: str) -> str:
    pills = "".join(
        (
            f'<span class="team-pill {qualifier_type}">'
            f'{html.escape(str(row["Team"]))}'
            f'<small>Group {html.escape(str(row["Group"]))}</small></span>'
        )
        for _, row in teams.iterrows()
    )
    return f'<div class="qualifier-cloud">{pills}</div>'


top_team = power_rankings.iloc[0]
default_group = "A"
default_group_summary, default_group_table = show_group(default_group)


# ---------------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

:root {
  --ink: #06101d;
  --panel: #0a1728;
  --panel-2: #0e1e32;
  --line: rgba(148, 163, 184, .16);
  --text: #f4f8fc;
  --muted: #8da1b8;
  --cyan: #22d3ee;
  --cyan-soft: rgba(34, 211, 238, .12);
  --gold: #d7a83f;
}

.gradio-container {
  --table-text-color: #f8fafc !important;
  --table-text-color-dark: #f8fafc !important;
  --table-even-background-fill: #0a1728 !important;
  --table-even-background-fill-dark: #0a1728 !important;
  --table-odd-background-fill: #102238 !important;
  --table-odd-background-fill-dark: #102238 !important;
  --table-row-focus: #173653 !important;
  --table-row-focus-dark: #173653 !important;
  background:
    radial-gradient(circle at 14% 3%, rgba(34, 211, 238, .10), transparent 28rem),
    radial-gradient(circle at 90% 16%, rgba(215, 168, 63, .08), transparent 27rem),
    #06101d !important;
  color: var(--text);
  font-family: Inter, sans-serif !important;
}
.main { max-width: 1440px !important; padding: 0 28px 60px !important; }
footer { display: none !important; }

.hero {
  position: relative;
  overflow: hidden;
  padding: 54px 54px 48px;
  margin: 0 0 24px;
  border: 1px solid rgba(34, 211, 238, .18);
  border-top: 0;
  border-radius: 0 0 28px 28px;
  background:
    linear-gradient(120deg, rgba(7, 20, 35, .97), rgba(10, 29, 48, .92)),
    repeating-linear-gradient(90deg, transparent 0, transparent 79px, rgba(255,255,255,.02) 80px);
  box-shadow: 0 28px 80px rgba(0,0,0,.28);
}
.hero:after {
  content: "26";
  position: absolute;
  right: 42px;
  top: -46px;
  font: 800 240px/1 Manrope, sans-serif;
  color: transparent;
  -webkit-text-stroke: 1px rgba(34,211,238,.12);
  transform: rotate(-4deg);
}
.eyebrow, .section-kicker, .result-kicker {
  color: var(--cyan);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .2em;
  text-transform: uppercase;
}
.hero h1 {
  position: relative;
  z-index: 1;
  max-width: 790px;
  margin: 13px 0 14px;
  color: #fff;
  font: 800 clamp(34px, 5vw, 64px)/1.03 Manrope, sans-serif;
  letter-spacing: -.045em;
}
.hero h1 span { color: var(--cyan); }
.hero p {
  position: relative;
  z-index: 1;
  max-width: 740px;
  margin: 0;
  color: #aebed0;
  font-size: 15px;
  line-height: 1.75;
}
.hero-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 28px;
}
.hero-meta span {
  padding: 8px 12px;
  border: 1px solid rgba(148,163,184,.16);
  border-radius: 999px;
  color: #c8d5e3;
  background: rgba(255,255,255,.035);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .04em;
}

.kpi-row { gap: 14px !important; }
.kpi-card {
  height: 100%;
  min-height: 155px;
  padding: 20px;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: linear-gradient(145deg, rgba(15,31,51,.92), rgba(8,21,36,.92));
  box-shadow: 0 14px 36px rgba(0,0,0,.15);
}
.kpi-icon { color: var(--cyan); font-size: 20px; margin-bottom: 14px; }
.kpi-value {
  color: #fff;
  font: 800 28px/1 Manrope, sans-serif;
  letter-spacing: -.03em;
}
.kpi-label { margin-top: 7px; color: #dbe7f2; font-size: 12px; font-weight: 700; }
.kpi-note { margin-top: 6px; color: var(--muted); font-size: 10px; line-height: 1.4; }

.section-head { margin: 28px 0 15px; }
.section-head h2 {
  margin: 6px 0;
  color: #f8fbff;
  font: 800 25px/1.2 Manrope, sans-serif;
  letter-spacing: -.025em;
}
.section-head p { margin: 0; color: var(--muted); font-size: 13px; }

.gradio-container .tabs {
  border: 1px solid var(--line) !important;
  border-radius: 22px !important;
  background: rgba(7,18,31,.8) !important;
  overflow: hidden;
}
.gradio-container .tab-nav {
  padding: 8px 10px 0 !important;
  border-bottom: 1px solid var(--line) !important;
  background: rgba(8,20,34,.88) !important;
}
.gradio-container .tab-nav button {
  color: #91a4b9 !important;
  font-weight: 700 !important;
  font-size: 12px !important;
}
.gradio-container .tab-nav button.selected {
  color: var(--cyan) !important;
  border-color: var(--cyan) !important;
}
.tabitem { padding: 8px 20px 24px !important; }

.panel {
  border: 1px solid var(--line) !important;
  border-radius: 18px !important;
  background: linear-gradient(145deg, rgba(14,30,50,.92), rgba(8,20,34,.94)) !important;
  box-shadow: 0 16px 42px rgba(0,0,0,.14);
}
.panel.pad { padding: 14px !important; }
.gradio-container label, .gradio-container .label-wrap {
  color: #aebfd1 !important;
  font-weight: 700 !important;
  font-size: 11px !important;
}
.gradio-container input, .gradio-container textarea,
.gradio-container .secondary-wrap {
  background: #071522 !important;
  border-color: rgba(148,163,184,.18) !important;
  color: #eef6ff !important;
}

/* Explicit table palette: Gradio uses generated classes around dataframe cells. */
.gradio-container .data-table,
.gradio-container .data-table .table-wrap,
.gradio-container .data-table table,
.gradio-container .data-table thead,
.gradio-container .data-table tbody,
.gradio-container .data-table tr,
.gradio-container .data-table td,
.gradio-container .data-table th,
.gradio-container .table-wrap,
.gradio-container table {
  color: #f8fafc !important;
}
.gradio-container .data-table th,
.gradio-container .data-table thead tr,
.gradio-container table th {
  color: #f8fafc !important;
  background: #11263d !important;
  border-color: #294159 !important;
}
.gradio-container .data-table td,
.gradio-container .data-table tbody tr,
.gradio-container table td {
  color: #f8fafc !important;
  background: #0a1728 !important;
  border-color: #294159 !important;
  font-weight: 600 !important;
}
.gradio-container .data-table tbody tr:nth-child(odd),
.gradio-container .data-table tbody tr:nth-child(odd) td {
  color: #f8fafc !important;
  background: #102238 !important;
}
.gradio-container .data-table input,
.gradio-container .data-table textarea,
.gradio-container .data-table button,
.gradio-container .data-table span,
.gradio-container .data-table div {
  color: #f8fafc !important;
}
.gradio-container [role="listbox"],
.gradio-container [role="option"] {
  color: #102033 !important;
  background: #ffffff !important;
}
.gradio-container [role="option"]:hover,
.gradio-container [role="option"][aria-selected="true"] {
  color: #06101d !important;
  background: #a5f3fc !important;
}
.primary-btn {
  min-height: 48px !important;
  border: 0 !important;
  border-radius: 12px !important;
  color: #031018 !important;
  background: linear-gradient(100deg, #22d3ee, #67e8f9) !important;
  font-weight: 800 !important;
  letter-spacing: .02em;
  box-shadow: 0 10px 28px rgba(34,211,238,.18) !important;
}

.prediction-result {
  padding: 26px;
  border: 1px solid rgba(34,211,238,.17);
  border-radius: 18px;
  background:
    radial-gradient(circle at 50% 0, rgba(34,211,238,.09), transparent 55%),
    #081624;
}
.scoreline {
  display: grid;
  grid-template-columns: minmax(130px,1fr) 56px 20px 56px minmax(130px,1fr);
  align-items: center;
  gap: 10px;
  margin: 23px 0 18px;
}
.score-team { display: flex; flex-direction: column; }
.score-team-left { text-align: right; }
.team-name { color: #f8fafc; font: 700 18px/1.2 Manrope, sans-serif; }
.team-meta { margin-top: 5px; color: #637a91; font-size: 8px; font-weight: 800; letter-spacing: .12em; }
.score-number {
  display: grid;
  place-items: center;
  height: 58px;
  border: 1px solid rgba(34,211,238,.22);
  border-radius: 12px;
  color: #fff;
  background: rgba(34,211,238,.08);
  font: 800 30px Manrope, sans-serif;
}
.score-divider { color: #597087; text-align: center; font-size: 24px; }
.verdict {
  width: fit-content;
  margin: 0 auto 24px;
  padding: 7px 13px;
  border-radius: 999px;
  color: var(--cyan);
  background: rgba(34,211,238,.09);
  font-size: 11px;
  font-weight: 800;
}
.verdict.draw { color: var(--gold); background: rgba(215,168,63,.1); }
.prob-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; }
.prob-grid div { display: flex; flex-direction: column; text-align: center; }
.prob-grid strong { color: #f8fafc; font: 800 20px Manrope, sans-serif; }
.prob-grid span { margin-top: 3px; color: var(--muted); font-size: 10px; }
.prob-track { display: flex; height: 5px; margin-top: 15px; overflow: hidden; border-radius: 9px; }
.prob-track span { display: block; }
.prob-home { background: var(--cyan); }
.prob-draw { background: var(--gold); }
.prob-away { background: #64748b; }

.group-summary { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin: 4px 0 16px; }
.group-summary div {
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 13px;
  background: rgba(255,255,255,.025);
}
.group-summary span { display:block; color: var(--muted); font-size: 9px; font-weight:800; letter-spacing:.1em; }
.group-summary strong { display:block; margin-top:8px; color:#f8fafc; font:700 17px Manrope,sans-serif; }

.qualifier-cloud { display: flex; flex-wrap: wrap; gap: 9px; padding: 8px 0; }
.team-pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 9px 13px;
  border: 1px solid rgba(34,211,238,.14);
  border-radius: 999px;
  color: #cfe8ee;
  background: rgba(34,211,238,.055);
  font-size: 11px;
  font-weight: 700;
}
.team-pill small {
  color: #7890a8;
  font-size: 8px;
  font-weight: 800;
  text-transform: uppercase;
}
.team-pill.best-third {
  border-color: rgba(215,168,63,.28);
  color: #f3d68d;
  background: rgba(215,168,63,.08);
}
.qualification-rule {
  display: grid;
  grid-template-columns: 3fr 1fr;
  gap: 12px;
  margin: 12px 0 20px;
}
.qualification-rule div {
  padding: 17px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(255,255,255,.025);
}
.qualification-rule strong {
  display: block;
  color: #f8fafc;
  font: 800 22px Manrope, sans-serif;
}
.qualification-rule span {
  display: block;
  margin-top: 5px;
  color: var(--muted);
  font-size: 10px;
}
.insight-box {
  padding: 20px;
  border-left: 3px solid var(--gold);
  border-radius: 4px 14px 14px 4px;
  color: #b8c8d8;
  background: rgba(215,168,63,.055);
  font-size: 12px;
  line-height: 1.65;
}
.insight-box strong { color: #f3d68d; }
.method-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
.method-card { padding:18px; border:1px solid var(--line); border-radius:14px; background:rgba(255,255,255,.025); }
.method-card b { display:block; color:var(--cyan); font:800 19px Manrope,sans-serif; }
.method-card span { display:block; margin-top:7px; color:var(--muted); font-size:10px; line-height:1.5; }

.dataframe { border-radius: 14px !important; overflow: hidden !important; }
.gallery { border-radius: 16px !important; }
.footer-note {
  padding: 26px 10px 8px;
  color: #61758a;
  text-align: center;
  font-size: 10px;
  letter-spacing: .04em;
}

@media (max-width: 760px) {
  .main { padding: 0 12px 36px !important; }
  .hero { padding: 36px 22px; }
  .hero:after { display:none; }
  .scoreline { grid-template-columns: 1fr 45px 12px 45px 1fr; }
  .team-name { font-size: 12px; }
  .score-number { height:46px; font-size:24px; }
  .group-summary, .method-grid, .qualification-rule { grid-template-columns: 1fr; }
}
"""


theme = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#06101d",
    block_background_fill="#0a1728",
    block_border_color="rgba(148,163,184,.16)",
    block_label_text_color="#9fb2c6",
    body_text_color="#f4f8fc",
    input_background_fill="#071522",
    button_primary_background_fill="#22d3ee",
    button_primary_text_color="#031018",
    table_text_color="#f8fafc",
    table_text_color_dark="#f4f8fc",
    table_even_background_fill="#0a1728",
    table_odd_background_fill="#102238",
    table_even_background_fill_dark="#0a1728",
    table_odd_background_fill_dark="#102238",
    table_row_focus="#173653",
    table_row_focus_dark="#173653",
)


# ---------------------------------------------------------------------------
# Gradio application
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="FIFA 2026 Intelligence Suite",
    fill_width=True,
    analytics_enabled=False,
) as demo:
    gr.HTML("""
    <header class="hero">
      <div class="eyebrow">ADVANCED FOOTBALL ANALYTICS · 2026</div>
      <h1>FIFA Match <span>Intelligence</span> Suite</h1>
      <p>
        A decision-grade view of team strength, predictive match outcomes and
        simulated tournament progression—powered by historical results, FIFA
        rankings, Elo ratings and ensemble machine learning.
      </p>
      <div class="hero-meta">
        <span>RANDOM FOREST ENSEMBLE</span>
        <span>226 INTERNATIONAL TEAMS</span>
        <span>12 GROUPS SIMULATED</span>
        <span>CLIENT REPORT · JUNE 2026</span>
      </div>
    </header>
    """)

    with gr.Row(elem_classes="kpi-row"):
        gr.HTML(kpi_card("◈", "18,178", "International results", "Historical model training records"))
        gr.HTML(kpi_card("◎", "13,130", "FIFA ranking records", "Official ranking observations"))
        gr.HTML(kpi_card("↗", "6,678", "Elo observations", "Long-run team strength history"))
        gr.HTML(kpi_card("◆", "32", "Projected qualifiers", "24 direct + 8 best third-place teams"))
        gr.HTML(kpi_card("★", str(top_team["team"]), "Current power leader", f"Composite score {top_team['power_score']:.3f}"))

    with gr.Tabs():
        with gr.Tab("Executive Overview"):
            gr.HTML("""
            <div class="section-head">
              <div class="section-kicker">PORTFOLIO VIEW</div>
              <h2>What the model sees at a glance</h2>
              <p>Key rankings, methodology and tournament-level conclusions.</p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=6):
                    gr.Plot(
                        value=make_power_chart(12),
                        show_label=False,
                        elem_classes="panel",
                    )
                with gr.Column(scale=4):
                    gr.HTML("""
                    <div class="insight-box">
                      <strong>Executive signal</strong><br><br>
                      Spain leads the composite power index, closely followed by
                      Argentina and France. The index blends current performance
                      strength with official FIFA standing, while the score model
                      separately estimates goals for both sides of each fixture.
                    </div>
                    <div class="section-head">
                      <div class="section-kicker">MODEL COMPOSITION</div>
                      <h2>Three strength lenses</h2>
                    </div>
                    <div class="method-grid">
                      <div class="method-card"><b>40%</b><span>Elo performance score</span></div>
                      <div class="method-card"><b>35%</b><span>FIFA points score</span></div>
                      <div class="method-card"><b>25%</b><span>FIFA rank score</span></div>
                    </div>
                    """)

            gr.HTML("""
            <div class="section-head">
              <div class="section-kicker">DATA FOUNDATION</div>
              <h2>Analytical coverage</h2>
              <p>Core datasets used across exploration, ranking and prediction.</p>
            </div>
            """)
            gr.Dataframe(
                value=dataset_summary,
                interactive=False,
                show_row_numbers=False,
                elem_classes=["panel", "data-table"],
            )

        with gr.Tab("Match Predictor"):
            gr.HTML("""
            <div class="section-head">
              <div class="section-kicker">LIVE SCENARIO ENGINE</div>
              <h2>Build a match-up</h2>
              <p>Select any two supported international teams for an instant model forecast.</p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=4, elem_classes=["panel", "pad"]):
                    home_dropdown = gr.Dropdown(
                        choices=TEAMS,
                        value="Argentina",
                        label="TEAM 1 · HOME DESIGNATION",
                        filterable=True,
                    )
                    away_dropdown = gr.Dropdown(
                        choices=TEAMS,
                        value="France",
                        label="TEAM 2 · AWAY DESIGNATION",
                        filterable=True,
                    )
                    predict_button = gr.Button(
                        "RUN MATCH FORECAST  →",
                        variant="primary",
                        elem_classes="primary-btn",
                    )
                    gr.HTML("""
                    <div class="insight-box" style="margin-top:10px">
                      Predictions combine team identity, Elo differential, FIFA
                      ranking differential and historical goal strength. Scores
                      are rounded to regulation-time goals.
                    </div>
                    """)
                with gr.Column(scale=6):
                    initial_result, initial_plot, initial_comparison = predict_match(
                        "Argentina", "France"
                    )
                    result_card = gr.HTML(initial_result)

            with gr.Row():
                probability_chart = gr.Plot(
                    value=initial_plot,
                    show_label=False,
                    elem_classes="panel",
                )
                comparison_table = gr.Dataframe(
                    value=initial_comparison,
                    label="TEAM COMPARISON",
                    interactive=False,
                    show_row_numbers=False,
                    elem_classes=["panel", "data-table"],
                )

            predict_button.click(
                fn=predict_match,
                inputs=[home_dropdown, away_dropdown],
                outputs=[result_card, probability_chart, comparison_table],
            )

        with gr.Tab("Power Rankings"):
            gr.HTML("""
            <div class="section-head">
              <div class="section-kicker">COMPOSITE STRENGTH INDEX</div>
              <h2>Global team hierarchy</h2>
              <p>Normalized blend of Elo performance, FIFA points and official rank.</p>
            </div>
            """)
            ranking_limit = gr.Slider(
                minimum=5,
                maximum=30,
                value=15,
                step=1,
                label="NUMBER OF TEAMS TO DISPLAY",
            )
            with gr.Row():
                ranking_plot = gr.Plot(
                    value=make_power_chart(15),
                    show_label=False,
                    elem_classes="panel",
                )
                ranking_table = gr.Dataframe(
                    value=power_rankings.head(15).assign(
                        power_score=lambda frame: frame["power_score"].round(4)
                    ),
                    interactive=False,
                    show_row_numbers=False,
                    elem_classes=["panel", "data-table"],
                )
            ranking_limit.change(
                fn=update_power_view,
                inputs=ranking_limit,
                outputs=[ranking_plot, ranking_table],
            )

        with gr.Tab("Group Stage"):
            gr.HTML("""
            <div class="section-head">
              <div class="section-kicker">TOURNAMENT SIMULATION</div>
              <h2>Group-stage outlook</h2>
              <p>Every round-robin fixture is scored by the trained goal models.</p>
            </div>
            """)
            group_selector = gr.Radio(
                choices=list(group_tables.keys()),
                value=default_group,
                label="SELECT GROUP",
            )
            group_summary_component = gr.HTML(default_group_summary)
            group_table_component = gr.Dataframe(
                value=default_group_table,
                interactive=False,
                show_row_numbers=False,
                elem_classes=["panel", "data-table"],
            )
            group_selector.change(
                fn=show_group,
                inputs=group_selector,
                outputs=[group_summary_component, group_table_component],
            )
            gr.HTML("""
            <div class="section-head">
              <div class="section-kicker">PROJECTED ADVANCEMENT</div>
              <h2>32 teams advance to the next round</h2>
              <p>Group winners and runners-up qualify directly; the best eight third-place teams complete the field.</p>
            </div>
            """)
            gr.HTML("""
            <div class="qualification-rule">
              <div><strong>24 direct qualifiers</strong><span>First and second position from each of the 12 groups</span></div>
              <div><strong>8 best thirds</strong><span>Ranked by points, goal difference, then goals scored</span></div>
            </div>
            """)
            gr.HTML("""
            <div class="section-head">
              <div class="section-kicker">DIRECT QUALIFICATION</div>
              <h2>Group winners and runners-up</h2>
            </div>
            """)
            gr.HTML(qualifier_html(direct_qualifiers, "direct"))
            gr.HTML("""
            <div class="section-head">
              <div class="section-kicker">THIRD-PLACE COMPARISON</div>
              <h2>Best eight third-place teams</h2>
              <p>The gold-highlighted teams claim the remaining eight places.</p>
            </div>
            """)
            gr.HTML(
                qualifier_html(
                    third_place_rankings[
                        third_place_rankings["Status"] == "Qualified"
                    ],
                    "best-third",
                )
            )
            gr.Dataframe(
                value=third_place_rankings,
                label="ALL THIRD-PLACE TEAMS · RANKED",
                interactive=False,
                show_row_numbers=False,
                elem_classes=["panel", "data-table"],
            )

        with gr.Tab("Research & Findings"):
            gr.HTML("""
            <div class="section-head">
              <div class="section-kicker">EXPLORATORY ANALYSIS</div>
              <h2>Evidence behind the model</h2>
              <p>Distribution checks, historical coverage and ranking snapshots.</p>
            </div>
            """)
            gr.Gallery(
                value=EDA_GALLERY,
                label=None,
                columns=2,
                rows=4,
                height="auto",
                object_fit="contain",
                elem_classes="gallery",
            )
            gr.HTML("""
            <div class="insight-box" style="margin-top:18px">
              <strong>Interpretation note</strong><br><br>
              Goal distributions are concentrated at low scores, consistent with
              football's low-event structure. Elo and FIFA rankings provide
              complementary strength signals: Elo reacts to match performance,
              while FIFA points provide an official competitive benchmark.
            </div>
            """)

    gr.HTML("""
    <div class="footer-note">
      FIFA MATCH INTELLIGENCE SUITE · MACHINE-LEARNING DECISION SUPPORT ·
      PREDICTIONS ARE ANALYTICAL ESTIMATES, NOT GUARANTEES
    </div>
    """)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        footer_links=[],
        theme=theme,
        css=CSS,
    )
