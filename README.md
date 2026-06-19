# FIFA Match Intelligence Suite 2026

A machine-learning football analytics project with a client-facing Gradio
dashboard for match forecasts, power rankings, group-stage simulations and
exploratory findings.

## Launch the dashboard

From PowerShell:

```powershell
.\.venv\Scripts\python.exe 07_dashboard.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860).

Alternatively, double-click `launch_dashboard.bat`.

## Dashboard sections

- **Executive Overview** — headline metrics, methodology and leading teams
- **Match Predictor** — interactive score and outcome forecast for 226 teams
- **Power Rankings** — composite Elo/FIFA strength hierarchy
- **Group Stage** — standings and 32 qualifiers across Groups A–L, including
  the eight best third-place teams ranked by points, goal difference and goals
  scored
- **Research & Findings** — EDA charts and analytical interpretation

## Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Predictions are analytical estimates and should not be treated as guarantees.
