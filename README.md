# SafeRoute — Road Safety Intelligence

A Streamlit project that explores the accident case summaries in data/accident_ppt_summary.csv and uses a Python Random Forest model to estimate accident severity.

## Dashboard pages

- **Overview:** Annual totals, severity mix, fatalities, and people injured.
- **Case explorer:** Compare PowerPoint case counts by month, season, time, collision, accident nature, traffic violation, road direction, vehicle, or junction.
- **Driver trip check:** A parked/pre-trip checklist distinguishes tasks marked not done from items not yet checked, warns about possible consequences, and gives an action to take. It also shows a historical-pattern screen and a separate Random Forest severity estimate if a crash occurs.
- **Risk insights:** Compare fatal shares and severity patterns in a selected breakdown.
- **Data & reports:** Inspect and download the summary rows used by the project.

The annual accident total is calculated from the three calendar-year rows in the PowerPoint: 2023 (100), 2024 (71), and 2025 (74), for 245 accidents. The separate PowerPoint Total row is not added a second time. Counts in different factor breakdowns overlap, so they are never summed to calculate distinct accidents.

The Random Forest is trained on grouped PowerPoint counts, weighted by the reported counts. The historical screening signal is not a calibrated crash probability. The PowerPoint does not contain individual rows with joined factors, non-crash periods, or traffic exposure, so the project cannot make a real route-specific crash probability yet. Checklist alerts are reminders, not live vehicle/sensor alerts; use the screen only while parked. Safety advice is informed by [WHO road-safety guidance](https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries/).

## Run locally

    python -m pip install -r requirements.txt
    python -m streamlit run app.py --server.port 8507

Open http://localhost:8507 in your browser.
