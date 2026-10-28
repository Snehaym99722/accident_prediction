from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="SafeRoute | Road Safety Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --ink:#13283d; --muted:#718198; --line:#e2e9f0; --teal:#087e83; --orange:#f4a340; }
.stApp { background:#f5f8fb; color:var(--ink); }
.block-container { max-width:1450px; padding-top:1.1rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#10253a; }
[data-testid="stSidebar"] * { color:#eaf1f7; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.28rem 0; }
.hero {
  background:linear-gradient(115deg,#102a43 0%,#145d68 64%,#087e83 100%);
  color:#fff; padding:1.7rem 2rem; border-radius:20px; margin-bottom:1.1rem;
  box-shadow:0 12px 30px rgba(16,42,67,.14);
}
.hero .eyebrow { color:#ffc36c; font-size:.76rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { color:#fff; font-size:2.35rem; margin:.3rem 0 .25rem; }
.hero p { color:#e5f0f3; margin:0; font-size:1.03rem; }
.section-kicker { color:#087e83; font-size:.74rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.panel {
 background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem 1.2rem;
 box-shadow:0 4px 16px rgba(16,42,67,.035); margin-bottom:.8rem;
}
.small-note { color:#718198; font-size:.88rem; }
div[data-testid="stMetric"] {
 background:#fff; border:1px solid var(--line); padding:1rem 1.05rem;
 border-radius:15px; box-shadow:0 4px 16px rgba(16,42,67,.035);
}
div[data-testid="stMetricLabel"] { color:#718198; }
div[data-testid="stMetricValue"] { color:#13283d; }
.stButton button[kind="primary"] { background:#087e83; border-color:#087e83; }
hr { border-color:#e2e9f0; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "accident_ppt_summary.csv"
MODEL_PATH = ROOT / "models" / "accident_severity_random_forest.joblib"

SEVERITIES = {
    "accidents_fatal": "Fatal",
    "accidents_grievous_injury": "Grievous Injury",
    "accidents_minor_injury": "Minor Injury",
    "accidents_no_injury": "No Injury",
}
BREAKDOWNS = {
    4: "Month",
    5: "Weather season",
    6: "Time of day",
    7: "Collision type",
    8: "Accident nature",
    10: "Traffic violation",
    13: "Road direction",
    14: "Impacting vehicle",
    15: "Impacted vehicle",
    16: "Junction type",
}
MONTH_ORDER = ["January","February","March","April","May","June","July","August","September","October","November","December"]
SEVERITY_ORDER = ["Fatal", "Grievous Injury", "Minor Injury", "No Injury"]

if not DATA_PATH.exists():
    st.error(f"PowerPoint summary was not found at {DATA_PATH}")
    st.stop()

raw = pd.read_csv(DATA_PATH)
raw.columns = raw.columns.str.strip().str.lower()
needed = {"slide", "table_title", "category", "year", "metric", "value"}
if not needed.issubset(raw.columns):
    st.error("The PowerPoint summary is missing required columns: " + ", ".join(sorted(needed - set(raw.columns))))
    st.stop()
raw["slide"] = pd.to_numeric(raw["slide"], errors="coerce")
raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
for col in ["table_title", "category", "year", "metric"]:
    raw[col] = raw[col].fillna("").astype(str).str.strip()
raw["metric"] = raw["metric"].str.lower()

# Only the calendar-year rows make up the distinct accident total.
annual = raw[(raw["slide"] == 1) & (raw["metric"] == "accidents_total")].copy()
annual = annual[annual["value"].notna() & annual["year"].str.fullmatch(r"\d{4}", na=False)]
annual = annual.drop_duplicates("year").sort_values("year")
if annual.empty:
    st.error("No annual accident totals were found in slide 1 of the PowerPoint summary.")
    st.stop()
annual_totals = annual.set_index("year")["value"].astype(int)
accident_total = int(annual_totals.sum())
years = annual_totals.index.tolist()

# The PowerPoint's category-level breakdowns are separate views. Exclude their
# category Total rows because those repeat the category counts in that slide.
detail = raw[raw["slide"].isin(BREAKDOWNS)].copy()
detail = detail[detail["metric"].isin(SEVERITIES) & detail["value"].notna()]
detail = detail[detail["category"].str.casefold() != "total"]
detail["factor"] = detail["slide"].map(BREAKDOWNS)
detail["severity"] = detail["metric"].map(SEVERITIES)
detail["category"] = detail["category"].replace("", "Unspecified")
detail["year_period"] = detail["year"].replace("", "All years (2023–2025)")
detail["year_period"] = detail["year_period"].replace("2023-2025", "All years (2023–2025)")
cases = (
    detail.groupby(["factor", "category", "year_period", "severity"], as_index=False)["value"]
    .sum()
    .rename(columns={"value": "reported_count"})
)
if cases.empty:
    st.error("No accident severity breakdowns were found in the PowerPoint summary.")
    st.stop()

# Train on the summary categories, weighting each aggregate row by the reported
# count. These are not fabricated individual accident records.
features = ["factor", "category", "year_period"]
X = cases[features].astype(str)
y = cases["severity"].astype(str)
weights = cases["reported_count"].astype(float)
groups = cases["factor"].astype(str) + " | " + cases["category"].astype(str)
try:
    splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
    train_i, test_i = next(splitter.split(X, y, groups=groups))
except ValueError:
    st.error("There are not enough distinct PowerPoint categories to evaluate the model.")
    st.stop()

pre = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore"), features)])
model = Pipeline([
    ("prepare", pre),
    ("forest", RandomForestClassifier(
        class_weight="balanced_subsample", min_samples_leaf=2, random_state=42, n_jobs=-1
    )),
])
try:
    model.fit(X.iloc[train_i], y.iloc[train_i], forest__sample_weight=weights.iloc[train_i])
    y_test = y.iloc[test_i]
    predicted_test = model.predict(X.iloc[test_i])
    score = accuracy_score(y_test, predicted_test, sample_weight=weights.iloc[test_i])
except Exception as exc:
    st.error(f"Could not train the Random Forest from the PowerPoint summary: {exc}")
    st.stop()
try:
    pi = permutation_importance(
        model, X.iloc[test_i], y_test, n_repeats=5, random_state=42,
        scoring="accuracy", sample_weight=weights.iloc[test_i], n_jobs=-1
    )
    importance = pd.Series(pi.importances_mean, index=features).sort_values(ascending=False)
except Exception:
    importance = pd.Series(dtype=float)

# Refit the predictor using all summary rows.
model.fit(X, y, forest__sample_weight=weights)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": model, "features": features, "target": "severity"}, MODEL_PATH)

# Year-by-year outcomes and people figures from the overall table only.
overall = raw[(raw["slide"] == 1) & raw["year"].isin(years)].copy()
overall_severity = overall[overall["metric"].isin(SEVERITIES)].copy()
overall_severity["severity"] = overall_severity["metric"].map(SEVERITIES)
severity_by_year = overall_severity.pivot_table(
    index="year", columns="severity", values="value", aggfunc="sum", fill_value=0
).reindex(index=years, columns=SEVERITY_ORDER, fill_value=0)
people = overall[overall["metric"].isin(["persons_killed", "persons_grievous_injury", "persons_minor_injury"])]
people_totals = people.groupby("metric")["value"].sum().to_dict()
killed = int(people_totals.get("persons_killed", 0))
injured = int(people_totals.get("persons_grievous_injury", 0) + people_totals.get("persons_minor_injury", 0))

# Sidebar project navigation and source context.
st.sidebar.markdown("## 🚦 SafeRoute")
st.sidebar.caption("ROAD SAFETY INTELLIGENCE")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Workspace",
    ["Driver trip check", "Overview", "Case explorer", "Risk insights", "Data & reports"],
    label_visibility="visible",
)
st.sidebar.markdown("---")
st.sidebar.caption("Random Forest · Safety guidance")

st.markdown(
    """<div class="hero">
      <div class="eyebrow">Road safety analytics · SafeRoute</div>
      <h1>Accident intelligence</h1>
      <p>Explore reported cases and estimate severity from the road factors documented in your project.</p>
    </div>""",
    unsafe_allow_html=True,
)
st.caption("Road-safety decision support · Historical patterns from the supplied case study")

if page == "Overview":
    st.markdown('<div class="section-kicker">Corridor snapshot</div>', unsafe_allow_html=True)
    st.header("What the reported cases show")
    a, b, c, d = st.columns(4)
    a.metric("Reported accidents", f"{accident_total:,}")
    b.metric("Fatal accidents", f"{int(overall_severity.loc[overall_severity['severity'] == 'Fatal', 'value'].sum()):,}")
    c.metric("People killed", f"{killed:,}")
    d.metric("People injured", f"{injured:,}")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Accident trend by year")
        st.line_chart(annual_totals.rename("Accidents"), color="#087e83")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Severity mix")
        severity_totals = severity_by_year.sum(axis=0).sort_values(ascending=False)
        st.bar_chart(severity_totals.rename("Reported accidents"), color="#f4a340")
        st.markdown("</div>", unsafe_allow_html=True)
    st.subheader("Year-by-year severity")
    st.bar_chart(severity_by_year, color=["#b23b46", "#e18d43", "#e8c65b", "#148577"])
    st.info("The annual totals come from the PowerPoint’s overall accident table. Other slides break those cases down by factor and are not added again.")

elif page == "Case explorer":
    st.markdown('<div class="section-kicker">Explore the source</div>', unsafe_allow_html=True)
    st.header("Find patterns by road factor")
    col1, col2 = st.columns([1, 1])
    factor_options = sorted(cases["factor"].unique())
    factor = col1.selectbox("Choose a factor", factor_options)
    periods = sorted(cases.loc[cases["factor"] == factor, "year_period"].unique())
    period = col2.selectbox("Year or reported period", periods)
    selected = cases[(cases["factor"] == factor) & (cases["year_period"] == period)]
    pivot = selected.pivot_table(index="category", columns="severity", values="reported_count", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(columns=SEVERITY_ORDER, fill_value=0)
    sort_by = st.selectbox("Rank categories by", ["All reported cases", "Fatal cases", "Grievous injuries"])
    if sort_by == "Fatal cases":
        pivot = pivot.sort_values("Fatal", ascending=False)
    elif sort_by == "Grievous injuries":
        pivot = pivot.sort_values("Grievous Injury", ascending=False)
    else:
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    if not pivot.empty:
        totals = pivot.sum(axis=1)
        fatal_share = pivot["Fatal"].div(totals.replace(0, np.nan)).fillna(0)
        k1, k2, k3 = st.columns(3)
        k1.metric("Factor", factor)
        k2.metric("Categories", f"{len(pivot)}")
        k3.metric("Highest fatal share", f"{fatal_share.max():.0%}")
        st.bar_chart(pivot, color=["#b23b46", "#e18d43", "#e8c65b", "#148577"])
        st.dataframe(pivot.assign(**{"All outcomes": pivot.sum(axis=1)}), use_container_width=True)
    st.caption("Counts are shown within the selected PowerPoint factor and period; separate factors overlap.")

elif page == "Driver trip check":
    st.markdown('<div class="section-kicker">Driver mode · before you start</div>', unsafe_allow_html=True)
    st.header("Safe trip check")
    st.info("Use this screen only while parked. Do not interact with the dashboard while driving.")
    st.write("Mark each task as done or not done. A warning appears for a task marked not done; unconfirmed items stay as reminders.")
    with st.container(border=True):
        check_col1, check_col2 = st.columns(2)
        checks = {
            "protection": check_col1.selectbox("Seat belt fastened / helmet secured", ["Not checked", "Done", "Not done"], key="safe_check_protection"),
            "phone": check_col1.selectbox("Phone put away; no distraction", ["Not checked", "Done", "Not done"], key="safe_check_phone"),
            "driver": check_col1.selectbox("Driver rested and alcohol/drug free", ["Not checked", "Done", "Not done"], key="safe_check_driver"),
            "speed": check_col2.selectbox("Safe speed and following distance planned", ["Not checked", "Done", "Not done"], key="safe_check_speed"),
            "vehicle": check_col2.selectbox("Lights, tyres, and brakes checked", ["Not checked", "Done", "Not done"], key="safe_check_vehicle"),
        }
        completed = sum(status == "Done" for status in checks.values())
        st.progress(completed / len(checks), text=f"{completed} of {len(checks)} checks confirmed")
        warning_text = {
            "protection": "Protection is not in place. Without a properly fastened seat belt or helmet, injuries may be more severe if a crash occurs. Fasten it before moving.",
            "phone": "The phone is not put away. Phone use can delay reactions and make it harder to stay in lane or keep a safe gap. Put it away before starting.",
            "driver": "The driver is not confirmed rested and sober. Alcohol, drugs, or fatigue can impair judgement and reactions. Do not drive impaired or drowsy; arrange a rested, sober driver.",
            "speed": "A safe speed and gap are not planned. Higher speed can increase crash likelihood and injury severity, and leaves less time to stop. Follow the posted limit and leave a safe gap.",
            "vehicle": "The vehicle check is incomplete. A fault with tyres, lights, or brakes may affect grip, visibility, or stopping. Check the vehicle and address faults before departure.",
        }
        not_followed = [key for key, status in checks.items() if status == "Not done"]
        not_checked = [key for key, status in checks.items() if status == "Not checked"]
        if not_followed:
            st.error(f"Pause before departure: {len(not_followed)} safety task(s) are marked not done.")
            with st.expander("View warnings and what to do", expanded=True):
                for key in not_followed:
                    st.error(warning_text[key])
        if not_checked:
            st.info(f"Confirm {len(not_checked)} remaining checklist item(s) before departure.")
        if completed == len(checks):
            st.success("All listed checks confirmed. Keep following road signs and conditions; this checklist cannot guarantee safety.")
    st.caption("These are pre-trip reminders, not live vehicle or sensor alerts. Safety actions follow [WHO road-safety guidance](https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries/).")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        direction_options = sorted(cases.loc[cases["factor"] == "Road direction", "category"].unique())
        direction = col1.selectbox("Travel direction", direction_options, key="trip_direction")
        factor_options = sorted(cases.loc[cases["factor"] != "Road direction", "factor"].unique())
        factor = col2.selectbox("Main road condition", factor_options, key="pred_factor")
        factor_rows = cases[cases["factor"] == factor]
        category = col3.selectbox("Condition detail", sorted(factor_rows["category"].unique()), key="pred_category")
        run = st.button("Screen trip and estimate severity", type="primary", use_container_width=True)
    if run:
        all_period = "All years (2023–2025)"
        match = cases.loc[
            (cases["factor"] == factor)
            & (cases["category"] == category)
            & (cases["year_period"] == all_period)
        ]
        period = all_period if not match.empty else cases.loc[
            (cases["factor"] == factor) & (cases["category"] == category), "year_period"
        ].iloc[0]
        factor_counts = cases.loc[
            (cases["factor"] == factor) & (cases["year_period"] == all_period)
        ].groupby("category")["reported_count"].sum()
        condition_cases = int(factor_counts.get(category, 0))
        factor_cases = int(factor_counts.sum())
        condition_share = condition_cases / factor_cases if factor_cases else 0.0
        direction_cases = int(cases.loc[
            (cases["factor"] == "Road direction")
            & (cases["category"] == direction)
            & (cases["year_period"] == all_period), "reported_count"
        ].sum())
        st.subheader("1 · Future-trip screening signal")
        st.caption("Historical pattern only—not the probability that a crash will happen on this trip.")
        signal_a, signal_b, signal_c = st.columns(3)
        signal_a.metric("Past crashes in this condition", f"{condition_cases:,}")
        signal_b.metric("Share within this factor summary", f"{condition_share:.1%}")
        signal_c.metric("Past crashes in this direction", f"{direction_cases:,}")
        st.caption("The direction and condition counts come from separate PowerPoint summaries, so they cannot be combined into a route-specific probability.")
        query = pd.DataFrame([{"factor": factor, "category": category, "year_period": period}])
        label = model.predict(query)[0]
        class_scores = model.predict_proba(query)[0]
        classes = model.named_steps["forest"].classes_
        order = np.argsort(class_scores)[::-1]
        st.subheader("2 · Severity estimate if a crash occurs")
        result, scores = st.columns([1, 1.4])
        result.success(f"If a crash occurs, estimated severity: **{label}**")
        result.metric("Top Random Forest score", f"{class_scores[order[0]]:.1%}")
        result.caption(f"Scenario: {direction} · {factor}: {category}")
        scores.subheader("Scores across severity classes")
        scores.bar_chart(pd.Series({str(classes[i]): float(class_scores[i]) for i in order}, name="Model score"), color="#087e83")
        st.subheader("Safety actions for this scenario")
        factor_key = factor.casefold()
        category_key = category.casefold()
        if factor_key == "traffic violation" and "speed" in category_key:
            guidance = ["Reduce speed and follow the posted limit.", "Increase following distance; avoid risky overtaking.", "Use visible speed-limit signs and enforcement on this route."]
        elif factor_key == "traffic violation" and ("drunk" in category_key or "alcohol" in category_key):
            guidance = ["Do not drive after alcohol or drugs; arrange a sober ride.", "Use checkpoints and safe transport options where needed."]
        elif factor_key == "traffic violation" and "wrong side" in category_key:
            guidance = ["Stay in the marked lane and use signed U-turn points only.", "Improve lane-direction arrows, dividers, and wrong-way warning signs."]
        elif factor_key == "traffic violation" and "mobile" in category_key:
            guidance = ["Put the phone away before moving; pull over safely before using it.", "Use roadside reminders and enforcement at high-risk stretches."]
        elif factor_key == "traffic violation":
            guidance = ["Follow lane markings and traffic signs; avoid abrupt manoeuvres.", "Use targeted awareness and enforcement for this recorded violation."]
        elif factor_key == "weather season":
            guidance = ["Reduce speed and leave more stopping distance in poor conditions.", "Check visibility, tyre condition, and lights before travel.", "Keep drainage and road-surface hazards visible and promptly addressed."]
        elif factor_key == "time of day":
            guidance = ["Use headlights and reflective markings; make sure the route is well lit.", "Avoid driving when fatigued and take breaks on long trips.", "Allow extra time and keep a safe following distance."]
        elif factor_key == "collision type" and "pedestrian" in category_key:
            guidance = ["Slow down near settlements, bus stops, and pedestrian crossings.", "Keep crossings visible and provide safe footpaths or barriers where possible."]
        elif factor_key == "junction type":
            guidance = ["Approach slowly and check every direction before entering.", "Keep junction markings, signs, sightlines, and lighting clear."]
        elif factor_key in {"impacting vehicle", "impacted vehicle"}:
            guidance = ["Keep vulnerable road users separated from fast-moving traffic where feasible.", "Use protective barriers, visible crossings, and suitable speed control."]
        elif factor_key == "accident nature":
            guidance = ["Increase following distance and avoid sudden lane changes.", "Review roadside hazards, surface condition, and warning signs on this route."]
        else:
            guidance = ["Reduce speed, stay alert, and leave a safe gap.", "Follow road signs and markings, and report hazards to the road authority."]
        if label in {"Fatal", "Grievous Injury"}:
            guidance.append("Treat this as a high-consequence scenario: use protective equipment and know the nearest emergency response route.")
        for action in guidance:
            st.markdown(f"- {action}")
    st.markdown("---")
    st.caption("The Random Forest estimates injury severity from historical grouped counts. True crash probability needs records for both crash and no-crash periods, traffic exposure, and joined route/time conditions. The current PowerPoint has separate factor summaries, so this page screens historical patterns without claiming a real crash probability.")

elif page == "Risk insights":
    st.markdown('<div class="section-kicker">Patterns, not causes</div>', unsafe_allow_html=True)
    st.header("Compare severity across factors")
    factor = st.selectbox("Select a factor to inspect", sorted(cases["factor"].unique()), key="insight_factor")
    periods = sorted(cases.loc[cases["factor"] == factor, "year_period"].unique())
    period = st.selectbox("Select year / period", periods, key="insight_period")
    selected = cases[(cases["factor"] == factor) & (cases["year_period"] == period)]
    summary = selected.pivot_table(index="category", columns="severity", values="reported_count", aggfunc="sum", fill_value=0).reindex(columns=SEVERITY_ORDER, fill_value=0)
    if not summary.empty:
        summary["Total cases"] = summary.sum(axis=1)
        summary["Fatal share"] = (summary["Fatal"] / summary["Total cases"].replace(0, np.nan)).fillna(0)
        summary = summary.sort_values("Fatal share", ascending=False)
        highest = summary.iloc[0]
        x, ycol = st.columns([1, 1.5])
        x.metric("Highest fatal share in this view", f"{highest['Fatal share']:.1%}")
        x.write(f"Category: **{summary.index[0]}**")
        x.caption(f"{int(highest['Fatal'])} fatal cases out of {int(highest['Total cases'])} reported in that category.")
        ycol.bar_chart(summary[SEVERITY_ORDER], color=["#b23b46", "#e18d43", "#e8c65b", "#148577"])
        st.dataframe(summary, use_container_width=True)
    st.caption("A higher share describes the historical counts in this summary; it does not show that the factor caused an accident.")

elif page == "Data & reports":
    st.markdown('<div class="section-kicker">Project data</div>', unsafe_allow_html=True)
    st.header("PowerPoint records used in SafeRoute")
    st.write("The annual table provides the distinct accident total. Breakdown tables support the case explorer and the Random Forest.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Distinct annual total", f"{accident_total:,}")
    k2.metric("Calendar years", len(years))
    k3.metric("Breakdown categories", f"{cases[['factor', 'category']].drop_duplicates().shape[0]:,}")
    st.subheader("Annual totals")
    st.dataframe(annual[["year", "value"]].rename(columns={"year": "Year", "value": "Reported accidents"}), use_container_width=True, hide_index=True)
    st.subheader("Breakdown counts")
    view = cases[["factor", "category", "year_period", "severity", "reported_count"]].sort_values(["factor", "year_period", "category", "severity"])
    st.dataframe(view, use_container_width=True, height=420)
    st.download_button("Download model summary rows", view.to_csv(index=False).encode("utf-8"), "saferoute_powerpoint_summary.csv", "text/csv")
    st.info("Rows from different factors overlap. Do not sum breakdown counts to calculate the number of unique accidents.")
