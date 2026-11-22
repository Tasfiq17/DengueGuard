import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="DengueGuard — AI for Urban Dengue Outbreak Forecasting",
    page_icon="🦟",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dataset_snapshot.csv"
MODEL_DIR = ROOT / "models"

# ── Data Loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

@st.cache_data
def load_metadata():
    return json.loads((MODEL_DIR / "metadata.json").read_text())

@st.cache_resource
def load_models():
    return {h: joblib.load(MODEL_DIR / f"model_h{h}.joblib") for h in [2, 3, 4]}

df = load_data()
meta = load_metadata()
models = load_models()

# ── Feature Engineering ───────────────────────────────────────────────────────
def build_features(data):
    x = data.copy().sort_values(["ward_id", "date"]).reset_index(drop=True)
    gb = x.groupby("ward_id")
    for col in ["rainfall_mm", "avg_temp_c", "max_temp_c", "min_temp_c", "relative_humidity_pct"]:
        for lag in range(1, 5):
            x[f"{col}_lag_{lag}w"] = gb[col].shift(lag)
    for win in [2, 4, 8]:
        x[f"rainfall_sum_prev_{win}w"] = gb["rainfall_mm"].transform(
            lambda s: s.shift(1).rolling(win, min_periods=win).sum())
    for col in ["avg_temp_c", "relative_humidity_pct"]:
        x[f"{col}_mean_prev_4w"] = gb[col].transform(
            lambda s: s.shift(1).rolling(4, min_periods=4).mean())
    for lag in [1, 2, 3, 4]:
        x[f"cases_lag_{lag}w"] = gb["dengue_cases"].shift(lag)
    x["cases_mean_prev_4w"] = gb["dengue_cases"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=4).mean())
    x["cases_current"] = x["dengue_cases"]
    x["month"] = x.date.dt.month
    x["week_of_year"] = x.date.dt.isocalendar().week.astype(int)
    x["week_sin"] = np.sin(2 * np.pi * x.week_of_year / 52.18)
    x["week_cos"] = np.cos(2 * np.pi * x.week_of_year / 52.18)
    return x

def risk_label(value, thresholds):
    q75, q90, q97 = thresholds
    if value < q75:
        return "Low"
    if value < q90:
        return "Moderate"
    if value < q97:
        return "High"
    return "Critical"

RISK_EMOJI = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Critical": "🔴"}

# ── Top Header Banner ─────────────────────────────────────────────────────────
st.title("🦟 DengueGuard: AI for Urban Dengue Outbreak Forecasting")
st.caption("Multi-Week Early Warning Decision-Support System | Dhaka & Chattogram Wards")

# ── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.header("Forecast Controls")
h = st.sidebar.selectbox("Forecast horizon", [2, 3, 4], format_func=lambda v: f"{v} weeks ahead (H{v})")
ward_options = ["All wards"] + sorted(df.ward_id.astype(str).unique().tolist())
ward = st.sidebar.selectbox("Ward filter", ward_options)
risk_filter = st.sidebar.multiselect(
    "Risk categories",
    ["Low", "Moderate", "High", "Critical"],
    default=["Low", "Moderate", "High", "Critical"]
)

# ── Compute Latest Surveillance Forecasts ─────────────────────────────────────
inputs = meta["model_inputs"]
features_df = build_features(df)
latest = features_df.sort_values("date").groupby("ward_id").tail(1).copy()

n_expected = df.ward_id.nunique()
if len(latest) != n_expected:
    st.error(f"Expected one latest record for each of {n_expected} wards but found {len(latest)}.")
    st.stop()

missing_inputs = [c for c in inputs if c not in latest.columns]
if missing_inputs:
    st.error(f"Dashboard is missing model input columns: {missing_inputs}")
    st.stop()

bundle = models[h]
X = bundle["preprocessor"].transform(latest[inputs])
latest["predicted_cases"] = np.maximum(0, bundle["model"].predict(X))
latest["risk"] = latest.predicted_cases.apply(lambda v: risk_label(v, meta["thresholds"][str(h)]))
latest["forecast_date"] = latest.date + pd.to_timedelta(7 * h, unit="D")

last_date = pd.to_datetime(df.date.max())
target_date = last_date + pd.to_timedelta(7 * h, unit="D")

# ── Metric KPI Cards ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wards Monitored", int(latest.ward_id.nunique()))
c2.metric("High / Critical Risk Wards", int(latest.risk.isin(["High", "Critical"]).sum()))
c3.metric("Last Observation Date", str(last_date.date()))
c4.metric("Forecast Target Window", f"{target_date.date()} (+{h}w)")

st.divider()

# ── Main Tabs: Live Surveillance vs Custom Scenario Forecasting ───────────────
tab_surveillance, tab_simulator = st.tabs(["📊 Surveillance & Forecasts", "🧪 Custom Scenario Forecasting (What-If Simulator)"])

with tab_surveillance:
    # ── Ward Forecast Ranking Table ──
    st.subheader(f"Ward Outbreak Risk Ranking ({h}-Week Horizon)")
    
    shown = latest.copy()
    if ward != "All wards":
        shown = shown[shown.ward_id.astype(str) == ward]
    shown = shown[shown.risk.isin(risk_filter)].sort_values("predicted_cases", ascending=False)
    
    table_display = shown[[
        "city", "ward_id", "date", "forecast_date", "predicted_cases", "risk",
        "dengue_cases", "rainfall_mm", "relative_humidity_pct",
        "water_logging_risk_index", "population_density_sqkm"
    ]].rename(columns={
        "city": "City",
        "ward_id": "Ward",
        "date": "Issue Date",
        "forecast_date": "Target Date",
        "predicted_cases": "Predicted Weekly Cases",
        "risk": "Risk Tier",
        "dengue_cases": "Last Obs Cases",
        "rainfall_mm": "Rainfall (mm)",
        "relative_humidity_pct": "Humidity (%)",
        "water_logging_risk_index": "Waterlog Risk",
        "population_density_sqkm": "Pop Density (/km²)"
    })
    table_display["Predicted Weekly Cases"] = table_display["Predicted Weekly Cases"].round(2)
    st.dataframe(table_display, hide_index=True, use_container_width=True)

    st.divider()

    # ── Historical Graphs ──
    if ward != "All wards":
        hist = df[df.ward_id.astype(str) == ward].sort_values("date")
        st.subheader(f"Historical Dengue Cases — Ward {ward}")
        st.line_chart(hist.set_index("date")["dengue_cases"], use_container_width=True)

        st.subheader(f"Historical Weather Indicators — Ward {ward}")
        st.line_chart(hist.set_index("date")[["rainfall_mm", "relative_humidity_pct", "avg_temp_c"]], use_container_width=True)

        selected = latest[latest.ward_id.astype(str) == ward]
        if len(selected):
            r = selected.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Weekly Cases", f"{r.predicted_cases:.2f}")
            m2.metric("Assigned Risk Tier", f"{RISK_EMOJI.get(r.risk, '')} {r.risk}")
            m3.metric("Forecast Target Date", str(r.forecast_date.date()))
    else:
        st.subheader("Aggregate Dengue Cases Over Time (All Monitored Wards)")
        agg_cases = df.groupby("date")["dengue_cases"].sum().reset_index()
        st.line_chart(agg_cases.set_index("date")["dengue_cases"], use_container_width=True)

        st.subheader("Regional Weather Dynamics (Average Rainfall & Humidity)")
        agg_weather = df.groupby("date")[["rainfall_mm", "relative_humidity_pct", "avg_temp_c"]].mean().reset_index()
        st.line_chart(agg_weather.set_index("date")[["rainfall_mm", "relative_humidity_pct", "avg_temp_c"]], use_container_width=True)

    st.divider()

    # ── Held-out Test Performance & Baseline Comparison ──
    col_eval_1, col_eval_2 = st.columns(2)
    with col_eval_1:
        st.subheader("Held-Out Test Performance (Jul 2024 – Dec 2025)")
        metrics_path = ROOT / "test_metrics.csv"
        if metrics_path.exists():
            st.dataframe(pd.read_csv(metrics_path), hide_index=True, use_container_width=True)
        else:
            st.info("Test metrics file is not present.")

    with col_eval_2:
        st.subheader(f"Selected Model vs Simple Baselines (H{h})")
        cmp_path = ROOT / "test_baseline_comparison.csv"
        if cmp_path.exists():
            cmp_df = pd.read_csv(cmp_path)
            st.dataframe(cmp_df[cmp_df.horizon == h], hide_index=True, use_container_width=True)

    st.divider()

    # ── Feature Importance & Peak Analysis ──
    col_feat, col_peak = st.columns(2)
    with col_feat:
        imp_path = ROOT / f"feature_importance_h{h}.csv"
        if imp_path.exists():
            st.subheader(f"Top Predictive Features (Horizon {h})")
            imp = pd.read_csv(imp_path).head(12).sort_values("importance")
            st.bar_chart(imp.set_index("feature")["importance"], use_container_width=True)

    with col_peak:
        peak_path = ROOT / "peak_error_analysis.csv"
        if peak_path.exists():
            st.subheader(f"Peak Week Error Analysis (Top 10% Outbreaks)")
            pk = pd.read_csv(peak_path)
            pk_h = pk[pk.horizon == h].iloc[0]
            p1, p2 = st.columns(2)
            p1.metric("Peak Threshold (p90)", f"{int(pk_h.peak_threshold_p90)} cases")
            p2.metric("Peak MAE", f"{pk_h.peak_MAE:.2f}")
            p3, p4 = st.columns(2)
            p3.metric("Avg Actual (Peaks)", f"{pk_h.peak_mean_actual:.1f}")
            p4.metric("Avg Predicted (Peaks)", f"{pk_h.peak_mean_predicted:.1f}")
            st.caption(f"Underprediction rate in peak outbreak weeks: {pk_h.peak_underprediction_rate*100:.1f}%. High and critical alerts provide early operational warnings.")

# ── Tab 2: Custom Scenario Forecasting (What-If Simulator) ────────────────────
with tab_simulator:
    st.subheader("🧪 Interactive What-If Outbreak Simulator")
    st.markdown(
        "Simulate climate anomalies, monsoon rainfall surges, or urban drainage changes for a specific ward. "
        "The model dynamically ingests these inputs and computes the predicted dengue outbreak risk."
    )

    sim_ward_options = sorted(df.ward_id.astype(str).unique().tolist())
    selected_sim_ward = st.selectbox("Select Ward to Pre-fill Data", sim_ward_options, index=0)
    
    # Pre-fill with the latest observation for the chosen ward
    base_row = latest[latest.ward_id.astype(str) == selected_sim_ward].iloc[0].copy()
    
    sim_horizon = st.radio("Simulation Forecast Horizon", [2, 3, 4], index=h-2, horizontal=True,
                           format_func=lambda v: f"{v} Weeks Ahead (H{v})")

    st.markdown("#### Environmental & Vulnerability Factors")
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        sim_city = st.selectbox("City", ["Chattogram", "Dhaka"], index=0 if base_row["city"] == "Chattogram" else 1)
        sim_rainfall = st.slider("Current Week Rainfall (mm)", 0.0, 300.0, float(base_row["rainfall_mm"]), step=1.0)
        sim_rain_lag1 = st.slider("Rainfall 1 Week Ago (mm)", 0.0, 300.0, float(base_row["rainfall_mm_lag_1w"]), step=1.0)
        sim_rain_lag2 = st.slider("Rainfall 2 Weeks Ago (mm)", 0.0, 300.0, float(base_row["rainfall_mm_lag_2w"]), step=1.0)

    with s_col2:
        sim_humidity = st.slider("Relative Humidity (%)", 30.0, 100.0, float(base_row["relative_humidity_pct"]), step=0.5)
        sim_avg_temp = st.slider("Average Temperature (°C)", 15.0, 42.0, float(base_row["avg_temp_c"]), step=0.5)
        sim_max_temp = st.slider("Max Temperature (°C)", 18.0, 48.0, float(base_row["max_temp_c"]), step=0.5)
        sim_min_temp = st.slider("Min Temperature (°C)", 10.0, 35.0, float(base_row["min_temp_c"]), step=0.5)

    with s_col3:
        sim_waterlog = st.slider("Water Logging Risk Index", 0.0, 1.0, float(base_row["water_logging_risk_index"]), step=0.01)
        sim_sanitation = st.slider("Sanitation Score", 0.0, 1.0, float(base_row["sanitation_score"]), step=0.01)
        sim_pop_density = st.number_input("Population Density (/km²)", min_value=1000, max_value=80000, value=int(base_row["population_density_sqkm"]), step=500)
        sim_current_cases = st.number_input("Current Week Dengue Cases", min_value=0, max_value=200, value=int(base_row["cases_current"]), step=1)

    if st.button("🚀 Run Outbreak Simulation", type="primary"):
        # Construct custom feature series starting from base_row
        custom_input = base_row.to_dict()
        custom_input["city"] = sim_city
        custom_input["rainfall_mm"] = sim_rainfall
        custom_input["rainfall_mm_lag_1w"] = sim_rain_lag1
        custom_input["rainfall_mm_lag_2w"] = sim_rain_lag2
        custom_input["rainfall_sum_prev_2w"] = sim_rain_lag1 + sim_rain_lag2
        custom_input["relative_humidity_pct"] = sim_humidity
        custom_input["avg_temp_c"] = sim_avg_temp
        custom_input["max_temp_c"] = sim_max_temp
        custom_input["min_temp_c"] = sim_min_temp
        custom_input["water_logging_risk_index"] = sim_waterlog
        custom_input["sanitation_score"] = sim_sanitation
        custom_input["population_density_sqkm"] = sim_pop_density
        custom_input["cases_current"] = sim_current_cases
        
        sim_df = pd.DataFrame([custom_input])
        sim_bundle = models[sim_horizon]
        X_sim = sim_bundle["preprocessor"].transform(sim_df[inputs])
        sim_pred = float(np.maximum(0, sim_bundle["model"].predict(X_sim)[0]))
        sim_risk = risk_label(sim_pred, meta["thresholds"][str(sim_horizon)])
        
        st.success(f"Simulation Complete for Ward {selected_sim_ward} at Horizon H{sim_horizon} ({sim_horizon} weeks ahead)")
        
        res1, res2, res3 = st.columns(3)
        res1.metric("Simulated Predicted Cases", f"{sim_pred:.2f}")
        res2.metric("Projected Risk Tier", f"{RISK_EMOJI.get(sim_risk, '')} {sim_risk}")
        res3.metric("Baseline Ward Comparison", f"{sim_pred - base_row['predicted_cases']:+.2f} vs current forecast")

        # Driver interpretation
        st.markdown("#### Primary Model Risk Drivers Identified:")
        driver_insights = []
        if sim_rain_lag1 + sim_rain_lag2 > 80:
            driver_insights.append("🌧️ **Heavy 2-Week Cumulative Rainfall:** Provides stagnant pools for Aedes breeding maturation (7–14 day cycle).")
        if sim_humidity > 75:
            driver_insights.append("💧 **High Relative Humidity (>75%):** Significantly extends adult mosquito survival and increases viral biting rates.")
        if sim_waterlog > 0.6:
            driver_insights.append("⚠️ **High Water Logging Vulnerability:** Poor urban drainage delays runoff, multiplying localized vector density.")
        if sim_current_cases > 5:
            driver_insights.append("🦟 **Active Transmission Reservoir:** Existing human cases sustain localized vector transmission cycles.")
        if not driver_insights:
            driver_insights.append("✅ **Moderate Environmental Profile:** Meteorological and urban factors are currently within standard ranges.")
        
        for insight in driver_insights:
            st.markdown(f"- {insight}")

# ── Clean Footer ──────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "DengueGuard · EDU AI Engineering Hackathon 2026 · East Delta University<br>"
    "Predictive early warning system for public health resource planning in Dhaka and Chattogram."
    "</div>",
    unsafe_allow_html=True
)
