# app.py
# Run: streamlit run app.py

import json, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Green Hydraulics Advisor", page_icon="🟢", layout="wide")

CSV_FILE = "hydraulics_ops_sustainability_dataset.csv"
ART = Path("artifacts"); ART.mkdir(exist_ok=True)
MODEL_EFF = ART / "model_efficiency.joblib"
MODEL_CO2 = ART / "model_co2.joblib"

# -------------------- Styles --------------------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1150px;}
.kpi {border: 1px solid rgba(120,120,120,0.2); border-radius: 16px; padding: 14px 16px; background: rgba(200,200,200,0.06);}
.kpi h3 {font-size: 0.95rem; margin: 0 0 6px 0; color: #777;}
.kpi div {font-size: 1.35rem; font-weight: 700;}
.caption {color: #8a8a8a;}
hr {border: none; height: 1px; background: rgba(120,120,120,0.25); margin: .6rem 0 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🟢 Green Hydraulics Advisor")
st.caption("Professional ML guidance for hydraulic efficiency & sustainability. Backed by your operations dataset.")

# -------------------- Data / models --------------------
@st.cache_data
def load_reference():
    df = pd.read_csv(CSV_FILE)
    cats = ['region','client_type','product_line','material','control_type','sensor_pack','oil_type']
    for c in cats:
        df[c] = df[c].astype("category")
    return df, cats

def train_models_inline():
    # Lightweight import to reuse the same logic without subprocess
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingRegressor

    df, _ = load_reference()
    # model 1: efficiency
    num1 = ['filtration_rating_micron','oil_change_interval_hours','test_pressure_bar','test_duration_min','leak_rate_ml_min']
    cat = ['region','client_type','product_line','material','control_type','sensor_pack','oil_type']
    X1, y1 = df[num1 + cat], df['efficiency_pct']

    pre1 = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num1),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat),
    ])
    m1 = Pipeline([
        ("prep", pre1),
        ("hgb", HistGradientBoostingRegressor(random_state=42, learning_rate=0.08,
                                              max_depth=6, max_iter=400, l2_regularization=0.02))
    ])
    m1.fit(X1, y1)
    joblib.dump(m1, MODEL_EFF)

    # model 2: CO2
    num2 = ['energy_used_kwh','filtration_rating_micron','test_pressure_bar','test_duration_min','efficiency_pct']
    X2, y2 = df[num2 + cat], df['co2_kg']
    pre2 = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num2),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat),
    ])
    m2 = Pipeline([
        ("prep", pre2),
        ("hgb", HistGradientBoostingRegressor(random_state=42, learning_rate=0.07,
                                              max_depth=6, max_iter=450, l2_regularization=0.04))
    ])
    m2.fit(X2, y2)
    joblib.dump(m2, MODEL_CO2)

@st.cache_resource
def load_models():
    if (not MODEL_EFF.exists()) or (not MODEL_CO2.exists()):
        train_models_inline()  # safety net: train on first run
    return joblib.load(MODEL_EFF), joblib.load(MODEL_CO2)

try:
    df_ref, cat_cols = load_reference()
    model_eff, model_co2 = load_models()
except FileNotFoundError:
    st.error(f"Missing files. Ensure `{CSV_FILE}` is in this folder.")
    st.stop()

# -------------------- Inputs --------------------
st.sidebar.header("Configure a Job")

def cat_select(col):
    opts = list(df_ref[col].cat.categories)
    idx = 0 if len(opts) else 0
    return st.sidebar.selectbox(col.replace("_"," ").title(), opts, index=idx)

region       = cat_select("region")
client_type  = cat_select("client_type")
product_line = cat_select("product_line")
material     = cat_select("material")
control_type = cat_select("control_type")
sensor_pack  = cat_select("sensor_pack")
oil_type     = cat_select("oil_type")

filtration_rating_micron  = st.sidebar.number_input("Filtration Rating (micron)", 1, 1000, int(df_ref["filtration_rating_micron"].median()))
oil_change_interval_hours = st.sidebar.number_input("Oil Change Interval (hours)", 10, 10000, int(df_ref["oil_change_interval_hours"].median()))
test_pressure_bar         = st.sidebar.number_input("Test Pressure (bar)", 1, 2000, int(df_ref["test_pressure_bar"].median()))
test_duration_min         = st.sidebar.number_input("Test Duration (min)", 1, 1000, int(df_ref["test_duration_min"].median()))
leak_rate_ml_min          = st.sidebar.number_input("Leak Rate (ml/min)", 0.0, 500.0, float(df_ref["leak_rate_ml_min"].median()))
energy_used_kwh           = st.sidebar.number_input("Planned Energy Used (kWh)", 1, 10000, int(df_ref["energy_used_kwh"].median()))

baseline = dict(
    region=region, client_type=client_type, product_line=product_line, material=material,
    control_type=control_type, sensor_pack=sensor_pack, oil_type=oil_type,
    filtration_rating_micron=filtration_rating_micron, oil_change_interval_hours=oil_change_interval_hours,
    test_pressure_bar=test_pressure_bar, test_duration_min=test_duration_min,
    leak_rate_ml_min=leak_rate_ml_min, energy_used_kwh=energy_used_kwh
)

def smart_upgrade(b):
    up = b.copy()
    # control pref
    cvals = [str(c).lower() for c in df_ref['control_type'].cat.categories]
    if "servo" in cvals: up['control_type'] = df_ref['control_type'].cat.categories[cvals.index("servo")]
    elif "proportional" in cvals: up['control_type'] = df_ref['control_type'].cat.categories[cvals.index("proportional")]
    # oil pref
    ovals = [str(o).lower() for o in df_ref['oil_type'].cat.categories]
    if "biodegradable" in ovals: up['oil_type'] = df_ref['oil_type'].cat.categories[ovals.index("biodegradable")]
    # filtration finer
    up['filtration_rating_micron'] = max(3, int(round(b['filtration_rating_micron'] * 0.7)))
    # material lighter if available
    mvals = [str(m).lower() for m in df_ref['material'].cat.categories]
    if str(b['material']).lower() == "steel":
        if "composite" in mvals: up['material'] = df_ref['material'].cat.categories[mvals.index("composite")]
        elif "aluminum" in mvals: up['material'] = df_ref['material'].cat.categories[mvals.index("aluminum")]
    return up

upgraded = smart_upgrade(baseline)

# -------------------- Predict --------------------
def predict_eff(d):
    X = pd.DataFrame([{
        'filtration_rating_micron': d['filtration_rating_micron'],
        'oil_change_interval_hours': d['oil_change_interval_hours'],
        'test_pressure_bar': d['test_pressure_bar'],
        'test_duration_min': d['test_duration_min'],
        'leak_rate_ml_min': d['leak_rate_ml_min'],
        'region': d['region'], 'client_type': d['client_type'], 'product_line': d['product_line'],
        'material': d['material'], 'control_type': d['control_type'], 'sensor_pack': d['sensor_pack'], 'oil_type': d['oil_type']
    }])
    return float(model_eff.predict(X)[0])

def predict_co2(d, eff_pct):
    X = pd.DataFrame([{
        'energy_used_kwh': d['energy_used_kwh'],
        'filtration_rating_micron': d['filtration_rating_micron'],
        'test_pressure_bar': d['test_pressure_bar'],
        'test_duration_min': d['test_duration_min'],
        'region': d['region'], 'client_type': d['client_type'], 'product_line': d['product_line'],
        'material': d['material'], 'control_type': d['control_type'], 'sensor_pack': d['sensor_pack'], 'oil_type': d['oil_type'],
        'efficiency_pct': eff_pct
    }])
    return float(model_co2.predict(X)[0])

eff_base = predict_eff(baseline)
co2_base = predict_co2(baseline, eff_base)

eff_up = predict_eff(upgraded)
co2_up_const_input = predict_co2(upgraded, eff_up)

adj_energy = energy_used_kwh * (eff_base / max(eff_up, 1e-6))
up_same_output = upgraded | {'energy_used_kwh': float(adj_energy)}
co2_up_same_output = predict_co2(up_same_output, eff_up)

waste_base     = energy_used_kwh * (1 - eff_base/100)
waste_up_const = energy_used_kwh * (1 - eff_up/100)
waste_up_same  = adj_energy      * (1 - eff_up/100)

# -------------------- KPIs --------------------
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="kpi"><h3>Baseline Efficiency</h3><div>{eff_base:.1f}%</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="kpi"><h3>Upgraded Efficiency</h3><div>{eff_up-eff_base:+.1f}% → {eff_up:.1f}%</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="kpi"><h3>CO₂ (Const. Input)</h3><div>{co2_base:.2f} → {co2_up_const_input:.2f} kg</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="kpi"><h3>CO₂ (Same Output)</h3><div>{co2_base:.2f} → {co2_up_same_output:.2f} kg</div></div>', unsafe_allow_html=True)

st.write("")

# -------------------- Tabs --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Scenario Comparison", "📈 Historical Insights", "🧭 Recommendations", "🛠️ Admin"])

with tab1:
    st.subheader("CO₂ Comparison for Current Job")
    fig, ax = plt.subplots(figsize=(6.5,4))
    labels = ["Baseline\n(const input)", "Upgraded\n(const input)", "Upgraded\n(same output)"]
    vals = [co2_base, co2_up_const_input, co2_up_same_output]
    ax.bar(labels, vals)
    ax.set_ylabel("Emissions (kg CO₂eq)")
    ax.set_title("Emissions vs. Scenario")
    st.pyplot(fig)

    st.markdown("**Details**")
    detail = pd.DataFrame({
        "Scenario": labels,
        "Efficiency (%)": [eff_base, eff_up, eff_up],
        "Energy Used (kWh)": [energy_used_kwh, energy_used_kwh, adj_energy],
        "Waste Energy (kWh)": [waste_base, waste_up_const, waste_up_same],
        "CO₂ (kg)": vals
    })
    st.dataframe(detail, use_container_width=True)
    st.download_button("Download Scenario Summary (CSV)", detail.to_csv(index=False).encode("utf-8"),
                       file_name="scenario_summary.csv")

with tab2:
    st.subheader("Waste Energy Distribution (Historical)")
    ref = df_ref.copy()
    def grp(x): 
        c = str(x).lower().strip()
        return "Upgraded" if c in {"servo","proportional"} else "Baseline"
    ref["group"] = ref["control_type"].apply(grp)

    fig2, ax2 = plt.subplots(figsize=(6.5,4))
    ax2.hist(ref[ref["group"]=="Baseline"]["heat_loss_kwh"], bins=30, alpha=0.65, label="Baseline")
    ax2.hist(ref[ref["group"]=="Upgraded"]["heat_loss_kwh"], bins=30, alpha=0.65, label="Upgraded")
    ax2.set_xlabel("Waste Energy per Window (kWh)")
    ax2.set_ylabel("Count")
    ax2.legend(); ax2.set_title("Waste Energy: Baseline vs Upgraded")
    st.pyplot(fig2)

    st.subheader("Emissions vs Energy (Historical)")
    fig3, ax3 = plt.subplots(figsize=(6.5,4))
    b = ref[ref["group"]=="Baseline"]; u = ref[ref["group"]=="Upgraded"]
    ax3.scatter(b["energy_used_kwh"], b["co2_kg"], alpha=0.65, label="Baseline", s=16)
    ax3.scatter(u["energy_used_kwh"], u["co2_kg"], alpha=0.65, label="Upgraded", s=16)
    x, y = ref["energy_used_kwh"].to_numpy(), ref["co2_kg"].to_numpy()
    m, c = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 200)
    ax3.plot(xs, m*xs + c, linestyle="--", linewidth=2, label="Trend")
    ax3.set_xlabel("Energy Used (kWh)"); ax3.set_ylabel("Emissions (kg CO₂eq)")
    ax3.set_title("Emissions vs Energy Used"); ax3.legend()
    st.pyplot(fig3)

with tab3:
    st.subheader("Targeted Recommendations")
    recs = []
    if str(control_type).lower() not in {"servo","proportional"}:
        recs.append("Upgrade to **Servo/Proportional control** for efficiency and reliability gains.")
    oils = [str(o).lower() for o in df_ref['oil_type'].cat.categories]
    if str(oil_type).lower() != "biodegradable" and "biodegradable" in oils:
        recs.append("Use **biodegradable oil** to reduce spill impact and extend oil life.")
    if smart_upgrade(baseline)['filtration_rating_micron'] < baseline['filtration_rating_micron']:
        recs.append(f"Adopt **finer filtration** (~{smart_upgrade(baseline)['filtration_rating_micron']} µm) to extend oil life.")
    mvals = [str(m).lower() for m in df_ref['material'].cat.categories]
    if str(material).lower() == "steel" and ("composite" in mvals or "aluminum" in mvals):
        recs.append("Evaluate **composite/aluminum** components in mobile applications to reduce footprint.")
    if not recs:
        recs.append("Your configuration is close to optimal based on dataset patterns.")
    for r in recs: st.markdown(f"- {r}")

with tab4:
    st.subheader("Admin")
    st.write("Manage models and data.")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Re-train models now"):
            with st.spinner("Training models…"):
                train_models_inline()
                st.success("Models retrained and saved to ./artifacts. Refresh this page.")
    with colB:
        st.write(f"CSV detected: **{CSV_FILE}**")
        st.write(f"Models present: **{MODEL_EFF.exists() and MODEL_CO2.exists()}**")

st.markdown('<hr/>', unsafe_allow_html=True)
st.markdown('<span class="caption">Notes: “Same output” assumes equal useful work; input energy scales by baseline vs upgraded efficiency. Models: HistGradientBoostingRegressor (5-fold CV in training script).</span>', unsafe_allow_html=True)