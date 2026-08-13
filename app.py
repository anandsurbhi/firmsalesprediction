
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Firm Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# EXECUTIVE LIGHT UI
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy:#172554;
    --ink:#182338;
    --muted:#667085;
    --purple:#6957e8;
    --purple2:#8878f4;
    --lav:#f1efff;
    --blue:#2f6fed;
    --green:#20a05a;
    --orange:#ef9b22;
    --red:#e45567;
    --line:#e8ebf2;
    --bg:#f7f8fc;
}

* { font-family: Inter, sans-serif; }
.stApp { background:var(--bg); color:var(--ink); }
#MainMenu, footer, header { visibility:hidden; }

section[data-testid="stSidebar"] {
    background:#fff;
    border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] > div {
    padding:1.15rem 1rem;
}

.brand {
    padding:.35rem .25rem 1rem;
    border-bottom:1px solid var(--line);
    margin-bottom:1rem;
}
.brand-icon {
    display:inline-flex;
    width:38px;height:38px;
    align-items:center;justify-content:center;
    border-radius:11px;
    background:#eeeaff;
    color:var(--purple);
    font-size:1.15rem;
    margin-right:.5rem;
}
.brand-title {
    font-weight:800;
    color:var(--navy);
    font-size:1.08rem;
}
.brand-subtitle {
    color:#7c8799;
    font-size:.72rem;
    margin-top:.22rem;
    line-height:1.35;
}

.side-heading {
    color:var(--purple);
    font-size:.7rem;
    font-weight:800;
    letter-spacing:.1em;
    text-transform:uppercase;
    margin:1.1rem 0 .55rem;
}
.side-card {
    background:#fafaff;
    border:1px solid #ece9ff;
    border-radius:13px;
    padding:.75rem .8rem;
    margin-top:.55rem;
}
.side-row {
    display:flex;
    justify-content:space-between;
    padding:.25rem 0;
    color:#6b7485;
    font-size:.76rem;
}
.side-row b { color:#1d2942; }

.hero {
    background:
        radial-gradient(circle at 88% 18%, rgba(105,87,232,.14), transparent 20%),
        linear-gradient(135deg,#fff 0%,#f1efff 100%);
    border:1px solid #e5e1ff;
    border-radius:24px;
    padding:1.7rem 2rem;
    box-shadow:0 12px 32px rgba(25,35,65,.055);
    position:relative;
    overflow:hidden;
    margin-bottom:1rem;
}
.hero-grid {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:1.5rem;
}
.hero h1 {
    margin:0;
    color:var(--navy);
    font-size:2.15rem;
    line-height:1.08;
    font-weight:800;
    letter-spacing:-.045em;
}
.hero h1 span { color:var(--purple); }
.hero p {
    color:#697589;
    font-size:.92rem;
    line-height:1.5;
    max-width:720px;
    margin:.55rem 0 0;
}
.hero-visual {
    min-width:220px;
    text-align:center;
}
.bar {
    display:inline-block;
    width:19px;
    margin:0 3px;
    border-radius:7px 7px 2px 2px;
    background:linear-gradient(180deg,#9b8ff5,#6957e8);
    vertical-align:bottom;
}
.growth {
    color:var(--purple);
    font-size:.72rem;
    font-weight:800;
    margin-top:.35rem;
}

.workflow {
    display:flex;
    align-items:center;
    background:#fff;
    border:1px solid var(--line);
    border-radius:17px;
    padding:.55rem;
    box-shadow:0 6px 20px rgba(25,35,65,.045);
    margin-bottom:1rem;
    overflow-x:auto;
}
.step {
    flex:1;
    min-width:130px;
    display:flex;
    align-items:center;
    gap:.55rem;
    padding:.65rem .7rem;
    border-radius:12px;
}
.step.active { background:var(--lav); }
.step-num {
    width:31px;height:31px;
    flex:0 0 31px;
    border-radius:50%;
    background:#edf1ff;
    color:#5262c8;
    display:flex;
    align-items:center;
    justify-content:center;
    font-weight:800;
}
.step.active .step-num { background:var(--purple); color:#fff; }
.step-title { color:#17213d; font-weight:800; font-size:.8rem; }
.step-desc { color:#788397; font-size:.65rem; margin-top:.08rem; }
.arrow { color:#aab2c2; font-size:1.15rem; }

.kpi {
    background:#fff;
    border:1px solid var(--line);
    border-radius:17px;
    padding:.95rem 1rem .8rem;
    box-shadow:0 7px 22px rgba(25,35,65,.05);
    min-height:118px;
}
.kpi-top { display:flex; align-items:center; gap:.7rem; }
.kpi-icon {
    width:43px;height:43px;
    border-radius:13px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:1.12rem;
}
.kpi-label {
    color:#697589;
    font-size:.66rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.045em;
}
.kpi-value {
    color:var(--navy);
    font-size:1.45rem;
    font-weight:800;
    margin-top:.15rem;
}
.kpi-note { color:#8490a3; font-size:.68rem; margin-top:.35rem; }

.card {
    background:#fff;
    border:1px solid var(--line);
    border-radius:18px;
    padding:1rem 1.1rem;
    box-shadow:0 7px 22px rgba(25,35,65,.045);
    margin-bottom:1rem;
}
.card-head {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:1rem;
    margin-bottom:.35rem;
}
.card-title {
    color:var(--navy);
    font-size:.82rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.055em;
}
.card-sub {
    color:#8a94a6;
    font-size:.68rem;
}

.insight {
    background:linear-gradient(90deg,#f4f1ff,#faf9ff);
    border:1px solid #e5e0ff;
    border-left:4px solid var(--purple);
    border-radius:13px;
    padding:.85rem 1rem;
    margin:.25rem 0 1rem;
    color:#5f6b7d;
    font-size:.81rem;
    line-height:1.5;
}
.insight b { color:var(--navy); }

.perf {
    background:#fff;
    border:1px solid var(--line);
    border-radius:18px;
    padding:.75rem .45rem;
    box-shadow:0 7px 22px rgba(25,35,65,.045);
}
.perf-grid {
    display:grid;
    grid-template-columns:repeat(6,1fr);
}
.perf-cell {
    text-align:center;
    padding:.35rem .6rem;
    border-right:1px solid #edf0f4;
}
.perf-cell:last-child { border-right:0; }
.perf-label {
    color:#718096;
    font-size:.63rem;
    font-weight:800;
    text-transform:uppercase;
}
.perf-value {
    font-size:1.28rem;
    font-weight:800;
    margin:.15rem 0;
}
.perf-note { color:#8a94a6; font-size:.63rem; }

.driver {
    background:#fff;
    border:1px solid var(--line);
    border-radius:14px;
    padding:.75rem .85rem;
    margin:.45rem 0;
}
.driver-row {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:.7rem;
}
.driver-name { font-weight:700; color:#24304a; font-size:.8rem; }
.driver-val { font-weight:800; color:var(--purple); font-size:.78rem; }
.driver-track {
    height:6px;
    background:#eeeef5;
    border-radius:99px;
    margin-top:.5rem;
    overflow:hidden;
}
.driver-fill {
    height:100%;
    background:linear-gradient(90deg,#a69bf7,#6957e8);
    border-radius:99px;
}

.prediction {
    background:linear-gradient(135deg,#168449,#23aa63);
    border-radius:17px;
    padding:1.15rem 1.35rem;
    color:#fff;
    box-shadow:0 12px 28px rgba(22,132,73,.18);
}
.prediction-label { font-size:.7rem; opacity:.84; font-weight:700; text-transform:uppercase; }
.prediction-value { font-size:1.7rem; font-weight:800; margin-top:.15rem; }

.stButton > button, .stDownloadButton > button {
    background:var(--purple);
    color:white;
    border:0;
    border-radius:10px;
    font-weight:700;
    box-shadow:0 7px 18px rgba(105,87,232,.18);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background:#5948dc;
    color:#fff;
}

.stTabs [data-baseweb="tab-list"] {
    background:#fff;
    border:1px solid var(--line);
    border-radius:14px;
    padding:5px;
    gap:5px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:10px;
    color:#667085;
    font-weight:700;
}
.stTabs [aria-selected="true"] {
    background:var(--purple) !important;
    color:#fff !important;
}

div[data-testid="stMetric"] {
    background:#fff;
    border:1px solid var(--line);
    border-radius:14px;
    box-shadow:0 5px 18px rgba(25,35,65,.045);
}

.footer {
    text-align:center;
    color:#9aa3b2;
    font-size:.7rem;
    padding:1.2rem 0 .25rem;
}
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid")

# ============================================================
# HELPERS
# ============================================================
def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def backward_elimination(X, y, threshold=.05):
    cols = list(X.columns)
    while len(cols) > 1:
        model = sm.OLS(y, X[cols]).fit()
        p = model.pvalues.drop(labels="const", errors="ignore")
        if len(p) == 0 or p.max() <= threshold:
            break
        cols.remove(p.idxmax())
    return cols


def make_chart_axes(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", alpha=.14)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div class="brand">
  <div>
    <span class="brand-icon">📊</span>
    <span class="brand-title">Sales Intelligence</span>
  </div>
  <div class="brand-subtitle">Firm-level sales prediction & executive analytics</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="side-heading">Data Source</div>', unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

use_sample = False
if uploaded is None:
    use_sample = st.sidebar.checkbox("Use demonstration dataset", value=True)

# ============================================================
# DATA
# ============================================================
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    rng = np.random.default_rng(42)
    n = 738
    df = pd.DataFrame({
        "sales": rng.normal(.8, .75, n),
        "sga": rng.normal(1.1, .55, n),
        "rd": rng.exponential(.7, n),
        "ad": rng.exponential(.45, n),
        "tobinq": rng.normal(1.35, .55, n).clip(.2, 5),
        "de": rng.normal(1.0, .6, n).clip(.05, 4),
        "nwc": rng.normal(.8, .4, n),
        "profmarg": rng.normal(.12, .08, n),
        "salesgrowth": rng.normal(.08, .16, n),
    })
    df["sales"] = (
        .35 + .32*df["sga"] + .18*df["rd"] + .14*df["ad"]
        + .22*df["tobinq"] + .10*df["nwc"]
        + .20*df["profmarg"] + .18*df["salesgrowth"]
        + rng.normal(0, .35, n)
    )
else:
    st.info("Upload a CSV from the sidebar to start.")
    st.stop()

target = st.sidebar.selectbox(
    "Target variable",
    df.columns.tolist(),
    index=df.columns.tolist().index("sales") if "sales" in df.columns else 0,
)

test_size = st.sidebar.slider("Test size", .10, .40, .20, .05)
random_state = st.sidebar.number_input("Random state", 1, 999, 42)

missing = int(df.isna().sum().sum())
features = len(df.columns) - 1

st.sidebar.markdown('<div class="side-heading">Dataset Snapshot</div>', unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div class="side-card">
  <div class="side-row"><span>Observations</span><b>{len(df):,}</b></div>
  <div class="side-row"><span>Features</span><b>{features}</b></div>
  <div class="side-row"><span>Target</span><b>{target}</b></div>
  <div class="side-row"><span>Missing values</span><b>{missing:,}</b></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="side-heading">Model Controls</div>', unsafe_allow_html=True)
p_threshold = st.sidebar.slider("Feature selection p-value", .01, .20, .05, .01)

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
  <div class="hero-grid">
    <div>
      <h1>Firm-Level Sales <span>Regression Explorer</span></h1>
      <p>Executive analytics dashboard for understanding sales drivers,
      validating a Linear OLS model and generating decision-ready predictions.</p>
    </div>
    <div class="hero-visual">
      <div style="height:105px;display:flex;align-items:end;justify-content:center;">
        <span class="bar" style="height:25px;"></span>
        <span class="bar" style="height:40px;"></span>
        <span class="bar" style="height:53px;"></span>
        <span class="bar" style="height:69px;"></span>
        <span class="bar" style="height:84px;"></span>
      </div>
      <div class="growth">↗ SALES ANALYTICS</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# WORKFLOW
# ============================================================
st.markdown("""
<div class="workflow">
  <div class="step active"><div class="step-num">1</div><div><div class="step-title">Explore</div><div class="step-desc">Understand data</div></div></div>
  <div class="arrow">→</div>
  <div class="step"><div class="step-num">2</div><div><div class="step-title">Prepare</div><div class="step-desc">Clean & engineer</div></div></div>
  <div class="arrow">→</div>
  <div class="step"><div class="step-num">3</div><div><div class="step-title">Model</div><div class="step-desc">Build & train</div></div></div>
  <div class="arrow">→</div>
  <div class="step"><div class="step-num">4</div><div><div class="step-title">Validate</div><div class="step-desc">Test reliability</div></div></div>
  <div class="arrow">→</div>
  <div class="step"><div class="step-num">5</div><div><div class="step-title">Predict</div><div class="step-desc">Generate insight</div></div></div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# KPI ROW
# ============================================================
k1, k2, k3, k4 = st.columns(4)

cards = [
    ("🗄️", "TOTAL OBSERVATIONS", f"{len(df):,}", "Rows in dataset", "#eeeaff", "#5b4ee8"),
    ("☷", "TOTAL FEATURES", f"{features}", "Independent variables", "#edf4ff", "#2f6fed"),
    ("🎯", "TARGET VARIABLE", str(target), "Dependent variable", "#eaf8ee", "#20a05a"),
    ("✓", "DATA QUALITY", f"{100*(1-missing/max(len(df)*len(df.columns),1)):.1f}%", f"{missing:,} missing cells", "#fff4e5", "#ef9b22"),
]

for col, (icon, label, value, note, bg, color) in zip([k1,k2,k3,k4], cards):
    with col:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-top">
            <div class="kpi-icon" style="background:{bg};color:{color};">{icon}</div>
            <div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color};">{value}</div>
            </div>
          </div>
          <div class="kpi-note">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ============================================================
# CLEAN DATA / MODEL
# ============================================================
X_raw = df.drop(columns=[target]).copy()
y = pd.to_numeric(df[target], errors="coerce")
y = y.fillna(y.median())

for c in X_raw.columns:
    if pd.api.types.is_numeric_dtype(X_raw[c]):
        X_raw[c] = X_raw[c].fillna(X_raw[c].median())
    else:
        mode = X_raw[c].mode()
        X_raw[c] = X_raw[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")

X = pd.get_dummies(X_raw, drop_first=True)
X = sm.add_constant(X, has_constant="add").astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

full_model = sm.OLS(y_train, X_train).fit()

try:
    selected = backward_elimination(X_train, y_train, p_threshold)
    if "const" not in selected:
        selected = ["const"] + selected
except Exception:
    selected = list(X_train.columns)

final_model = sm.OLS(y_train, X_train[selected]).fit()

train_pred = final_model.predict(X_train[selected])
test_pred = final_model.predict(X_test[selected])

r2_train = final_model.rsquared
r2_test = 1 - np.sum((y_test-test_pred)**2) / np.sum((y_test-y_test.mean())**2)
adj_r2 = final_model.rsquared_adj
rmse = np.sqrt(mean_squared_error(y_test, test_pred))
mae = mean_absolute_error(y_test, test_pred)

# ============================================================
# CHARTS
# ============================================================
num_cols = df.select_dtypes(include=np.number).columns.tolist()

c_left, c_right = st.columns([1.15, 1])

with c_left:
    st.markdown("""
    <div class="card">
      <div class="card-head">
        <div class="card-title">Target Distribution</div>
        <div class="card-sub">Sales profile across firms</div>
      </div>
    """, unsafe_allow_html=True)

    if target in num_cols:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        sns.histplot(
            df[target].dropna(),
            bins=25,
            kde=True,
            color="#7968e8",
            edgecolor="white",
            ax=ax,
        )
        ax.axvline(df[target].mean(), color="#20a05a", linestyle="--", linewidth=2, label="Mean")
        ax.axvline(df[target].median(), color="#ef9b22", linestyle="-", linewidth=2, label="Median")
        ax.set_xlabel(target, fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        make_chart_axes(ax)
        ax.legend(frameon=False, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("</div>", unsafe_allow_html=True)

with c_right:
    st.markdown("""
    <div class="card">
      <div class="card-head">
        <div class="card-title">Correlation Heatmap</div>
        <div class="card-sub">Relationship between numeric variables</div>
      </div>
    """, unsafe_allow_html=True)

    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        corr = df[num_cols].corr()
        sns.heatmap(
            corr,
            cmap="PuOr_r",
            vmin=-1,
            vmax=1,
            center=0,
            annot=len(num_cols) <= 10,
            fmt=".2f",
            linewidths=.5,
            linecolor="white",
            cbar_kws={"shrink":.78},
            ax=ax,
        )
        ax.tick_params(labelsize=7)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PERFORMANCE
# ============================================================
st.markdown("""
<div class="perf">
<div class="perf-grid">
""", unsafe_allow_html=True)

perf_data = [
    ("R² (TRAIN)", f"{r2_train:.3f}", "Explained variance", "#5b4ee8"),
    ("R² (TEST)", f"{r2_test:.3f}", "Generalization", "#2f6fed"),
    ("ADJUSTED R²", f"{adj_r2:.3f}", "Model complexity", "#20a05a"),
    ("RMSE (TEST)", f"{rmse:.3f}", "Prediction error", "#e45567"),
    ("MAE (TEST)", f"{mae:.3f}", "Absolute error", "#ef9b22"),
    ("MODEL", "Linear", "OLS regression", "#6957e8"),
]

for label, value, note, color in perf_data:
    st.markdown(f"""
    <div class="perf-cell">
      <div class="perf-label">{label}</div>
      <div class="perf-value" style="color:{color};">{value}</div>
      <div class="perf-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# ============================================================
# EXECUTIVE INSIGHTS + TOP DRIVERS
# ============================================================
st.write("")
i1, i2 = st.columns([1.2, .8])

with i1:
    stability_gap = abs(r2_train - r2_test)
    stability_text = (
        "The train/test performance is closely aligned, indicating reasonable generalization."
        if stability_gap < .10
        else "There is a noticeable train/test performance gap; investigate possible overfitting."
    )

    st.markdown(f"""
    <div class="card">
      <div class="card-head">
        <div class="card-title">Executive Takeaway</div>
        <div class="card-sub">Decision-oriented interpretation</div>
      </div>
      <div class="insight">
        <b>Model fit:</b> The model explains <b>{r2_test:.1%}</b> of the variation in
        test-set sales.
      </div>
      <div class="insight">
        <b>Model stability:</b> {stability_text}
      </div>
      <div class="insight">
        <b>Business use:</b> Use the model to identify directional sales drivers and
        support what-if analysis; do not interpret coefficients as causal effects without
        additional business validation.
      </div>
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown("""
    <div class="card">
      <div class="card-head">
        <div class="card-title">Top Model Drivers</div>
        <div class="card-sub">By absolute coefficient</div>
      </div>
    """, unsafe_allow_html=True)

    coef = pd.DataFrame({
        "Feature": final_model.params.index,
        "Coefficient": final_model.params.values,
    })
    coef = coef[coef.Feature != "const"].copy()
    coef["Abs"] = coef["Coefficient"].abs()
    top = coef.sort_values("Abs", ascending=False).head(5)
    max_abs = max(top["Abs"].max(), 1e-9) if len(top) else 1

    for _, row in top.iterrows():
        width = 100 * row["Abs"] / max_abs
        st.markdown(f"""
        <div class="driver">
          <div class="driver-row">
            <div class="driver-name">{row['Feature']}</div>
            <div class="driver-val">{row['Coefficient']:+.3f}</div>
          </div>
          <div class="driver-track"><div class="driver-fill" style="width:{width:.1f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# DETAILED TABS
# ============================================================
tabs = st.tabs([
    "Overview",
    "Data Exploration",
    "Data Preparation",
    "Model Building",
    "Model Evaluation",
    "Diagnostics",
    "Make Prediction",
])

with tabs[0]:
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="card"><div class="card-title">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="card-title">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(include="all").T, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Data Exploration")
    feature = st.selectbox("Numeric feature", num_cols)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.histplot(df[feature].dropna(), kde=True, bins=25, color="#7968e8", ax=ax)
    ax.set_title(f"Distribution of {feature}", fontweight="bold")
    make_chart_axes(ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    corr_target = (
        df[num_cols].corr()[target]
        .drop(target)
        .sort_values(key=np.abs, ascending=False)
        .to_frame("Correlation with target")
    )
    st.dataframe(corr_target, use_container_width=True)

with tabs[2]:
    st.subheader("Data Preparation")
    st.info("Numeric missing values are median-imputed. Categorical values are one-hot encoded with drop_first=True.")
    st.write(f"**Training rows:** {len(X_train):,}  |  **Test rows:** {len(X_test):,}")
    st.dataframe(X.head(10), use_container_width=True)

with tabs[3]:
    st.subheader("Model Building")
    st.write(f"Final OLS model uses **{max(len(selected)-1, 0)} predictors** after backward elimination.")
    coef_view = coef.drop(columns="Abs").sort_values("Coefficient", key=np.abs, ascending=False)
    st.dataframe(coef_view, use_container_width=True, hide_index=True)
    with st.expander("View OLS statistical summary"):
        st.text(final_model.summary())

with tabs[4]:
    st.subheader("Model Evaluation")
    actual_pred = pd.DataFrame({"Actual": y_test, "Predicted": test_pred})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=actual_pred, x="Actual", y="Predicted", color="#6957e8", s=45, ax=ax)
    low = min(actual_pred.min())
    high = max(actual_pred.max())
    ax.plot([low, high], [low, high], "--", color="#e45567", linewidth=2)
    ax.set_title("Actual vs Predicted Sales", fontweight="bold")
    make_chart_axes(ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tabs[5]:
    st.subheader("Diagnostics")
    residuals = y_train - train_pred
    d1, d2 = st.columns(2)

    with d1:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(x=train_pred, y=residuals, color="#6957e8", ax=ax)
        ax.axhline(0, linestyle="--", color="#e45567")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted", fontweight="bold")
        make_chart_axes(ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with d2:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(residuals, kde=True, color="#2f6fed", ax=ax)
        ax.set_title("Residual Distribution", fontweight="bold")
        make_chart_axes(ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.expander("Variance Inflation Factor"):
        xv = X_train.drop(columns=["const"], errors="ignore")
        vif_rows = []
        for idx, col in enumerate(xv.columns):
            try:
                value = variance_inflation_factor(xv.values, idx)
            except Exception:
                value = np.nan
            vif_rows.append({"Feature": col, "VIF": value})
        st.dataframe(
            pd.DataFrame(vif_rows).sort_values("VIF", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

with tabs[6]:
    st.subheader("Make Prediction")
    st.caption("Enter a hypothetical firm's attributes and generate a sales estimate.")

    raw_input = {}
    input_cols = st.columns(3)

    for i, col in enumerate(X_raw.columns):
        with input_cols[i % 3]:
            if pd.api.types.is_numeric_dtype(X_raw[col]):
                raw_input[col] = st.number_input(
                    col,
                    value=float(X_raw[col].median()),
                    help=f"Typical observed value: {X_raw[col].median():.3f}",
                )
            else:
                opts = sorted(X_raw[col].dropna().unique().tolist())
                raw_input[col] = st.selectbox(col, opts)

    if st.button("✨ Generate Sales Prediction", type="primary"):
        new = pd.DataFrame([raw_input])

        for c in new.columns:
            if pd.api.types.is_numeric_dtype(X_raw[c]):
                new[c] = pd.to_numeric(new[c], errors="coerce").fillna(X_raw[c].median())

        combined = pd.concat([X_raw, new], ignore_index=True)
        encoded = pd.get_dummies(combined, drop_first=True).tail(1)
        encoded = sm.add_constant(encoded, has_constant="add")
        encoded = encoded.reindex(columns=X.columns, fill_value=0).astype(float)

        prediction = final_model.predict(encoded[selected]).iloc[0]

        st.markdown(f"""
        <div class="prediction">
          <div class="prediction-label">Predicted {target}</div>
          <div class="prediction-value">{prediction:,.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Input Summary")
        st.dataframe(new, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    Firm Sales Prediction Model · Linear Regression · Executive Analytics Demo
</div>
""", unsafe_allow_html=True)
