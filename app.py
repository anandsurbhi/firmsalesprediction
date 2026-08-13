
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(
    page_title="Firm-Level Sales Regression Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# LIGHT EXECUTIVE THEME — designed to match the requested image
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: #f8f9fd;
    color: #17213d;
}

#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8eaf2;
}

section[data-testid="stSidebar"] > div {
    padding: 1.1rem 1rem;
}

.brand {
    padding: 0.4rem 0.2rem 1.15rem;
    border-bottom: 1px solid #e8eaf2;
    margin-bottom: 1rem;
}
.brand-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #172554;
}
.brand-subtitle {
    color: #7b8498;
    font-size: .76rem;
    margin-top: .2rem;
}

.side-section {
    color: #6d5dfc;
    font-weight: 800;
    font-size: .75rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin: 1.1rem 0 .55rem;
}

.side-info {
    display: flex;
    align-items: center;
    gap: .65rem;
    padding: .45rem 0;
    color: #475569;
    font-size: .8rem;
}
.side-info strong {
    display: block;
    color: #17213d;
    font-size: .9rem;
}

.hero {
    background: linear-gradient(135deg, #ffffff 0%, #f0edff 100%);
    border: 1px solid #e5e1ff;
    border-radius: 22px;
    padding: 1.75rem 2rem;
    box-shadow: 0 10px 30px rgba(28, 39, 78, .06);
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.hero:after {
    content: "";
    position: absolute;
    right: 2%;
    top: 4%;
    width: 250px;
    height: 160px;
    background:
        linear-gradient(145deg, transparent 48%, rgba(109,93,252,.11) 49%, transparent 51%),
        linear-gradient(145deg, transparent 54%, rgba(109,93,252,.08) 55%, transparent 57%);
    opacity: .9;
}
.hero h1 {
    position: relative;
    z-index: 1;
    margin: 0;
    color: #172554;
    font-size: 2.2rem;
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -.045em;
}
.hero h1 span { color: #6d5dfc; }
.hero p {
    position: relative;
    z-index: 1;
    color: #64748b;
    margin: .55rem 0 0;
    font-size: .96rem;
    max-width: 760px;
}

.workflow {
    display: flex;
    align-items: center;
    gap: .25rem;
    flex-wrap: wrap;
    background: #ffffff;
    border: 1px solid #e8eaf2;
    border-radius: 16px;
    padding: .65rem;
    box-shadow: 0 6px 20px rgba(28,39,78,.05);
    margin-bottom: 1rem;
}
.step {
    flex: 1;
    min-width: 125px;
    display: flex;
    align-items: center;
    gap: .55rem;
    padding: .65rem .8rem;
    border-radius: 12px;
}
.step.active { background: #eeeaff; }
.step-num {
    min-width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #eef2ff;
    color: #5b4ee8;
    font-weight: 800;
}
.step.active .step-num {
    background: #6d5dfc;
    color: white;
}
.step-title { font-weight: 800; color: #17213d; font-size: .88rem; }
.step-desc { color: #718096; font-size: .69rem; }
.arrow { color: #a3adbf; font-size: 1.25rem; }

.kpi {
    background: #ffffff;
    border: 1px solid #e8eaf2;
    border-radius: 16px;
    padding: 1rem;
    min-height: 116px;
    box-shadow: 0 6px 20px rgba(28,39,78,.05);
}
.kpi-top {
    display: flex;
    align-items: center;
    gap: .7rem;
}
.kpi-icon {
    width: 43px;
    height: 43px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}
.kpi-label {
    color: #475569;
    font-size: .72rem;
    font-weight: 700;
    text-transform: uppercase;
}
.kpi-value {
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: .18rem;
}
.kpi-note {
    color: #718096;
    font-size: .72rem;
    margin-top: .35rem;
}

.card {
    background: #ffffff;
    border: 1px solid #e8eaf2;
    border-radius: 17px;
    padding: 1rem 1.1rem;
    box-shadow: 0 6px 20px rgba(28,39,78,.05);
    margin-bottom: 1rem;
}
.card-title {
    color: #17213d;
    font-size: .82rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .045em;
    margin-bottom: .5rem;
}

.performance {
    background: #ffffff;
    border: 1px solid #e8eaf2;
    border-radius: 17px;
    padding: .8rem .5rem;
    box-shadow: 0 6px 20px rgba(28,39,78,.05);
    margin-top: .1rem;
}
.perf {
    text-align: center;
    padding: .45rem .7rem;
    border-right: 1px solid #edf0f5;
}
.perf:last-child { border-right: 0; }
.perf-label {
    color: #64748b;
    font-size: .68rem;
    font-weight: 700;
    text-transform: uppercase;
}
.perf-value {
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: .18rem;
}
.perf-note {
    color: #718096;
    font-size: .68rem;
    margin-top: .18rem;
}

.insight {
    background: #f5f3ff;
    border-left: 4px solid #6d5dfc;
    border-radius: 10px;
    padding: .85rem 1rem;
    color: #526174;
    font-size: .84rem;
    line-height: 1.5;
    margin-bottom: 1rem;
}
.insight strong { color: #172554; }

.prediction {
    background: linear-gradient(135deg, #15803d, #22a65a);
    color: white;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    font-weight: 800;
    font-size: 1.35rem;
    box-shadow: 0 10px 24px rgba(21,128,61,.18);
}

.stButton > button, .stDownloadButton > button {
    background: #6d5dfc;
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #5949df;
    color: white;
}

.stTabs [data-baseweb="tab-list"] {
    background: white;
    border: 1px solid #e8eaf2;
    border-radius: 14px;
    padding: 5px;
    gap: 5px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-weight: 700;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    background: #6d5dfc !important;
    color: white !important;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e8eaf2;
    border-radius: 14px;
    box-shadow: 0 5px 18px rgba(28,39,78,.05);
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def mape_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def performance(model, X, y):
    pred = model.predict(X)
    return {
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAE": mean_absolute_error(y, pred),
        "MAPE": mape_score(y, pred),
    }


def backward_elimination(X, y, threshold=.05):
    cols = list(X.columns)
    while len(cols) > 1:
        model = sm.OLS(y, X[cols]).fit()
        pvals = model.pvalues.drop(labels="const", errors="ignore")
        if len(pvals) == 0 or pvals.max() <= threshold:
            break
        cols.remove(pvals.idxmax())
    return cols


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div class="brand">
    <div class="brand-title">📊 Sales Intelligence</div>
    <div class="brand-subtitle">Firm-level regression & decision support</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="side-section">Navigation</div>', unsafe_allow_html=True)
nav = st.sidebar.radio(
    "Navigation",
    ["Overview", "Data Exploration", "Data Preparation",
     "Model Building", "Model Evaluation", "Diagnostics", "Make Prediction"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="side-section">Dataset</div>', unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("Upload firm-level CSV", type=["csv"])

use_sample = False
if uploaded is None:
    use_sample = st.sidebar.checkbox("Use built-in sample dataset", value=True)

# ============================================================
# DATA
# ============================================================
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    rng = np.random.default_rng(42)
    n = 738

    df = pd.DataFrame({
        "sales": rng.normal(0.8, .75, n),
        "sga": rng.normal(1.1, .55, n),
        "rd": rng.exponential(.7, n),
        "ad": rng.exponential(.45, n),
        "tobinq": rng.normal(1.35, .55, n).clip(.2, 5),
        "de": rng.normal(1.0, .6, n).clip(.05, 4),
        "nwc": rng.normal(.8, .4, n),
        "profmarg": rng.normal(.12, .08, n),
        "salesgrowth": rng.normal(.08, .16, n),
    })
    # Give the synthetic target a sensible relationship with drivers.
    df["sales"] = (
        .35
        + .32 * df["sga"]
        + .18 * df["rd"]
        + .14 * df["ad"]
        + .22 * df["tobinq"]
        + .10 * df["nwc"]
        + .20 * df["profmarg"]
        + .18 * df["salesgrowth"]
        + rng.normal(0, .35, n)
    )
else:
    st.markdown("""
    <div class="hero">
        <h1>Firm-Level <span>Sales Regression Explorer</span></h1>
        <p>Upload a CSV file from the sidebar to begin the analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

target_col = st.sidebar.selectbox(
    "Target variable",
    df.columns.tolist(),
    index=df.columns.tolist().index("sales") if "sales" in df.columns else 0,
)

test_size = st.sidebar.slider("Test set size", .10, .40, .20, .05)
random_state = st.sidebar.number_input("Random state", 1, 100, 42)

# Sidebar dataset summary
st.sidebar.markdown('<div class="side-section">Dataset Info</div>', unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div class="side-info">📁 <span>Observations<strong>{len(df):,}</strong></span></div>
<div class="side-info">🔢 <span>Features<strong>{len(df.columns)-1}</strong></span></div>
<div class="side-info">🎯 <span>Target<strong>{target_col}</strong></span></div>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>Firm-Level Sales <span>Regression Explorer</span></h1>
    <p>Data-driven insights to understand what drives firm sales performance,
    validate the model, and generate decision-ready predictions.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="workflow">
    <div class="step active">
        <div class="step-num">1</div>
        <div><div class="step-title">Explore</div><div class="step-desc">Understand Data</div></div>
    </div>
    <div class="arrow">→</div>
    <div class="step">
        <div class="step-num">2</div>
        <div><div class="step-title">Prepare</div><div class="step-desc">Clean & Engineer</div></div>
    </div>
    <div class="arrow">→</div>
    <div class="step">
        <div class="step-num">3</div>
        <div><div class="step-title">Model</div><div class="step-desc">Build & Train</div></div>
    </div>
    <div class="arrow">→</div>
    <div class="step">
        <div class="step-num">4</div>
        <div><div class="step-title">Validate</div><div class="step-desc">Evaluate Performance</div></div>
    </div>
    <div class="arrow">→</div>
    <div class="step">
        <div class="step-num">5</div>
        <div><div class="step-title">Predict</div><div class="step-desc">Generate Insights</div></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# KPI CARDS
# ============================================================
k1, k2, k3, k4 = st.columns(4)

missing = int(df.isna().sum().sum())

with k1:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-top">
        <div class="kpi-icon" style="background:#eeeaff;">🗄️</div>
        <div>
          <div class="kpi-label">Total Observations</div>
          <div class="kpi-value" style="color:#5b4ee8;">{len(df):,}</div>
        </div>
      </div>
      <div class="kpi-note">Rows in dataset</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-top">
        <div class="kpi-icon" style="background:#edf4ff;">☷</div>
        <div>
          <div class="kpi-label">Total Features</div>
          <div class="kpi-value" style="color:#2563eb;">{len(df.columns)-1}</div>
        </div>
      </div>
      <div class="kpi-note">Independent variables</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-top">
        <div class="kpi-icon" style="background:#eaf8ee;">🎯</div>
        <div>
          <div class="kpi-label">Target Variable</div>
          <div class="kpi-value" style="color:#16803b;">{target_col}</div>
        </div>
      </div>
      <div class="kpi-note">Dependent variable</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-top">
        <div class="kpi-icon" style="background:#fff5e7;">📅</div>
        <div>
          <div class="kpi-label">Data Quality</div>
          <div class="kpi-value" style="color:#e88900;">{missing:,}</div>
        </div>
      </div>
      <div class="kpi-note">Missing values</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

left, right = st.columns([1.15, 1])

with left:
    st.markdown('<div class="card"><div class="card-title">Target Distribution</div>', unsafe_allow_html=True)
    if target_col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 4.25))
        sns.histplot(
            df[target_col].dropna(),
            kde=True,
            bins=25,
            color="#7968e8",
            ax=ax,
            edgecolor="white",
        )
        ax.set_xlabel(target_col, fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.grid(axis="y", alpha=.18)
        sns.despine(ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><div class="card-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(7.5, 4.25))
        corr = df[numeric_cols].corr()
        sns.heatmap(
            corr,
            cmap="PuOr_r",
            vmin=-1,
            vmax=1,
            center=0,
            annot=len(numeric_cols) <= 10,
            fmt=".2f",
            linewidths=.5,
            linecolor="white",
            ax=ax,
            cbar_kws={"shrink": .8},
        )
        ax.tick_params(labelsize=7)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================
X_raw = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Basic missing-value treatment for a robust demo.
for c in X_raw.columns:
    if pd.api.types.is_numeric_dtype(X_raw[c]):
        X_raw[c] = X_raw[c].fillna(X_raw[c].median())
    else:
        X_raw[c] = X_raw[c].fillna(X_raw[c].mode().iloc[0])

y = y.fillna(y.median())

X = pd.get_dummies(X_raw, drop_first=True)
X = sm.add_constant(X, has_constant="add").astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

full_model = sm.OLS(y_train, X_train).fit()

try:
    selected = backward_elimination(X_train, y_train, .05)
    if "const" not in selected:
        selected = ["const"] + selected
    reduced_model = sm.OLS(y_train, X_train[selected]).fit()
except Exception:
    selected = list(X_train.columns)
    reduced_model = full_model

final_model = reduced_model
train_pred = final_model.predict(X_train[selected])
test_pred = final_model.predict(X_test[selected])

r2_train = final_model.rsquared
r2_test = 1 - np.sum((y_test-test_pred)**2) / np.sum((y_test-y_test.mean())**2)
adj_r2 = final_model.rsquared_adj
rmse = np.sqrt(mean_squared_error(y_test, test_pred))
mae = mean_absolute_error(y_test, test_pred)


# ============================================================
# MODEL PERFORMANCE STRIP
# ============================================================
st.markdown("""
<div class="performance">
<div style="display:grid;grid-template-columns:repeat(6,1fr);">
""", unsafe_allow_html=True)

metrics = [
    ("R² (TRAIN)", f"{r2_train:.3f}", "Variance explained", "#5b4ee8"),
    ("R² (TEST)", f"{r2_test:.3f}", "Generalization", "#2563eb"),
    ("ADJUSTED R²", f"{adj_r2:.3f}", "Predictor adjusted", "#16803b"),
    ("RMSE (TEST)", f"{rmse:.3f}", "Root mean squared error", "#e11d48"),
    ("MAE (TEST)", f"{mae:.3f}", "Mean absolute error", "#e88900"),
    ("MODEL", "Linear", "Ordinary Least Squares", "#5b4ee8"),
]

for label, value, note, color in metrics:
    st.markdown(f"""
    <div class="perf">
      <div class="perf-label">{label}</div>
      <div class="perf-value" style="color:{color};">{value}</div>
      <div class="perf-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

st.write("")


# ============================================================
# EXECUTIVE INSIGHT
# ============================================================
coef_df = pd.DataFrame({
    "Feature": final_model.params.index,
    "Coefficient": final_model.params.values,
})
coef_df = coef_df[coef_df["Feature"] != "const"]
coef_df["Abs"] = coef_df["Coefficient"].abs()
top_drivers = coef_df.sort_values("Abs", ascending=False).head(5)

driver_text = ", ".join(top_drivers["Feature"].astype(str).tolist()) if len(top_drivers) else "the available predictors"

st.markdown(f"""
<div class="insight">
<strong>Executive takeaway:</strong>
The current model explains <strong>{r2_test:.1%}</strong> of the variation in the test data.
The strongest model drivers by absolute coefficient are <strong>{driver_text}</strong>.
Use these results as directional business insights and validate causality before making investment decisions.
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS FOR THE DETAILED DEMO
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

# ---------- Overview ----------
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Data Quality Summary</div>', unsafe_allow_html=True)
        quality = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
            "Value": [
                f"{len(df):,}",
                f"{len(df.columns):,}",
                f"{missing:,}",
                f"{df.duplicated().sum():,}",
            ],
        })
        st.dataframe(quality, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- EDA ----------
with tabs[1]:
    st.subheader("Data Exploration")
    feat = st.selectbox("Select numeric feature", numeric_cols)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df[feat].dropna(), kde=True, color="#7968e8", ax=ax)
    ax.set_title(f"Distribution of {feat}", fontweight="bold")
    sns.despine()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if len(numeric_cols) > 1:
        st.markdown("### Correlation with target")
        corr_target = (
            df[numeric_cols]
            .corr()[target_col]
            .drop(target_col)
            .sort_values(key=np.abs, ascending=False)
            .to_frame("Correlation")
        )
        st.dataframe(corr_target, use_container_width=True)

# ---------- PREP ----------
with tabs[2]:
    st.subheader("Data Preparation")
    st.write("Categorical variables are one-hot encoded and numeric missing values are median-imputed.")
    st.dataframe(X.head(10), use_container_width=True)
    st.write(f"Training rows: **{len(X_train):,}** | Test rows: **{len(X_test):,}**")

# ---------- MODEL ----------
with tabs[3]:
    st.subheader("Model Building")
    st.markdown(f"**Final model:** Linear Regression using **{len(selected)-1} predictors** after backward elimination.")
    st.dataframe(
        coef_df.drop(columns="Abs").sort_values("Coefficient", key=np.abs, ascending=False),
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Show OLS statistical summary"):
        st.text(final_model.summary())

# ---------- EVALUATION ----------
with tabs[4]:
    st.subheader("Model Evaluation")
    actual_vs_pred = pd.DataFrame({"Actual": y_test, "Predicted": test_pred})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=actual_vs_pred, x="Actual", y="Predicted", color="#6d5dfc", ax=ax)
    mn = min(actual_vs_pred.min())
    mx = max(actual_vs_pred.max())
    ax.plot([mn, mx], [mn, mx], "--", color="#ef4444")
    ax.set_title("Actual vs Predicted Sales", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ---------- DIAGNOSTICS ----------
with tabs[5]:
    st.subheader("Regression Diagnostics")
    residuals = y_train - train_pred

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(x=train_pred, y=residuals, color="#6d5dfc", ax=ax)
        ax.axhline(0, ls="--", color="#ef4444")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(residuals, kde=True, color="#2563eb", ax=ax)
        ax.set_title("Residual Distribution", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.expander("Variance Inflation Factor"):
        vif = []
        Xv = X_train.drop(columns=["const"], errors="ignore")
        if Xv.shape[1] > 0:
            for i, col in enumerate(Xv.columns):
                try:
                    vif.append((col, variance_inflation_factor(Xv.values, i)))
                except Exception:
                    vif.append((col, np.nan))
        st.dataframe(pd.DataFrame(vif, columns=["Feature", "VIF"]).sort_values("VIF", ascending=False),
                     hide_index=True, use_container_width=True)

# ---------- PREDICT ----------
with tabs[6]:
    st.subheader("Make Prediction")
    st.caption("Enter values for a new firm. The app encodes the inputs using the same feature structure as the trained model.")

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

    if st.button("✨ Predict Sales", type="primary"):
        new = pd.DataFrame([raw_input])
        for c in new.columns:
            if pd.api.types.is_numeric_dtype(X_raw[c]):
                new[c] = pd.to_numeric(new[c], errors="coerce").fillna(X_raw[c].median())
        new_enc = pd.get_dummies(
            pd.concat([X_raw, new], ignore_index=True),
            drop_first=True
        ).tail(1)
        new_enc = sm.add_constant(new_enc, has_constant="add")
        new_enc = new_enc.reindex(columns=X.columns, fill_value=0).astype(float)
        prediction = final_model.predict(new_enc[selected]).iloc[0]

        st.markdown(
            f'<div class="prediction">📈 Predicted {target_col}: {prediction:,.4f}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Input Summary")
        st.dataframe(new, hide_index=True, use_container_width=True)


st.markdown(
    '<div style="text-align:center;color:#94a3b8;font-size:.75rem;padding:1.2rem 0;">'
    'Firm Sales Prediction Model • Linear Regression • Executive Analytics Demo'
    '</div>',
    unsafe_allow_html=True,
)
