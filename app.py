
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

:root{
 --bg:#f6f7fb;
 --surface:#ffffff;
 --navy:#172554;
 --ink:#1f2937;
 --muted:#6b7280;
 --line:#e9ebf2;
 --purple:#6657d9;
 --purple-soft:#f0edff;
 --blue:#3675e8;
 --green:#19965a;
 --green-soft:#eaf8f0;
 --amber:#df941f;
 --amber-soft:#fff5e6;
 --red:#d95768;
}

*{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg);color:var(--ink);}
#MainMenu,header,footer{visibility:hidden;}

section[data-testid="stSidebar"]{
 background:#fff;
 border-right:1px solid var(--line);
}
section[data-testid="stSidebar"]>div{padding:1rem .9rem;}

.brand{
 padding:.35rem .2rem 1rem;
 border-bottom:1px solid var(--line);
 margin-bottom:1rem;
}
.brand-row{display:flex;align-items:center;gap:.65rem;}
.brand-icon{
 width:38px;height:38px;border-radius:12px;
 display:flex;align-items:center;justify-content:center;
 background:var(--purple-soft);color:var(--purple);
 font-size:1.1rem;
}
.brand-title{font-size:1.03rem;font-weight:800;color:var(--navy);}
.brand-sub{font-size:.69rem;color:#8992a3;margin-top:.18rem;}

.side-heading{
 color:#7568d9;
 font-size:.66rem;font-weight:800;
 letter-spacing:.1em;text-transform:uppercase;
 margin:1rem 0 .45rem;
}
.side-card{
 background:#fafaff;border:1px solid #ece9fb;
 border-radius:12px;padding:.65rem .75rem;
}
.side-row{
 display:flex;justify-content:space-between;
 color:#737d8e;font-size:.72rem;padding:.24rem 0;
}
.side-row b{color:#263249;}

.hero{
 background:linear-gradient(135deg,#fff 0%,#f3f1ff 100%);
 border:1px solid #e7e3fb;
 border-radius:22px;
 padding:1.45rem 1.7rem;
 box-shadow:0 10px 28px rgba(27,35,65,.055);
 margin-bottom:.85rem;
}
.hero-grid{
 display:flex;align-items:center;justify-content:space-between;
 gap:1rem;
}
.hero h1{
 margin:0;color:var(--navy);
 font-size:2rem;font-weight:800;
 letter-spacing:-.045em;line-height:1.1;
}
.hero h1 span{color:var(--purple);}
.hero p{
 margin:.5rem 0 0;max-width:720px;
 color:#6d7789;font-size:.87rem;line-height:1.5;
}
.hero-badge{
 display:inline-flex;align-items:center;gap:.35rem;
 background:#fff;border:1px solid #e5e1fa;
 border-radius:999px;padding:.42rem .7rem;
 color:var(--purple);font-size:.68rem;font-weight:800;
 margin-bottom:.65rem;
}
.hero-visual{min-width:180px;text-align:center;}
.mini-bars{height:78px;display:flex;align-items:end;justify-content:center;gap:5px;}
.mini-bar{
 width:16px;border-radius:5px 5px 2px 2px;
 background:linear-gradient(180deg,#a59af0,#6657d9);
}
.hero-visual-label{font-size:.63rem;color:#7d8798;font-weight:800;margin-top:.25rem;}

.workflow{
 display:flex;align-items:center;gap:.2rem;
 background:#fff;border:1px solid var(--line);
 border-radius:15px;padding:.45rem;
 box-shadow:0 5px 18px rgba(27,35,65,.04);
 margin-bottom:.9rem;
 overflow:auto;
}
.step{
 flex:1;min-width:115px;
 display:flex;align-items:center;gap:.5rem;
 padding:.55rem .65rem;border-radius:10px;
}
.step.active{background:var(--purple-soft);}
.step-num{
 width:29px;height:29px;flex:0 0 29px;border-radius:50%;
 display:flex;align-items:center;justify-content:center;
 background:#eef1ff;color:#5e67b8;font-size:.73rem;font-weight:800;
}
.step.active .step-num{background:var(--purple);color:#fff;}
.step-title{font-size:.76rem;font-weight:800;color:#263249;}
.step-desc{font-size:.61rem;color:#8992a3;margin-top:.06rem;}
.arrow{color:#b0b7c5;font-size:1rem;}

.kpi{
 background:#fff;border:1px solid var(--line);
 border-radius:16px;padding:.8rem .9rem;
 box-shadow:0 6px 20px rgba(27,35,65,.045);
 min-height:105px;
}
.kpi-top{display:flex;align-items:center;gap:.62rem;}
.kpi-icon{
 width:39px;height:39px;border-radius:11px;
 display:flex;align-items:center;justify-content:center;
 font-size:1rem;
}
.kpi-label{font-size:.61rem;color:#7a8495;font-weight:800;letter-spacing:.05em;}
.kpi-value{font-size:1.3rem;font-weight:800;margin-top:.12rem;}
.kpi-note{font-size:.63rem;color:#919aaa;margin-top:.3rem;}

.card{
 background:#fff;border:1px solid var(--line);
 border-radius:16px;padding:.9rem 1rem;
 box-shadow:0 6px 20px rgba(27,35,65,.04);
 margin-bottom:.85rem;
}
.card-head{
 display:flex;align-items:center;justify-content:space-between;
 gap:.6rem;margin-bottom:.3rem;
}
.card-title{
 color:var(--navy);font-size:.76rem;font-weight:800;
 letter-spacing:.06em;text-transform:uppercase;
}
.card-sub{color:#98a0ae;font-size:.62rem;}

.insight{
 background:#f8f7ff;border:1px solid #e8e4fb;
 border-left:3px solid var(--purple);
 border-radius:10px;padding:.65rem .8rem;
 color:#626d7e;font-size:.73rem;line-height:1.45;
 margin:.35rem 0;
}
.insight b{color:var(--navy);}

.perf-wrap{
 background:#fff;border:1px solid var(--line);
 border-radius:16px;padding:.35rem;
 box-shadow:0 6px 20px rgba(27,35,65,.04);
 margin-bottom:.9rem;
}
.perf-cell{
 text-align:center;padding:.45rem .3rem;
 border-right:1px solid #edf0f4;
}
.perf-cell:last-child{border-right:0;}
.perf-label{font-size:.58rem;color:#7b8494;font-weight:800;letter-spacing:.04em;}
.perf-value{font-size:1.18rem;font-weight:800;margin:.12rem 0;}
.perf-note{font-size:.58rem;color:#98a0ae;}

.driver{
 background:#fff;border:1px solid var(--line);
 border-radius:11px;padding:.55rem .65rem;margin:.3rem 0;
}
.driver-row{display:flex;justify-content:space-between;align-items:center;}
.driver-name{font-size:.72rem;font-weight:700;color:#344054;}
.driver-val{font-size:.7rem;font-weight:800;color:var(--purple);}
.driver-track{height:5px;background:#eeeef5;border-radius:99px;margin-top:.4rem;overflow:hidden;}
.driver-fill{height:100%;background:linear-gradient(90deg,#a79df0,#6657d9);border-radius:99px;}

.prediction{
 background:linear-gradient(135deg,#16834b,#21a461);
 border-radius:15px;padding:1rem 1.15rem;color:#fff;
 box-shadow:0 10px 24px rgba(22,131,75,.16);
}
.prediction-label{font-size:.62rem;font-weight:800;text-transform:uppercase;opacity:.82;}
.prediction-value{font-size:1.55rem;font-weight:800;margin-top:.12rem;}

.stButton>button,.stDownloadButton>button{
 background:var(--purple);color:#fff;border:0;border-radius:9px;
 font-weight:700;box-shadow:0 6px 15px rgba(102,87,217,.16);
}
.stButton>button:hover,.stDownloadButton>button:hover{background:#5647ca;color:#fff;}

.stTabs [data-baseweb="tab-list"]{
 background:#fff;border:1px solid var(--line);
 border-radius:12px;padding:4px;gap:4px;
}
.stTabs [data-baseweb="tab"]{
 border-radius:9px;color:#667085;font-weight:700;
 font-size:.78rem;
}
.stTabs [aria-selected="true"]{
 background:var(--purple)!important;color:#fff!important;
}

div[data-testid="stMetric"]{
 background:#fff;border:1px solid var(--line);
 border-radius:12px;
}
.stDataFrame{border-radius:10px;overflow:hidden;}

.footer{
 text-align:center;color:#a0a7b4;font-size:.63rem;
 padding:1rem 0 .2rem;
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
      <div class="hero-badge">● EXECUTIVE ANALYTICS · MODEL DEMO</div>
      <h1>Firm-Level Sales <span>Regression Explorer</span></h1>
      <p>Understand sales drivers, assess model reliability and generate decision-ready
      predictions through a single executive analytics view.</p>
    </div>
    <div class="hero-visual">
      <div class="mini-bars">
        <span class="mini-bar" style="height:25px"></span>
        <span class="mini-bar" style="height:38px"></span>
        <span class="mini-bar" style="height:48px"></span>
        <span class="mini-bar" style="height:61px"></span>
        <span class="mini-bar" style="height:74px"></span>
      </div>
      <div class="hero-visual-label">SALES PERFORMANCE</div>
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
        fig, ax = plt.subplots(figsize=(8, 3.7))
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
        fig, ax = plt.subplots(figsize=(7.5, 3.7))
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
# PERFORMANCE — HORIZONTAL SCORECARD
perf_data = [
    ("R² TRAIN", f"{r2_train:.3f}", "Explained variance", "#5b4ee8"),
    ("R² TEST", f"{r2_test:.3f}", "Generalization", "#2f6fed"),
    ("ADJUSTED R²", f"{adj_r2:.3f}", "Model complexity", "#20a05a"),
    ("RMSE TEST", f"{rmse:.3f}", "Prediction error", "#d95768"),
    ("MAE TEST", f"{mae:.3f}", "Absolute error", "#df941f"),
    ("MODEL", "Linear", "OLS regression", "#6657d9"),
]

perf_cols = st.columns(6, gap="small")
for col, (label, value, note, color) in zip(perf_cols, perf_data):
    with col:
        st.markdown(f"""
        <div class="perf-wrap">
          <div class="perf-cell">
            <div class="perf-label">{label}</div>
            <div class="perf-value" style="color:{color};">{value}</div>
            <div class="perf-note">{note}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# EXECUTIVE INSIGHTS + TOP DRIVERS
# ============================================================
st.write("")
status_color = "#20a05a" if abs(r2_train-r2_test) < .10 else "#df941f"
status_text = "STABLE GENERALIZATION" if abs(r2_train-r2_test) < .10 else "REVIEW MODEL GAP"
st.markdown(
    f'<div style="display:flex;justify-content:flex-end;margin:-.35rem 0 .45rem;">'
    f'<span style="background:#fff;border:1px solid #e8ebf2;border-radius:999px;'
    f'padding:.3rem .65rem;font-size:.62rem;font-weight:800;color:{status_color};">'
    f'● {status_text}</span></div>',
    unsafe_allow_html=True
)
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
