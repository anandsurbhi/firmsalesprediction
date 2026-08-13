"""
Firm-Level Sales Regression Explorer
=====================================
A polished, interactive Streamlit UI for the EDA + Linear Regression workflow
originally written as a Jupyter/Colab notebook.

Run with:
    pip install streamlit pandas numpy scikit-learn statsmodels seaborn matplotlib scipy --break-system-packages
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import streamlit as st

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"] = "none"
plt.rcParams["savefig.facecolor"] = "none"

st.set_page_config(
    page_title="Sales Intelligence | Regression Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom CSS — makes the default Streamlit look a bit more "designed"
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --primary: #6d5dfc;
            --primary-2: #8b7cff;
            --accent: #00c2a8;
            --pink: #ff5ca8;
            --text: #172033;
            --muted: #667085;
            --card: rgba(255,255,255,.92);
            --border: rgba(108,99,255,.14);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 5% 0%, rgba(109,93,252,.12), transparent 28%),
                radial-gradient(circle at 95% 5%, rgba(255,92,168,.10), transparent 25%),
                linear-gradient(180deg, #f8f9ff 0%, #f4f6fb 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #171a2b 0%, #222642 100%);
            border-right: 0;
        }

        section[data-testid="stSidebar"] * {
            color: #f8f9ff !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] input {
            background: rgba(255,255,255,.09) !important;
            border: 1px solid rgba(255,255,255,.12) !important;
        }

        .hero {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, #5146d8 0%, #7567f5 48%, #c653c9 100%);
            padding: 2.5rem 2.4rem;
            border-radius: 24px;
            margin: .4rem 0 1.7rem;
            box-shadow: 0 18px 45px rgba(81,70,216,.24);
        }

        .hero:after {
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            right: -70px;
            top: -110px;
            border-radius: 50%;
            background: rgba(255,255,255,.13);
        }

        .hero h1 {
            position: relative;
            z-index: 1;
            color: white;
            font-weight: 800;
            font-size: 2.25rem;
            letter-spacing: -.04em;
            margin: 0;
        }

        .hero p {
            position: relative;
            z-index: 1;
            color: rgba(255,255,255,.88);
            margin: .55rem 0 0;
            font-size: 1rem;
            max-width: 820px;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text);
            margin: .6rem 0 1rem;
        }

        .glass-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 28px rgba(31,41,55,.07);
            backdrop-filter: blur(12px);
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.92);
            border: 1px solid rgba(108,99,255,.13);
            border-radius: 18px;
            padding: 1rem 1rem .75rem;
            box-shadow: 0 7px 22px rgba(31,41,55,.06);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 800;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255,255,255,.7);
            padding: 6px;
            border-radius: 14px;
            border: 1px solid rgba(108,99,255,.10);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 9px 17px;
            font-weight: 700;
            color: #667085;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #6d5dfc, #8b7cff) !important;
            color: white !important;
        }

        .stButton > button, .stDownloadButton > button {
            border: 0;
            border-radius: 11px;
            padding: .62rem 1.35rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6d5dfc, #8b7cff);
            color: white;
            box-shadow: 0 8px 20px rgba(109,93,252,.24);
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 11px 25px rgba(109,93,252,.32);
        }

        .prediction-banner {
            background: linear-gradient(135deg, #00a98f 0%, #00c2a8 52%, #42d6b8 100%);
            border-radius: 18px;
            padding: 1.45rem 1.6rem;
            color: white;
            font-weight: 800;
            font-size: 1.45rem;
            box-shadow: 0 14px 32px rgba(0,194,168,.22);
            margin: .7rem 0 1.2rem;
        }

        .stAlert {
            border-radius: 14px;
        }

        div[data-baseweb="input"],
        div[data-baseweb="select"] {
            border-radius: 10px !important;
        }

        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Hide Streamlit's default footer/menu for a cleaner demo look */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Helper functions (ported from the notebook)
# --------------------------------------------------------------------------


def histogram_boxplot(data, feature, figsize=(10, 6), kde=False, bins=None):
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="#A084EE")
    if bins:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, color="#6C63FF")
    else:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, color="#6C63FF")
    ax_hist2.axvline(data[feature].mean(), color="#38ef7d", linestyle="--", label="mean")
    ax_hist2.axvline(data[feature].median(), color="#FF6FD8", linestyle="-", label="median")
    ax_hist2.legend()
    return f2


def labeled_barplot(data, feature, figsize=(6, 6), perc=False, n=None):
    total = len(data[feature])
    fig, ax = plt.subplots(figsize=figsize)
    plt.xticks(rotation=90)
    sns.countplot(
        data=data, x=feature, order=data[feature].value_counts().index[:n], ax=ax, color="#6C63FF"
    )
    for p in ax.patches:
        label = "{:.1f}%".format(100 * p.get_height() / total) if perc else int(p.get_height())
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(label, (x, y), ha="center", va="center", size=10, xytext=(0, 5), textcoords="offset points")
    return fig


def mape_score(targets, predictions):
    return np.mean(np.abs(targets - predictions) / targets) * 100


def model_performance_regression(model, predictors, target):
    pred = model.predict(predictors)
    rmse = np.sqrt(mean_squared_error(target, pred))
    mae = mean_absolute_error(target, pred)
    mape = mape_score(target, pred)
    return pd.DataFrame({"RMSE": [rmse], "MAE": [mae], "MAPE": [mape]})


def checking_vif(predictors):
    vif = pd.DataFrame()
    vif["feature"] = predictors.columns
    vif["VIF"] = [
        variance_inflation_factor(predictors.values, i) for i in range(len(predictors.columns))
    ]
    return vif


def backward_elimination(x_train, y_train, p_threshold=0.05):
    cols = x_train.columns.tolist()
    while len(cols) > 0:
        x_aux = x_train[cols]
        model = sm.OLS(y_train, x_aux).fit()
        p_values = model.pvalues
        max_p_value = max(p_values)
        feature_with_p_max = p_values.idxmax()
        if max_p_value > p_threshold:
            cols.remove(feature_with_p_max)
        else:
            break
    return cols


def build_encoded_row(raw_row: dict, X_raw_reference: pd.DataFrame, encoded_columns: list) -> pd.DataFrame:
    """Take a dict of raw (pre-dummy) feature values and encode it exactly like the
    training pipeline (add_constant + get_dummies(drop_first=True)), then reindex to
    match the columns the model was actually trained on."""
    new_row_raw = pd.DataFrame([raw_row])
    combined = pd.concat([X_raw_reference, new_row_raw], ignore_index=True)
    combined_enc = sm.add_constant(combined)
    combined_enc = pd.get_dummies(combined_enc, drop_first=True)
    combined_enc = combined_enc.astype(float)
    combined_enc = combined_enc.reindex(columns=encoded_columns, fill_value=0.0)
    return combined_enc.tail(1)


# --------------------------------------------------------------------------
# Sidebar: data loading & settings
# --------------------------------------------------------------------------

st.sidebar.markdown("## ⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload firm-level CSV data", type=["csv"])

use_sample = False
if uploaded_file is None:
    use_sample = st.sidebar.checkbox("Use a small built-in sample dataset instead", value=True)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif use_sample:
    rng = np.random.default_rng(1)
    n = 200
    data = pd.DataFrame(
        {
            "sales": rng.normal(6, 2, n).clip(0.1),
            "capital": rng.exponential(500, n),
            "patents": rng.poisson(20, n),
            "randd": rng.exponential(300, n),
            "employment": rng.exponential(10, n),
            "sp500": rng.choice(["yes", "no"], n, p=[0.3, 0.7]),
            "tobinq": rng.exponential(2, n).clip(0.1, 20),
            "value": rng.exponential(2000, n),
            "institutions": rng.uniform(0, 90, n),
        }
    )
else:
    st.markdown(
        '<div class="hero"><h1>📈 Firm-Level Sales Regression Explorer</h1>'
        '<p>Upload a CSV file in the sidebar (or tick "Use a small built-in sample dataset") to get started.</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()

target_col = st.sidebar.selectbox(
    "🎯 Target (dependent) variable",
    options=data.columns.tolist(),
    index=data.columns.get_loc("sales") if "sales" in data.columns else 0,
)

test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=1, step=1)
p_threshold = st.sidebar.slider("Backward-elimination p-value threshold", 0.01, 0.20, 0.05, 0.01)
drop_na_target = st.sidebar.checkbox(f"Drop rows with missing '{target_col}'", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "This app mirrors the EDA → cleaning → OLS regression → diagnostics "
    "workflow of the original notebook, wrapped in a polished, interactive UI."
)

# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------

st.markdown(
    '<div class="hero"><h1>📈 Firm-Level Sales Regression Explorer</h1>'
    '<p>Explore your data, fit an OLS model, check diagnostics, and predict '
    'sales for any custom combination of inputs — all in one place.</p></div>',
    unsafe_allow_html=True,
)

# Compact workflow indicator for presentation/demo use
st.markdown(
    '<div style="display:flex;gap:10px;flex-wrap:wrap;margin:-.4rem 0 1.2rem;">'
    '<span style="background:#eeeaff;color:#5146d8;padding:7px 12px;border-radius:999px;font-weight:700;">01 · Explore</span>'
    '<span style="background:#e8fbf7;color:#008f7b;padding:7px 12px;border-radius:999px;font-weight:700;">02 · Prepare</span>'
    '<span style="background:#fff0f7;color:#c53f7f;padding:7px 12px;border-radius:999px;font-weight:700;">03 · Model</span>'
    '<span style="background:#eef4ff;color:#3563c7;padding:7px 12px;border-radius:999px;font-weight:700;">04 · Validate</span>'
    '<span style="background:#ecfbf4;color:#16794c;padding:7px 12px;border-radius:999px;font-weight:700;">05 · Predict</span>'
    '</div>',
    unsafe_allow_html=True,
)

df = data.copy()
if drop_na_target and target_col in df.columns:
    df = df.dropna(subset=[target_col])

tab_overview, tab_eda, tab_prep, tab_model, tab_diag, tab_predict = st.tabs(
    ["🏠 Overview", "🔍 EDA", "🧹 Preprocessing", "📐 Model", "🩺 Diagnostics", "🔮 Predict"]
)

# --------------------------------------------------------------------------
# Overview tab
# --------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{data.shape[0]:,}")
    c2.metric("Columns", f"{data.shape[1]:,}")
    c3.metric("Missing values", f"{int(data.isnull().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Preview")
    n_rows = st.slider("Rows to preview", 5, 50, 10)
    st.dataframe(data.head(n_rows), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Data types")
        info_df = pd.DataFrame(
            {"dtype": data.dtypes.astype(str), "non-null": data.notnull().sum(), "nulls": data.isnull().sum()}
        )
        st.dataframe(info_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Descriptive statistics")
        st.dataframe(data.describe().T, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Unique value counts")
    st.dataframe(data.nunique().rename("unique values"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# EDA tab
# --------------------------------------------------------------------------
with tab_eda:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Univariate: Histogram + Boxplot")
    if numeric_cols:
        feat = st.selectbox("Numeric feature", numeric_cols, key="hist_feat")
        kde = st.checkbox("Show KDE curve", value=False)
        fig = histogram_boxplot(df, feat, kde=kde)
        st.pyplot(fig, transparent=True)
        plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    if cat_cols:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Categorical feature distribution")
        cfeat = st.selectbox("Categorical feature", cat_cols, key="bar_feat")
        perc = st.checkbox("Show percentages", value=True)
        fig2 = labeled_barplot(df, cfeat, perc=perc)
        st.pyplot(fig2, transparent=True)
        plt.close(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    if cat_cols and numeric_cols:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Numeric feature by category")
        c1, c2 = st.columns(2)
        with c1:
            x_cat = st.selectbox("Category (x)", cat_cols, key="box_x")
        with c2:
            y_num = st.selectbox("Numeric (y)", numeric_cols, key="box_y")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=df, x=x_cat, y=y_num, ax=ax3, palette="cool")
        st.pyplot(fig3, transparent=True)
        plt.close(fig3)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Correlation heatmap")
    if len(numeric_cols) > 1:
        fig4, ax4 = plt.subplots(figsize=(min(15, 1.2 * len(numeric_cols)), 7))
        sns.heatmap(df[numeric_cols].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral", ax=ax4)
        st.pyplot(fig4, transparent=True)
        plt.close(fig4)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Outlier scan (boxplots)")
    if numeric_cols:
        n_cols_grid = 4
        n_rows_grid = int(np.ceil(len(numeric_cols) / n_cols_grid))
        fig5, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(15, 3 * n_rows_grid))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(numeric_cols):
            axes[i].boxplot(df[col].dropna(), whis=1.5)
            axes[i].set_title(col)
        for j in range(len(numeric_cols), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        st.pyplot(fig5, transparent=True)
        plt.close(fig5)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Preprocessing tab
# --------------------------------------------------------------------------
with tab_prep:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Feature matrix construction")
    st.markdown(
        f"- Target: **{target_col}**\n"
        "- An intercept column is added.\n"
        "- Categorical columns are one-hot encoded (`drop_first=True`)."
    )

    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in data.")
        st.stop()

    X_raw = df.drop([target_col], axis=1)
    y = df[target_col]

    X = sm.add_constant(X_raw)
    X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)

    st.dataframe(X.head(), use_container_width=True)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )

    c1, c2 = st.columns(2)
    c1.metric("Train rows", x_train.shape[0])
    c2.metric("Test rows", x_test.shape[0])
    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state["x_train"] = x_train
    st.session_state["x_test"] = x_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["X_columns"] = X.columns.tolist()
    st.session_state["X_raw"] = X_raw

# --------------------------------------------------------------------------
# Model tab
# --------------------------------------------------------------------------
with tab_model:
    if "x_train" not in st.session_state:
        st.warning("Please visit the Preprocessing tab first.")
        st.stop()

    x_train = st.session_state["x_train"]
    x_test = st.session_state["x_test"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Full model (all features)")
    olsmodel1 = sm.OLS(y_train, x_train).fit()
    with st.expander("Show OLS summary (full model)"):
        st.text(olsmodel1.summary())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Training performance**")
        st.dataframe(model_performance_regression(olsmodel1, x_train, y_train))
    with c2:
        st.markdown("**Test performance**")
        st.dataframe(model_performance_regression(olsmodel1, x_test, y_test))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Multicollinearity (VIF)")
    try:
        vif_df = checking_vif(x_train)
        st.dataframe(vif_df.sort_values("VIF", ascending=False), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute VIF: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Backward feature elimination")
    st.caption(f"Iteratively drops the feature with the highest p-value while it exceeds {p_threshold:.2f}.")
    selected_features = backward_elimination(x_train, y_train, p_threshold=p_threshold)
    st.write("Selected features:", selected_features)

    x_train2 = x_train[selected_features]
    x_test2 = x_test[selected_features]

    olsmodel2 = sm.OLS(y_train, x_train2).fit()
    with st.expander("Show OLS summary (reduced model)"):
        st.text(olsmodel2.summary())

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Training performance (reduced model)**")
        st.dataframe(model_performance_regression(olsmodel2, x_train2, y_train))
    with c4:
        st.markdown("**Test performance (reduced model)**")
        st.dataframe(model_performance_regression(olsmodel2, x_test2, y_test))
    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state["olsmodel_final"] = olsmodel2
    st.session_state["olsmodel_full"] = olsmodel1
    st.session_state["x_train2"] = x_train2
    st.session_state["x_test2"] = x_test2
    st.session_state["selected_features"] = selected_features

# --------------------------------------------------------------------------
# Diagnostics tab
# --------------------------------------------------------------------------
with tab_diag:
    if "olsmodel_final" not in st.session_state:
        st.warning("Please visit the Model tab first.")
        st.stop()

    model = st.session_state["olsmodel_final"]
    x_train2 = st.session_state["x_train2"]
    y_train = st.session_state["y_train"]

    df_pred = pd.DataFrame(
        {
            "Actual Values": y_train,
            "Fitted Values": model.fittedvalues,
            "Residuals": model.resid,
        }
    )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Fitted vs residuals")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.residplot(data=df_pred, x="Fitted Values", y="Residuals", color="#A084EE", lowess=True, ax=ax)
    ax.set_title("Fitted vs Residual plot")
    st.pyplot(fig, transparent=True)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Residual distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df_pred, x="Residuals", kde=True, ax=ax2, color="#6C63FF")
        ax2.set_title("Normality of residuals")
        st.pyplot(fig2, transparent=True)
        plt.close(fig2)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Q-Q plot")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        (osm, osr), (slope, intercept, r) = stats.probplot(df_pred["Residuals"], dist="norm")
        ax3.scatter(osm, osr, s=10, color="#6C63FF")
        ax3.plot(osm, slope * osm + intercept, color="#FF6FD8")
        ax3.set_title("Q-Q plot")
        st.pyplot(fig3, transparent=True)
        plt.close(fig3)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Statistical tests")
    shapiro_stat, shapiro_p = stats.shapiro(df_pred["Residuals"])
    c3, c4 = st.columns(2)
    c3.metric("Shapiro-Wilk statistic", f"{shapiro_stat:.4f}")
    c4.metric("Shapiro-Wilk p-value", f"{shapiro_p:.4f}")

    try:
        gq_stat, gq_p, _ = sms.het_goldfeldquandt(df_pred["Residuals"], x_train2)
        c5, c6 = st.columns(2)
        c5.metric("Goldfeld-Quandt F", f"{gq_stat:.4f}")
        c6.metric("Goldfeld-Quandt p-value", f"{gq_p:.4f}")
    except Exception as e:
        st.warning(f"Could not run Goldfeld-Quandt test: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Residuals table")
    st.dataframe(df_pred.head(20), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Predict tab
# --------------------------------------------------------------------------
with tab_predict:
    if "olsmodel_final" not in st.session_state:
        st.warning("Please visit the Model tab first.")
        st.stop()

    model_final = st.session_state["olsmodel_final"]
    model_full = st.session_state["olsmodel_full"]
    selected_features = st.session_state["selected_features"]
    X_columns = st.session_state["X_columns"]
    X_raw_reference = st.session_state["X_raw"]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔮 Predict sales for a custom firm")
    st.caption(
        "Fill in values for **every original column** in your dataset (numeric inputs for "
        "numeric columns, dropdowns for categorical ones). You can also add brand-new custom "
        "columns below — useful for what-if experiments even if they weren't used by the final model."
    )

    numeric_raw_cols = X_raw_reference.select_dtypes(include=np.number).columns.tolist()
    cat_raw_cols = X_raw_reference.select_dtypes(exclude=np.number).columns.tolist()

    raw_input = {}

    if numeric_raw_cols:
        st.markdown("#### 🔢 Numeric columns")
        n_grid = 3
        cols = st.columns(n_grid)
        for i, col in enumerate(numeric_raw_cols):
            default_val = float(X_raw_reference[col].mean())
            col_min = float(X_raw_reference[col].min())
            col_max = float(X_raw_reference[col].max())
            with cols[i % n_grid]:
                raw_input[col] = st.number_input(
                    col,
                    value=round(default_val, 3),
                    help=f"Observed range: {col_min:.2f} – {col_max:.2f}",
                    key=f"num_{col}",
                )

    if cat_raw_cols:
        st.markdown("#### 🏷️ Categorical columns")
        n_grid = 3
        cols = st.columns(n_grid)
        for i, col in enumerate(cat_raw_cols):
            options = sorted(X_raw_reference[col].dropna().unique().tolist())
            with cols[i % n_grid]:
                raw_input[col] = st.selectbox(col, options=options, key=f"cat_{col}")

    with st.expander("➕ Add custom / extra columns"):
        st.caption(
            "Add any additional column name & value here. If it matches a column the model "
            "was trained on it will be used; otherwise it's ignored by the model but still "
            "shown in the input summary below."
        )
        n_custom = st.number_input("How many custom columns to add?", min_value=0, max_value=10, value=0, step=1)
        for i in range(int(n_custom)):
            c1, c2 = st.columns(2)
            with c1:
                custom_name = st.text_input(f"Custom column {i+1} name", key=f"custom_name_{i}")
            with c2:
                custom_value = st.text_input(f"Custom column {i+1} value", key=f"custom_value_{i}")
            if custom_name:
                try:
                    raw_input[custom_name] = float(custom_value)
                except (TypeError, ValueError):
                    raw_input[custom_name] = custom_value

    predict_col1, predict_col2 = st.columns([1, 2])
    with predict_col1:
        model_choice = st.radio(
            "Model to use",
            options=["Reduced model (backward elimination)", "Full model (all features)"],
            index=0,
        )
    with predict_col2:
        st.write("")
        do_predict = st.button("✨ Predict Sales", type="primary")

    if do_predict:
        try:
            encoded_row = build_encoded_row(raw_input, X_raw_reference, X_columns)
            if model_choice.startswith("Reduced"):
                row_for_model = encoded_row[selected_features]
                prediction = model_final.predict(row_for_model)
            else:
                prediction = model_full.predict(encoded_row)

            st.markdown(
                f'<div class="prediction-banner">📈 Predicted {target_col}: '
                f'{prediction.iloc[0]:,.4f}</div>',
                unsafe_allow_html=True,
            )

            with st.expander("Show raw input & encoded feature row used"):
                st.markdown("**Raw input values**")
                st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)
                st.markdown("**Encoded row (as fed to the model)**")
                st.dataframe(encoded_row, use_container_width=True)
        except Exception as e:
            st.error(f"Could not compute prediction: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
