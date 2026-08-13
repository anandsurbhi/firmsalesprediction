import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Firm Sales Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# LIGHT PROFESSIONAL THEME — NO ICONS
# ============================================================
st.markdown("""
<style>
/* ============================================================
   COLOURFUL PROFESSIONAL LIGHT THEME
   ============================================================ */

.stApp {
    background: linear-gradient(135deg, #F7FAFF 0%, #FDFBFF 100%);
    color: #172033;
}

.block-container {
    padding: 0.4rem 2.2rem 2rem 2.2rem;
    max-width: 1500px;
}

/* ================= TOP NAVIGATION ================= */

.topbar {
    background: linear-gradient(90deg, #FFFFFF 0%, #F4F8FF 100%);
    border: 1px solid #DCE6F5;
    border-radius: 14px;
    padding: 12px 18px;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(47, 107, 186, 0.08);
}

.brand {
    font-size: 21px;
    font-weight: 800;
    color: #173B6C;
    letter-spacing: -0.4px;
}

.subbrand {
    color: #718096;
    font-size: 12px;
    margin-top: 3px;
}

/* ================= BOOKMARK NAVIGATION ================= */

.bookmark-label {
    color: #526581;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 3px 0 8px 2px;
}

.stButton > button {
    border: 1px solid #D5DEED;
    background: #FFFFFF;
    color: #38506F;
    border-radius: 9px;
    font-weight: 650;
    min-height: 38px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #4F86C6;
    color: #173B6C;
    background: #EDF5FF;
    box-shadow: 0 3px 8px rgba(79, 134, 198, 0.12);
}

/* ================= KPI CARDS ================= */

.kpi-card {
    background: #FFFFFF;
    border: 1px solid #DDE6F2;
    border-radius: 13px;
    padding: 18px 20px;
    min-height: 112px;
    box-shadow: 0 4px 12px rgba(31, 55, 90, 0.055);
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #3B82F6, #8B5CF6);
}

.kpi-label {
    color: #667085;
    font-size: 11px;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.kpi-value {
    color: #173B6C;
    font-size: 28px;
    font-weight: 800;
    margin-top: 7px;
}

.kpi-caption {
    color: #8A94A6;
    font-size: 11px;
    margin-top: 4px;
}

/* ================= SECTION CARDS ================= */

.section-card {
    background: #FFFFFF;
    border: 1px solid #DDE6F2;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(31, 55, 90, 0.045);
}

.section-title {
    color: #173B6C;
    font-size: 18px;
    font-weight: 750;
    margin-bottom: 3px;
}

.section-subtitle {
    color: #718096;
    font-size: 12px;
    margin-bottom: 14px;
}

/* ================= STREAMLIT TABS ================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background: #FFFFFF;
    border: 1px solid #DDE6F2;
    border-radius: 12px;
    padding: 5px;
    margin-bottom: 18px;
}

.stTabs [data-baseweb="tab"] {
    height: 38px;
    border-radius: 8px;
    color: #667085;
    font-weight: 650;
    padding: 0 17px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #E7F0FF, #F0EAFF);
    color: #173B6C !important;
}

/* ================= STREAMLIT METRICS ================= */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DDE6F2;
    border-radius: 12px;
    padding: 13px 16px;
    box-shadow: 0 3px 10px rgba(31, 55, 90, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #667085 !important;
}

[data-testid="stMetricValue"] {
    color: #173B6C !important;
}

/* ================= INSIGHT BOX ================= */

.insight {
    background: linear-gradient(90deg, #EFF7FF, #F8F4FF);
    border-left: 4px solid #5B7FC4;
    border-radius: 8px;
    padding: 13px 16px;
    color: #344054;
    font-size: 13px;
    line-height: 1.6;
    margin: 10px 0;
}

/* ================= SELECTBOX / INPUTS ================= */

.stSelectbox > div > div,
.stNumberInput > div > div {
    background: #FFFFFF;
    border-color: #D5DEED;
    border-radius: 8px;
}

.stSelectbox label,
.stNumberInput label,
.stSlider label,
.stFileUploader label {
    color: #475467 !important;
    font-weight: 600 !important;
}

/* ================= DATAFRAME ================= */

[data-testid="stDataFrame"] {
    border: 1px solid #DDE6F2;
    border-radius: 10px;
    overflow: hidden;
}

/* ================= HEADINGS ================= */

h1, h2, h3 {
    color: #173B6C;
}

h4, h5, h6 {
    color: #344054;
}

/* ================= SUCCESS / WARNING / INFO ================= */

[data-testid="stAlert"] {
    border-radius: 9px;
}

/* ================= FOOTER ================= */

.footer {
    color: #98A2B3;
    font-size: 11px;
    text-align: center;
    padding: 18px 0 4px 0;
}

/* ================= DIVIDER ================= */

hr {
    border: none;
    border-top: 1px solid #E1E7F0;
    margin: 12px 0 18px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
TARGET = "sales"

def clean_dataframe(df):
    df = df.copy()
    if TARGET not in df.columns:
        return df, "Target column 'sales' was not found."

    # Match the original notebook: remove rows with missing target.
    before = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    removed = before - len(df)

    return df, f"{removed:,} rows removed because sales was missing."

def prepare_features(df):
    """Create a model matrix consistent with the notebook."""
    X = df.drop(columns=[TARGET], errors="ignore").copy()

    # Original notebook uses dummy variables for categorical columns.
    X = pd.get_dummies(X, drop_first=True)

    # Convert boolean columns generated by get_dummies to numeric.
    X = X.astype(float)

    # Statsmodels intercept.
    X = sm.add_constant(X, has_constant="add")
    return X

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def performance(model, X, y):
    pred = model.predict(X)
    return {
        "R²": r2_score(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAE": mean_absolute_error(y, pred),
        "MAPE": safe_mape(y, pred),
    }

def fmt_num(x):
    if pd.isna(x):
        return "—"
    return f"{x:,.2f}"

def make_model(df, test_size=0.20, random_state=1):
    X = prepare_features(df)
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = sm.OLS(y_train, X_train).fit()
    return model, X_train, X_test, y_train, y_test

def kpi(label, value, caption=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "KPI"

if "data" not in st.session_state:
    st.session_state.data = None

if "model" not in st.session_state:
    st.session_state.model = None

if "model_data" not in st.session_state:
    st.session_state.model_data = None

# ============================================================
# TOP HEADER
# ============================================================
st.markdown("""
<div class="topbar">
    <div class="brand">Firm Sales Analytics</div>
    <div class="subbrand">Linear Regression | Business Intelligence | Model Diagnostics</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload firm-level CSV", type=["csv"])
    st.caption("Expected target column: sales")

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.session_state.data, _ = clean_dataframe(raw)

elif st.session_state.data is None:
    # Optional local fallback for the original notebook environment.
    default_paths = [
        "Firm_level_data (1).csv",
        "/content/Firm_level_data (1).csv",
    ]
    for path in default_paths:
        try:
            raw = pd.read_csv(path)
            st.session_state.data, _ = clean_dataframe(raw)
            break
        except Exception:
            pass

if st.session_state.data is None:
    st.warning("Upload the firm-level CSV to start the dashboard.")
    st.stop()

df = st.session_state.data

# ============================================================
# BOOKMARK NAVIGATION
# ============================================================
st.markdown('<div class="bookmark-label">Bookmarks</div>', unsafe_allow_html=True)

bookmark_cols = st.columns(7)
bookmarks = [
    "KPI",
    "Data preprocessing",
    "EDA",
    "Model building",
    "Model summary",
    "Predict",
    "About",
]

for col, name in zip(bookmark_cols, bookmarks):
    with col:
        if st.button(name, use_container_width=True, key=f"bookmark_{name}"):
            st.session_state.page = name

page = st.session_state.page

st.markdown("---")

# ============================================================
# KPI
# ============================================================
if page == "KPI":
    st.markdown(
        '<div class="section-card"><div class="section-title">Executive KPI Overview</div>'
        '<div class="section-subtitle">High-level view of the firm-level sales dataset and model readiness.</div></div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    with cols[0]:
        kpi("Firms", f"{len(df):,}", "records after target cleaning")
    with cols[1]:
        kpi("Features", f"{df.shape[1]-1:,}", "predictor variables")
    with cols[2]:
        kpi("Avg. Sales", fmt_num(df[TARGET].mean()), "mean firm sales")
    with cols[3]:
        kpi("Median Sales", fmt_num(df[TARGET].median()), "median firm sales")
    with cols[4]:
        kpi("Missing Cells", f"{int(df.isna().sum().sum()):,}", "remaining missing values")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-card"><div class="section-title">Sales Distribution</div>'
                    '<div class="section-subtitle">Distribution of the dependent variable.</div></div>',
                    unsafe_allow_html=True)
        fig = px.histogram(df, x=TARGET, nbins=30, marginal="box")
        fig.update_layout(
            template="simple_white",
            height=390,
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(color="#344054"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        numeric = df.select_dtypes(include=np.number)
        if TARGET in numeric.columns and numeric.shape[1] > 1:
            corr = numeric.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
            fig = px.bar(
                x=corr.values,
                y=corr.index,
                orientation="h",
                labels={"x": "Correlation with sales", "y": ""},
            )
            fig.update_layout(
                template="simple_white",
                height=390,
                margin=dict(l=20, r=20, t=20, b=20),
                font=dict(color="#344054"),
            )
            st.markdown('<div class="section-card"><div class="section-title">Sales Correlation</div>'
                        '<div class="section-subtitle">Linear relationship between numeric predictors and sales.</div></div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DATA PREPROCESSING
# ============================================================
elif page == "Data preprocessing":
    st.markdown(
        '<div class="section-card"><div class="section-title">Data Preprocessing</div>'
        '<div class="section-subtitle">Dataset quality, structure, missing values, duplicates and descriptive statistics.</div></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Duplicates", f"{int(df.duplicated().sum()):,}")
    c4.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Missing Values")
    missing = df.isna().sum().sort_values(ascending=False)
    missing_df = pd.DataFrame({"Column": missing.index, "Missing Values": missing.values})
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.markdown(
        '<div class="insight"><b>Preprocessing applied:</b> rows with missing sales are removed, '
        'categorical predictors are converted to dummy variables, and the regression matrix receives an intercept.</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# EDA
# ============================================================
elif page == "EDA":
    st.markdown(
        '<div class="section-card"><div class="section-title">Exploratory Data Analysis</div>'
        '<div class="section-subtitle">Understand distributions, relationships, outliers and correlations.</div></div>',
        unsafe_allow_html=True,
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns available for EDA.")
    else:
        selected = st.selectbox("Select a numeric variable", numeric_cols)

        c1, c2 = st.columns(2)

        with c1:
            fig = px.histogram(df, x=selected, nbins=30, marginal="box")
            fig.update_layout(template="simple_white", height=430)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if selected != TARGET:
                fig = px.scatter(
                    df,
                    x=selected,
                    y=TARGET,
                    trendline="ols",
                    opacity=0.65,
                    labels={selected: selected, TARGET: "Sales"},
                )
                fig.update_layout(template="simple_white", height=430)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a predictor to view its relationship with sales.")

    st.subheader("Correlation Matrix")

    corr = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(template="simple_white", height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Outlier Overview")
    outlier_rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = ((s < lower) | (s > upper)).sum()
        outlier_rows.append({
            "Variable": col,
            "Outliers": int(count),
            "Outlier %": round(count / len(s) * 100, 2),
        })

    st.dataframe(
        pd.DataFrame(outlier_rows).sort_values("Outliers", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# MODEL BUILDING
# ============================================================
elif page == "Model building":
    st.markdown(
        '<div class="section-card"><div class="section-title">Model Building</div>'
        '<div class="section-subtitle">Ordinary Least Squares linear regression with an 80:20 train-test split.</div></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test size", 0.10, 0.40, 0.20, 0.05)
    random_state = c2.number_input("Random state", 1, 100, 1)
    run_model = c3.button("Build Linear Regression Model", use_container_width=True)

    if run_model or st.session_state.model is None:
        try:
            model, X_train, X_test, y_train, y_test = make_model(
                df, test_size=float(test_size), random_state=int(random_state)
            )

            st.session_state.model = model
            st.session_state.model_data = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }
            st.success("Linear regression model built successfully.")
        except Exception as e:
            st.error(f"Model could not be built: {e}")

    if st.session_state.model is not None:
        md = st.session_state.model_data
        train_perf = performance(st.session_state.model, md["X_train"], md["y_train"])
        test_perf = performance(st.session_state.model, md["X_test"], md["y_test"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test R²", f"{test_perf['R²']:.3f}")
        c2.metric("Test RMSE", f"{test_perf['RMSE']:.3f}")
        c3.metric("Test MAE", f"{test_perf['MAE']:.3f}")
        c4.metric("Test MAPE", f"{test_perf['MAPE']:.2f}%")

        comparison = pd.DataFrame({
            "Metric": ["R²", "RMSE", "MAE", "MAPE"],
            "Train": [
                train_perf["R²"], train_perf["RMSE"],
                train_perf["MAE"], train_perf["MAPE"]
            ],
            "Test": [
                test_perf["R²"], test_perf["RMSE"],
                test_perf["MAE"], test_perf["MAPE"]
            ],
        })

        st.subheader("Train vs Test Performance")
        st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)

        fig = px.bar(
            comparison,
            x="Metric",
            y=["Train", "Test"],
            barmode="group",
        )
        fig.update_layout(template="simple_white", height=390)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODEL SUMMARY
# ============================================================
elif page == "Model summary":
    st.markdown(
        '<div class="section-card"><div class="section-title">Model Summary</div>'
        '<div class="section-subtitle">Statistical significance, coefficients, VIF and regression diagnostics.</div></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.model is None:
        st.info("Build the model first from the Model building bookmark.")
    else:
        model = st.session_state.model
        md = st.session_state.model_data

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{model.rsquared:.3f}")
        c2.metric("Adjusted R²", f"{model.rsquared_adj:.3f}")
        c3.metric("AIC", f"{model.aic:.2f}")
        c4.metric("Observations", f"{int(model.nobs):,}")

        st.subheader("Regression Coefficients")

        coef = pd.DataFrame({
            "Feature": model.params.index,
            "Coefficient": model.params.values,
            "P-value": model.pvalues.values,
            "Significant at 5%": model.pvalues.values < 0.05,
        }).sort_values("P-value")

        st.dataframe(
            coef.style.format({
                "Coefficient": "{:.5f}",
                "P-value": "{:.5f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Feature Significance")

        plot_coef = coef[coef["Feature"] != "const"].copy()
        plot_coef["Abs Coefficient"] = plot_coef["Coefficient"].abs()
        plot_coef = plot_coef.sort_values("Abs Coefficient", ascending=False).head(15)

        fig = px.bar(
            plot_coef.sort_values("Coefficient"),
            x="Coefficient",
            y="Feature",
            orientation="h",
        )
        fig.update_layout(template="simple_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Multicollinearity — VIF")

        X_no_const = md["X_train"].drop(columns=["const"], errors="ignore")
        if X_no_const.shape[1] > 0:
            vif_rows = []
            for i, col in enumerate(X_no_const.columns):
                try:
                    vif_value = variance_inflation_factor(X_no_const.values, i)
                except Exception:
                    vif_value = np.nan
                vif_rows.append({"Feature": col, "VIF": vif_value})

            vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
            st.dataframe(
                vif_df.style.format({"VIF": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Residual Diagnostics")

        pred = model.predict(md["X_train"])
        residuals = md["y_train"] - pred
        diagnostic_df = pd.DataFrame({
            "Actual": md["y_train"].values,
            "Fitted": pred.values,
            "Residual": residuals.values,
        })

        c1, c2 = st.columns(2)

        with c1:
            fig = px.scatter(
                diagnostic_df,
                x="Fitted",
                y="Residual",
                trendline="ols",
                opacity=0.65,
            )
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(template="simple_white", height=420)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(diagnostic_df, x="Residual", nbins=30)
            fig.update_layout(template="simple_white", height=420)
            st.plotly_chart(fig, use_container_width=True)

        shapiro_note = ""
        if len(residuals) <= 5000:
            try:
                _, p = stats.shapiro(residuals)
                shapiro_note = f"Shapiro-Wilk p-value: {p:.4f}"
            except Exception:
                shapiro_note = "Shapiro-Wilk test unavailable."
        else:
            shapiro_note = "Shapiro-Wilk skipped for more than 5,000 residuals."

        st.markdown(
            f'<div class="insight"><b>Diagnostic note:</b> {shapiro_note}. '
            'Residual plots should be checked for visible patterns and changing variance.</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Full Statsmodels Regression Summary"):
            st.text(model.summary().as_text())

# ============================================================
# PREDICT
# ============================================================
elif page == "Predict":
    st.markdown(
        '<div class="section-card"><div class="section-title">Predict Firm Sales</div>'
        '<div class="section-subtitle">Enter firm attributes and generate a sales estimate from the trained regression model.</div></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.model is None:
        st.info("Build the model first from the Model building bookmark.")
    else:
        model = st.session_state.model
        X_train = st.session_state.model_data["X_train"]

        feature_columns = [c for c in X_train.columns if c != "const"]

        st.write("Enter values for the model features:")

        input_values = {}

        cols = st.columns(3)
        for i, feature in enumerate(feature_columns):
            with cols[i % 3]:
                original_feature = feature

                # Handle one-hot encoded columns.
                if feature.endswith("_yes") or feature.endswith("_Yes"):
                    input_values[feature] = st.selectbox(
                        feature.replace("_", " ").title(),
                        [0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        key=f"pred_{feature}",
                    )
                else:
                    # Recover the base column where possible.
                    base_col = feature
                    if base_col in df.columns and pd.api.types.is_numeric_dtype(df[base_col]):
                        default = float(df[base_col].median())
                    else:
                        default = 0.0

                    input_values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        value=default,
                        key=f"pred_{feature}",
                    )

        if st.button("Generate Sales Prediction", use_container_width=True):
            try:
                new_firm = pd.DataFrame([input_values])
                new_firm = new_firm.reindex(columns=feature_columns, fill_value=0)
                new_firm = sm.add_constant(new_firm, has_constant="add")
                new_firm = new_firm.reindex(columns=X_train.columns, fill_value=0).astype(float)

                prediction = float(model.predict(new_firm).iloc[0])

                st.markdown(
                    f"""
                    <div class="section-card" style="text-align:center;">
                        <div class="section-subtitle">Predicted Sales</div>
                        <div style="font-size:38px;font-weight:800;color:#173B6C;">
                            {prediction:,.2f}
                        </div>
                        <div class="kpi-caption">Estimated using the fitted OLS regression model</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ============================================================
# ABOUT
# ============================================================
elif page == "About":
    st.markdown(
        '<div class="section-card"><div class="section-title">Project Overview</div>'
        '<div class="section-subtitle">Firm-level sales prediction using linear regression.</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    **Objective**

    Predict firm sales using firm-level attributes and identify the variables that
    have the strongest statistical relationship with sales.

    **Workflow**

    Data preprocessing → EDA → Linear regression → Model diagnostics →
    Performance evaluation → Sales prediction

    **Primary model**

    Ordinary Least Squares (OLS) linear regression.

    **Performance metrics**

    R², Adjusted R², RMSE, MAE and MAPE.

    **Diagnostics**

    Multicollinearity using VIF, residual analysis, normality assessment and
    heteroscedasticity-oriented diagnostics.
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer">Firm Sales Analytics Dashboard • Linear Regression • Built with Streamlit</div>',
    unsafe_allow_html=True,
)
