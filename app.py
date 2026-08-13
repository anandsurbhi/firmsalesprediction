"""
Firm Performance Analytics Dashboard
=====================================
A professional Streamlit dashboard for exploring firm-level financial data,
visualizing key relationships, and comparing regression models
(OLS Linear Regression, Random Forest, XGBoost) that predict firm `sales`.

Run with:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Firm Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# THEME / COLOR SYSTEM
# --------------------------------------------------------------------------------------
PRIMARY = "#4F46E5"      # Indigo
PRIMARY_DARK = "#3730A3"
ACCENT = "#06B6D4"       # Cyan
SUCCESS = "#10B981"      # Emerald
WARNING = "#F59E0B"      # Amber
DANGER = "#EF4444"       # Red
BG_CARD = "#111827"
BG_CARD_LIGHT = "#1F2937"
TEXT_MUTED = "#9CA3AF"
PALETTE = [PRIMARY, ACCENT, SUCCESS, WARNING, DANGER, "#A855F7", "#EC4899", "#14B8A6"]
MODEL_COLORS = {"OLS Linear Regression": PRIMARY, "Random Forest": SUCCESS, "XGBoost": WARNING}

st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(circle at 10% 0%, #111827 0%, #0B0F19 55%, #0B0F19 100%);
    }}
    section[data-testid="stSidebar"] {{
        background: #0B0F19;
        border-right: 1px solid #1F2937;
    }}
    h1, h2, h3, h4 {{
        color: #F9FAFB !important;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }}
    p, li, span, label {{
        color: #E5E7EB;
    }}
    .kpi-card {{
        background: linear-gradient(145deg, {BG_CARD} 0%, {BG_CARD_LIGHT} 100%);
        border: 1px solid #273349;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }}
    .kpi-label {{
        color: {TEXT_MUTED};
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 1.65rem;
        font-weight: 700;
        color: #F9FAFB;
    }}
    .kpi-sub {{
        font-size: 0.78rem;
        margin-top: 4px;
        font-weight: 600;
    }}
    .section-banner {{
        background: linear-gradient(90deg, {PRIMARY} 0%, {ACCENT} 100%);
        padding: 14px 22px;
        border-radius: 12px;
        color: white;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 18px 0 14px 0;
    }}
    .pill {{
        display:inline-block; padding:3px 12px; border-radius:999px;
        font-size:0.72rem; font-weight:700; letter-spacing:.03em;
    }}
    div[data-testid="stMetricValue"] {{ color: #F9FAFB; }}
    thead tr th {{ background-color: {BG_CARD_LIGHT} !important; color: #F9FAFB !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label, value, sub=None, sub_color=SUCCESS):
    sub_html = f'<div class="kpi-sub" style="color:{sub_color}">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_banner(text):
    st.markdown(f'<div class="section-banner">{text}</div>', unsafe_allow_html=True)


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E5E7EB"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=50, b=10),
)


def style_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#1F2937", zerolinecolor="#1F2937")
    fig.update_yaxes(gridcolor="#1F2937", zerolinecolor="#1F2937")
    return fig


# --------------------------------------------------------------------------------------
# SIDEBAR - DATA SOURCE
# --------------------------------------------------------------------------------------
st.sidebar.markdown("## 📊 Firm Analytics")
st.sidebar.caption("Upload your firm-level dataset to begin, or use the bundled sample data.")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])


@st.cache_data(show_spinner=False)
def load_sample_data(n=400, seed=1):
    """Synthetic fallback dataset mirroring the original firm-level schema
    (used only if no file is uploaded and no sample file is found on disk)."""
    rng = np.random.default_rng(seed)
    capital = rng.gamma(4, 300, n)
    employment = rng.gamma(3, 150, n)
    randd = rng.gamma(2, 40, n) * (rng.random(n) > 0.2)
    patents = (randd * rng.uniform(0.3, 1.2, n) + rng.normal(0, 5, n)).clip(min=0)
    institutions = rng.uniform(5, 95, n)
    tobinq = rng.gamma(3, 0.7, n) + 0.5
    value = capital * rng.uniform(1.5, 4, n) + randd * 5
    sp500 = rng.choice(["yes", "no"], size=n, p=[0.35, 0.65])
    noise = rng.normal(0, 150, n)
    sales = (
        2.1 * capital
        + 1.4 * employment
        + 3.0 * randd
        + 0.8 * patents
        + 40 * tobinq
        + (sp500 == "yes") * 500
        + noise
        + 300
    ).clip(min=10)
    return pd.DataFrame(
        {
            "sales": sales,
            "capital": capital,
            "patents": patents,
            "randd": randd,
            "employment": employment,
            "tobinq": tobinq,
            "value": value,
            "institutions": institutions,
            "sp500": sp500,
        }
    )


@st.cache_data(show_spinner=False)
def load_uploaded(file_bytes):
    import io
    return pd.read_csv(io.BytesIO(file_bytes))


if uploaded_file is not None:
    df_raw = load_uploaded(uploaded_file.getvalue())
    data_source_label = f"Uploaded: {uploaded_file.name}"
else:
    df_raw = load_sample_data()
    data_source_label = "Sample synthetic firm data (upload your own CSV in the sidebar)"

st.sidebar.info(data_source_label)

# Basic cleaning: drop fully-empty rows/cols, de-duplicate column names
df = df_raw.copy()
df = df.dropna(axis=1, how="all")
if df.columns.duplicated().any():
    dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
    st.sidebar.warning(f"Duplicate column names found and de-duplicated: {dup_cols}")
    df = df.loc[:, ~df.columns.duplicated()]

numeric_cols_all = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols_all = [c for c in df.columns if c not in numeric_cols_all]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Configuration")

default_target = "sales" if "sales" in numeric_cols_all else (numeric_cols_all[0] if numeric_cols_all else None)
target_col = st.sidebar.selectbox(
    "Target variable (to predict)",
    options=numeric_cols_all,
    index=numeric_cols_all.index(default_target) if default_target in numeric_cols_all else 0,
)

drop_na_target = st.sidebar.checkbox(f"Drop rows with missing '{target_col}'", value=True)
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
run_xgb = st.sidebar.checkbox("Include XGBoost", value=XGB_AVAILABLE, disabled=not XGB_AVAILABLE)
if not XGB_AVAILABLE:
    st.sidebar.caption("⚠️ xgboost not installed — install with `pip install xgboost` to enable it.")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · statsmodels · scikit-learn · XGBoost")

if drop_na_target:
    df = df.dropna(subset=[target_col])

# --------------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 6px;">
        <div>
            <h1 style="margin-bottom:0;">📊 Firm Performance Analytics Dashboard</h1>
            <p style="color:{TEXT_MUTED}; margin-top:2px;">
                Exploratory analysis and predictive modeling of firm-level financial &amp; operational data.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_kpi, tab_viz, tab_model, tab_predict, tab_data = st.tabs(
    ["🏠 Overview & KPIs", "📈 Visualizations", "🧠 Model Summary", "🔮 Predict", "🗂️ Raw Data"]
)

# ========================================================================================
# TAB 1: OVERVIEW & KPIs
# ========================================================================================
with tab_kpi:
    section_banner("Key Performance Indicators")

    n_rows, n_cols = df.shape
    missing_pct = df.isnull().mean().mean() * 100
    dup_count = df.duplicated().sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Records", f"{n_rows:,}", f"{n_cols} columns", TEXT_MUTED)
    with c2:
        kpi_card("Avg " + target_col.title(), f"{df[target_col].mean():,.1f}",
                  f"Median {df[target_col].median():,.1f}", ACCENT)
    with c3:
        kpi_card("Missing Data", f"{missing_pct:.1f}%",
                  "Clean" if missing_pct < 1 else "Needs attention",
                  SUCCESS if missing_pct < 1 else WARNING)
    with c4:
        kpi_card("Duplicate Rows", f"{dup_count:,}",
                  "None found" if dup_count == 0 else "Review recommended",
                  SUCCESS if dup_count == 0 else DANGER)

    st.write("")
    c5, c6, c7, c8 = st.columns(4)
    other_num = [c for c in numeric_cols_all if c != target_col][:3]
    for col, container in zip(other_num, [c5, c6, c7]):
        with container:
            kpi_card(col.title(), f"{df[col].mean():,.2f}", f"Std {df[col].std():,.2f}", PRIMARY)
    if categorical_cols_all:
        cat = categorical_cols_all[0]
        with c8:
            top_val = df[cat].value_counts(normalize=True).idxmax()
            top_pct = df[cat].value_counts(normalize=True).max() * 100
            kpi_card(f"Top '{cat}'", str(top_val), f"{top_pct:.1f}% of records", ACCENT)

    st.write("")
    section_banner("Correlation with Target")
    corr_target = df[numeric_cols_all].corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    fig = px.bar(
        corr_target,
        orientation="h",
        color=corr_target.values,
        color_continuous_scale=["#EF4444", "#374151", "#10B981"],
        labels={"value": "Correlation", "index": "Feature"},
        title=f"Feature correlation with {target_col}",
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(style_fig(fig), use_container_width=True)

# ========================================================================================
# TAB 2: VISUALIZATIONS
# ========================================================================================
with tab_viz:
    section_banner("Distribution Explorer")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        dist_col = st.selectbox("Select variable", numeric_cols_all, index=numeric_cols_all.index(target_col))
        show_kde = st.checkbox("Show density curve", value=True)
    with col_b:
        fig = px.histogram(
            df, x=dist_col, nbins=40, marginal="box",
            color_discrete_sequence=[PRIMARY],
            title=f"Distribution of {dist_col}",
        )
        fig.add_vline(x=df[dist_col].mean(), line_dash="dash", line_color=ACCENT,
                       annotation_text="mean", annotation_font_color=ACCENT)
        fig.add_vline(x=df[dist_col].median(), line_dash="dot", line_color=WARNING,
                       annotation_text="median", annotation_font_color=WARNING)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        section_banner("Correlation Heatmap")
        corr = df[numeric_cols_all].corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu", zmin=-1, zmax=1,
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with col2:
        section_banner("Category Breakdown")
        if categorical_cols_all:
            cat_col = st.selectbox("Categorical variable", categorical_cols_all)
            vc = df[cat_col].value_counts().reset_index()
            vc.columns = [cat_col, "count"]
            fig = px.bar(vc, x=cat_col, y="count", color=cat_col,
                         color_discrete_sequence=PALETTE, title=f"{target_col} volume by {cat_col}")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("No categorical columns detected in this dataset.")

    st.write("")
    section_banner("Relationship Explorer (Scatter)")
    col1, col2, col3 = st.columns(3)
    with col1:
        x_var = st.selectbox("X-axis", numeric_cols_all, index=0)
    with col2:
        y_var = st.selectbox("Y-axis", numeric_cols_all,
                              index=numeric_cols_all.index(target_col) if target_col in numeric_cols_all else 0)
    with col3:
        color_var = st.selectbox("Color by", ["None"] + categorical_cols_all)

    if x_var == y_var:
        st.warning("Please select two **different** variables for the X and Y axes.")
    else:
        scatter_cols = [x_var, y_var] + ([color_var] if color_var != "None" else [])
        scatter_df = df.loc[:, ~df.columns.duplicated()][scatter_cols].copy()
        fig = px.scatter(
            scatter_df, x=x_var, y=y_var,
            color=None if color_var == "None" else color_var,
            trendline="ols",
            color_discrete_sequence=PALETTE,
            opacity=0.75,
            title=f"{y_var} vs {x_var}",
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    if categorical_cols_all:
        st.write("")
        section_banner("Boxplot by Category")
        cbox = st.selectbox("Category for boxplot", categorical_cols_all, key="box_cat")
        nbox = st.selectbox("Numeric variable", numeric_cols_all, key="box_num",
                             index=numeric_cols_all.index(target_col) if target_col in numeric_cols_all else 0)
        fig = px.box(df, x=cbox, y=nbox, color=cbox, color_discrete_sequence=PALETTE)
        st.plotly_chart(style_fig(fig), use_container_width=True)

# ========================================================================================
# SHARED: DATA PREP FOR MODELING
# ========================================================================================
@st.cache_data(show_spinner=False)
def prepare_model_data(df, target_col, test_size):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = pd.get_dummies(X, drop_first=True)
    X = X.select_dtypes(include=np.number)
    X = X.fillna(X.mean(numeric_only=True))
    y = y.astype(float)

    X_sm = sm.add_constant(X, has_constant="add")

    x_train, x_test, y_train, y_test = train_test_split(
        X_sm, y, test_size=test_size, random_state=1
    )
    return X_sm, x_train, x_test, y_train, y_test


def mape_score(targets, predictions):
    targets = np.array(targets)
    predictions = np.array(predictions)
    mask = targets != 0
    return np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100


def perf_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mape_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, R2=r2)


@st.cache_resource(show_spinner=True)
def train_models(x_train, x_test, y_train, y_test, use_xgb):
    results = {}

    # ---- OLS Linear Regression ----
    ols = sm.OLS(y_train, x_train).fit()
    results["OLS Linear Regression"] = {
        "model": ols,
        "train_pred": ols.predict(x_train),
        "test_pred": ols.predict(x_test),
        "train_perf": perf_metrics(y_train, ols.predict(x_train)),
        "test_perf": perf_metrics(y_test, ols.predict(x_test)),
    }

    # ---- Random Forest ----
    rf = RandomForestRegressor(random_state=1, n_estimators=200)
    rf.fit(x_train, y_train)
    results["Random Forest"] = {
        "model": rf,
        "train_pred": rf.predict(x_train),
        "test_pred": rf.predict(x_test),
        "train_perf": perf_metrics(y_train, rf.predict(x_train)),
        "test_perf": perf_metrics(y_test, rf.predict(x_test)),
        "importance": pd.Series(rf.feature_importances_, index=x_train.columns).sort_values(ascending=False),
    }

    # ---- XGBoost ----
    if use_xgb and XGB_AVAILABLE:
        xgb = XGBRegressor(random_state=1, n_estimators=300, max_depth=4, learning_rate=0.08)
        xgb.fit(x_train, y_train)
        results["XGBoost"] = {
            "model": xgb,
            "train_pred": xgb.predict(x_train),
            "test_pred": xgb.predict(x_test),
            "train_perf": perf_metrics(y_train, xgb.predict(x_train)),
            "test_perf": perf_metrics(y_test, xgb.predict(x_test)),
            "importance": pd.Series(xgb.feature_importances_, index=x_train.columns).sort_values(ascending=False),
        }

    return results


X_sm, x_train, x_test, y_train, y_test = prepare_model_data(df, target_col, test_size)

with st.spinner("Training models..."):
    model_results = train_models(x_train, x_test, y_train, y_test, run_xgb)

# ========================================================================================
# TAB 3: MODEL SUMMARY
# ========================================================================================
with tab_model:
    section_banner("Model Performance Comparison (Test Set)")

    perf_table = pd.DataFrame(
        {name: res["test_perf"] for name, res in model_results.items()}
    ).T
    perf_table = perf_table[["R2", "RMSE", "MAE", "MAPE"]]
    perf_table.columns = ["R² Score", "RMSE", "MAE", "MAPE (%)"]

    best_model = perf_table["R² Score"].idxmax()

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        kpi_card("Best Model", best_model, f"R² = {perf_table.loc[best_model, 'R² Score']:.3f}", SUCCESS)
    with mc2:
        kpi_card("Lowest RMSE", perf_table["RMSE"].idxmin(), f"{perf_table['RMSE'].min():,.2f}", ACCENT)
    with mc3:
        kpi_card("Lowest MAE", perf_table["MAE"].idxmin(), f"{perf_table['MAE'].min():,.2f}", PRIMARY)
    with mc4:
        kpi_card("Lowest MAPE", perf_table["MAPE (%)"].idxmin(), f"{perf_table['MAPE (%)'].min():,.1f}%", WARNING)

    st.write("")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("##### 📋 Performance Table")
        st.dataframe(
            perf_table.style.format({"R² Score": "{:.3f}", "RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE (%)": "{:.1f}"})
            .background_gradient(subset=["R² Score"], cmap="Greens")
            .background_gradient(subset=["RMSE", "MAE", "MAPE (%)"], cmap="Reds_r"),
            use_container_width=True,
        )
    with col2:
        fig = go.Figure()
        for metric in ["RMSE", "MAE"]:
            fig.add_trace(go.Bar(
                x=list(model_results.keys()),
                y=perf_table[metric if metric != "RMSE" else "RMSE"],
                name=metric,
            ))
        fig.update_layout(barmode="group", title="RMSE / MAE by Model")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.write("")
    fig = px.bar(
        perf_table.reset_index().rename(columns={"index": "Model"}),
        x="Model", y="R² Score", color="Model",
        color_discrete_map=MODEL_COLORS,
        title="R² Score by Model (higher is better)",
        text_auto=".3f",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.write("")
    section_banner("Predicted vs Actual")
    model_choice = st.selectbox("Select model", list(model_results.keys()))
    res = model_results[model_choice]
    plot_df = pd.DataFrame({"Actual": y_test, "Predicted": res["test_pred"]})
    fig = px.scatter(
        plot_df, x="Actual", y="Predicted", opacity=0.7,
        color_discrete_sequence=[MODEL_COLORS.get(model_choice, PRIMARY)],
        title=f"{model_choice}: Predicted vs Actual ({target_col})",
    )
    min_v, max_v = plot_df["Actual"].min(), plot_df["Actual"].max()
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                              line=dict(color=TEXT_MUTED, dash="dash"), name="Perfect fit"))
    st.plotly_chart(style_fig(fig), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        section_banner("Residual Plot")
        resid = y_test - res["test_pred"]
        fig = px.scatter(
            x=res["test_pred"], y=resid, opacity=0.7,
            labels={"x": "Fitted Values", "y": "Residuals"},
            color_discrete_sequence=[DANGER],
            title="Fitted vs Residuals",
        )
        fig.add_hline(y=0, line_dash="dash", line_color=TEXT_MUTED)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with col2:
        section_banner("Residual Distribution")
        fig = px.histogram(resid, nbins=30, color_discrete_sequence=[ACCENT], title="Residual Distribution")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    if "importance" in res:
        st.write("")
        section_banner(f"Feature Importance — {model_choice}")
        imp = res["importance"].head(12).sort_values()
        fig = px.bar(
            imp, orientation="h",
            color=imp.values, color_continuous_scale=[BG_CARD_LIGHT, MODEL_COLORS.get(model_choice, PRIMARY)],
            title="Top Feature Importances",
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    if model_choice == "OLS Linear Regression":
        st.write("")
        section_banner("OLS Regression Summary")
        with st.expander("📄 Full statsmodels OLS summary", expanded=False):
            st.text(res["model"].summary())

        section_banner("Multicollinearity Check (VIF)")
        try:
            vif_df = pd.DataFrame()
            vif_df["Feature"] = x_train.columns
            vif_df["VIF"] = [variance_inflation_factor(x_train.values, i) for i in range(len(x_train.columns))]
            vif_df = vif_df.sort_values("VIF", ascending=False)
            st.dataframe(
                vif_df.style.format({"VIF": "{:.2f}"}).background_gradient(subset=["VIF"], cmap="OrRd"),
                use_container_width=True,
            )
            st.caption("Rule of thumb: VIF > 5 indicates moderate multicollinearity, VIF > 10 indicates high multicollinearity.")
        except Exception as e:
            st.warning(f"Could not compute VIF: {e}")

# ========================================================================================
# TAB 4: PREDICT
# ========================================================================================
with tab_predict:
    section_banner(f"Predict {target_col.title()} for a New Firm")
    st.caption("Enter firm characteristics below and generate a prediction from the selected model.")

    pred_model_name = st.selectbox("Model to use", list(model_results.keys()), key="predict_model")

    feature_cols = [c for c in x_train.columns if c != "const"]
    input_vals = {}
    n_per_row = 3
    rows = [feature_cols[i:i + n_per_row] for i in range(0, len(feature_cols), n_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for c, feat in zip(cols, row):
            with c:
                default_val = float(x_train[feat].median())
                input_vals[feat] = st.number_input(feat, value=default_val, format="%.3f")

    if st.button("🔮 Predict", type="primary"):
        input_df = pd.DataFrame([input_vals])
        input_df.insert(0, "const", 1.0)
        input_df = input_df[x_train.columns]

        model_obj = model_results[pred_model_name]["model"]
        prediction = model_obj.predict(input_df)[0] if pred_model_name == "OLS Linear Regression" else model_obj.predict(input_df.drop(columns=["const"]))[0]

        st.markdown(
            f"""
            <div class="kpi-card" style="border: 1px solid {SUCCESS}; margin-top:10px;">
                <div class="kpi-label">Predicted {target_col.title()}</div>
                <div class="kpi-value" style="color:{SUCCESS}; font-size:2.2rem;">{prediction:,.2f}</div>
                <div class="kpi-sub" style="color:{TEXT_MUTED};">using {pred_model_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ========================================================================================
# TAB 5: RAW DATA
# ========================================================================================
with tab_data:
    section_banner("Dataset Preview")
    st.dataframe(df.head(200), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Summary Statistics")
        st.dataframe(df.describe().T, use_container_width=True)
    with col2:
        st.markdown("##### Missing Values")
        miss = df.isnull().sum()
        miss = miss[miss > 0]
        if len(miss):
            st.dataframe(miss.rename("Missing Count"), use_container_width=True)
        else:
            st.success("No missing values in the current dataset.")

    st.download_button(
        "⬇️ Download current dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="firm_data_export.csv",
        mime="text/csv",
    )
