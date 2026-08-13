
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Firm Sales Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: #f7f8fc;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 750;
            color: #111827;
            margin-bottom: 0.15rem;
        }

        .subtitle {
            color: #667085;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .kpi-card {
            background: white;
            border: 1px solid #e7e9ef;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
        }

        .kpi-label {
            color: #667085;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .kpi-value {
            color: #111827;
            font-size: 1.65rem;
            font-weight: 750;
            margin-top: 4px;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #111827;
            margin-top: 0.7rem;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e7e9ef;
            padding: 15px;
            border-radius: 14px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 10px 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
TARGET = "sales"
EXPECTED_FEATURES = [
    "capital",
    "patents",
    "randd",
    "employment",
    "sp500",
    "tobinq",
    "value",
    "institutions",
]


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "R²": r2_score(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAE": mean_absolute_error(y, pred),
        "MAPE": np.mean(
            np.abs((y - pred) / np.where(np.abs(y) < 1e-8, 1e-8, y))
        ) * 100,
    }, pred


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


@st.cache_data
def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Keep the project target and available predictors.
    if TARGET not in df.columns:
        raise ValueError("The uploaded file must contain a 'sales' column.")

    df = df.dropna(subset=[TARGET])

    # Match the notebook's categorical variable.
    if "sp500" in df.columns:
        df["sp500"] = df["sp500"].astype(str).str.lower().str.strip()

    return df


@st.cache_resource
def train_models(df, test_size, random_state):
    features = [c for c in EXPECTED_FEATURES if c in df.columns]

    if not features:
        raise ValueError("No expected predictor columns were found.")

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(X_train)

    models = {}

    # Linear Regression
    lr = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    lr.fit(X_train, y_train)
    models["Linear Regression"] = lr

    # Random Forest
    rf = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    # XGBoost
    if XGB_AVAILABLE:
        xgb = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=350,
                        max_depth=4,
                        learning_rate=0.05,
                        min_child_weight=3,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        xgb.fit(X_train, y_train)
        models["XGBoost"] = xgb

    results = []

    for name, model in models.items():
        train_metrics, train_pred = evaluate(model, X_train, y_train)
        test_metrics, test_pred = evaluate(model, X_test, y_test)

        results.append(
            {
                "Model": name,
                "Train R²": train_metrics["R²"],
                "Test R²": test_metrics["R²"],
                "Train RMSE": train_metrics["RMSE"],
                "Test RMSE": test_metrics["RMSE"],
                "Train MAE": train_metrics["MAE"],
                "Test MAE": test_metrics["MAE"],
                "Train MAPE": train_metrics["MAPE"],
                "Test MAPE": test_metrics["MAPE"],
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        "Test R²", ascending=False
    ).reset_index(drop=True)

    return models, results_df, X_train, X_test, y_train, y_test


def get_feature_importance(model, feature_names):
    """
    Returns feature importance / coefficients for the fitted pipeline.
    Works with Linear Regression, Random Forest and XGBoost.
    """
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    transformed_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_).ravel())
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    imp = pd.DataFrame(
        {
            "Feature": transformed_names,
            "Importance": values,
        }
    )

    # Remove transformer prefixes for cleaner dashboard labels.
    imp["Feature"] = (
        imp["Feature"]
        .str.replace(r"^(num|cat)__", "", regex=True)
        .str.replace("sp500_", "sp500: ", regex=False)
    )

    return imp.sort_values("Importance", ascending=False).head(15)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 Firm Sales Analytics")
    st.caption("Linear Regression • Random Forest • XGBoost")

    uploaded_file = st.file_uploader(
        "Upload firm-level CSV",
        type=["csv"],
        help="Upload the same Firm_level_data CSV used in the notebook.",
    )

    st.divider()

    test_size = st.slider(
        "Test set size",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
    )

    random_state = st.number_input(
        "Random state",
        min_value=0,
        max_value=999,
        value=1,
        step=1,
    )

    st.divider()
    st.caption("Expected target: sales")
    st.caption("Expected predictors: capital, patents, randd, employment, sp500, tobinq, value, institutions")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
if uploaded_file is None:
    st.markdown(
        '<div class="main-title">Firm Sales Prediction Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Upload your firm-level CSV from the sidebar to launch the interactive dashboard.</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This dashboard is built around the dataset and modeling workflow in your notebook: "
        "738 firms, sales as the target, Linear Regression, Random Forest and XGBoost."
    )

    st.markdown("### Dataset structure")
    preview = pd.DataFrame(
        {
            "Variable": [
                "sales",
                "capital",
                "patents",
                "randd",
                "employment",
                "sp500",
                "tobinq",
                "value",
                "institutions",
            ],
            "Role": [
                "Target",
                "Predictor",
                "Predictor",
                "Predictor",
                "Predictor",
                "Categorical predictor",
                "Predictor",
                "Predictor",
                "Predictor",
            ],
        }
    )
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.stop()


try:
    data = prepare_data(pd.read_csv(uploaded_file))
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------
try:
    models, results, X_train, X_test, y_train, y_test = train_models(
        data, test_size, random_state
    )
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

best_model_name = results.iloc[0]["Model"]
best_model = models[best_model_name]


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">Firm Sales Prediction Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Investment analytics view for understanding and predicting firm-level sales.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    metric_card("Firms", f"{len(data):,}")

with c2:
    metric_card("Avg Sales", f"{data[TARGET].mean():.2f}")

with c3:
    metric_card("Best Model", best_model_name)

with c4:
    metric_card("Best Test R²", f"{results.iloc[0]['Test R²']:.3f}")

with c5:
    metric_card("Test RMSE", f"{results.iloc[0]['Test RMSE']:.3f}")


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview",
        "🤖 Model Performance",
        "🔎 Business Drivers",
        "🎯 Sales Prediction",
        "🗃️ Data Explorer",
    ]
)


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab1:
    st.markdown('<div class="section-title">Sales Distribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            data,
            x="sales",
            nbins=30,
            marginal="box",
            title="Distribution of Firm Sales",
        )
        fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "sp500" in data.columns:
            sp_summary = (
                data.groupby("sp500", as_index=False)["sales"]
                .mean()
                .sort_values("sales", ascending=False)
            )
            fig = px.bar(
                sp_summary,
                x="sp500",
                y="sales",
                title="Average Sales by S&P 500 Status",
                text_auto=".2f",
            )
            fig.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Correlation with Sales</div>', unsafe_allow_html=True)

    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    corr = (
        data[numeric_cols]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values()
        .reset_index()
    )
    corr.columns = ["Feature", "Correlation"]

    fig = px.bar(
        corr,
        x="Correlation",
        y="Feature",
        orientation="h",
        title="Feature Correlation with Sales",
        text_auto=".2f",
    )
    fig.update_layout(template="plotly_white", height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"**Data quality:** {data.isna().sum().sum():,} missing values after target cleaning and "
        f"{data.duplicated().sum():,} duplicate rows."
    )


# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================
with tab2:
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    display_results = results.copy()
    for col in display_results.columns:
        if col != "Model":
            display_results[col] = display_results[col].round(3)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            results,
            x="Model",
            y="Test R²",
            title="Test R² — Higher is Better",
            text_auto=".3f",
        )
        fig.update_layout(template="plotly_white", yaxis_title="R²")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            results,
            x="Model",
            y="Test RMSE",
            title="Test RMSE — Lower is Better",
            text_auto=".3f",
        )
        fig.update_layout(template="plotly_white", yaxis_title="RMSE")
        st.plotly_chart(fig, use_container_width=True)

    # Actual vs predicted
    st.markdown('<div class="section-title">Actual vs Predicted — Best Model</div>', unsafe_allow_html=True)

    metrics, predictions = evaluate(best_model, X_test, y_test)

    pred_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": predictions,
        }
    )

    fig = px.scatter(
        pred_df,
        x="Actual",
        y="Predicted",
        title=f"{best_model_name}: Actual vs Predicted Sales",
        opacity=0.75,
    )

    min_val = float(min(pred_df["Actual"].min(), pred_df["Predicted"].min()))
    max_val = float(max(pred_df["Actual"].max(), pred_df["Predicted"].max()))

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
        )
    )

    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Best model selected by highest test R²: **{best_model_name}**. "
        f"Test R² = **{metrics['R²']:.3f}**, RMSE = **{metrics['RMSE']:.3f}**, "
        f"MAE = **{metrics['MAE']:.3f}**."
    )


# =========================================================
# TAB 3 — BUSINESS DRIVERS
# =========================================================
with tab3:
    st.markdown('<div class="section-title">What Drives Sales?</div>', unsafe_allow_html=True)

    importance = get_feature_importance(best_model, EXPECTED_FEATURES)

    if not importance.empty:
        fig = px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top Feature Drivers — {best_model_name}",
            text_auto=".3f",
        )
        fig.update_layout(template="plotly_white", height=520)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            importance.round(4),
            use_container_width=True,
            hide_index=True,
        )

    # Feature scatter
    available_numeric = [
        c for c in EXPECTED_FEATURES
        if c in data.columns and pd.api.types.is_numeric_dtype(data[c])
    ]

    if available_numeric:
        selected_feature = st.selectbox(
            "Explore relationship with sales",
            available_numeric,
            index=0,
        )

        fig = px.scatter(
            data,
            x=selected_feature,
            y=TARGET,
            trendline="ols",
            title=f"{selected_feature} vs Sales",
            opacity=0.65,
        )
        fig.update_layout(template="plotly_white", height=480)
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 4 — SALES PREDICTION
# =========================================================
with tab4:
    st.markdown('<div class="section-title">Predict Sales for a New Firm</div>', unsafe_allow_html=True)

    st.info(
        "Enter the firm's attributes below. The dashboard uses the selected trained model "
        "to estimate sales."
    )

    input_cols = st.columns(2)
    user_values = {}

    numeric_inputs = [
        c for c in EXPECTED_FEATURES
        if c in data.columns and pd.api.types.is_numeric_dtype(data[c])
    ]

    categorical_inputs = [
        c for c in EXPECTED_FEATURES
        if c in data.columns and not pd.api.types.is_numeric_dtype(data[c])
    ]

    idx = 0

    for feature in numeric_inputs:
        default = float(data[feature].median())
        min_v = float(data[feature].min())
        max_v = float(data[feature].max())

        # Avoid slider problems for very large ranges.
        with input_cols[idx % 2]:
            user_values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=min_v,
                max_value=max_v,
                value=default,
                format="%.4f",
            )
        idx += 1

    for feature in categorical_inputs:
        options = sorted(data[feature].dropna().astype(str).unique().tolist())
        with input_cols[idx % 2]:
            user_values[feature] = st.selectbox(
                feature.replace("_", " ").title(),
                options,
            )
        idx += 1

    selected_prediction_model = st.selectbox(
        "Prediction model",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )

    if st.button("🚀 Predict Sales", type="primary", use_container_width=True):
        prediction_model = models[selected_prediction_model]

        input_df = pd.DataFrame([user_values])
        prediction = float(prediction_model.predict(input_df)[0])

        st.success(
            f"Estimated Sales: **{prediction:.4f}** "
            f"using **{selected_prediction_model}**"
        )

        st.metric(
            "Predicted Sales",
            f"{prediction:.4f}",
        )


# =========================================================
# TAB 5 — DATA EXPLORER
# =========================================================
with tab5:
    st.markdown('<div class="section-title">Dataset Explorer</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Rows", f"{data.shape[0]:,}")

    with c2:
        st.metric("Columns", f"{data.shape[1]:,}")

    with c3:
        st.metric("Missing Values", f"{data.isna().sum().sum():,}")

    st.dataframe(
        data,
        use_container_width=True,
        height=500,
    )

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Cleaned Dataset",
        data=csv,
        file_name="firm_level_data_cleaned.csv",
        mime="text/csv",
    )
