"""
Firm-Level Sales Regression Explorer
=====================================
An interactive Streamlit UI for the EDA + Linear Regression workflow
originally written as a Jupyter/Colab notebook.

Run with:
    pip install streamlit pandas numpy scikit-learn statsmodels seaborn matplotlib scipy --break-system-packages
    streamlit run app.py
"""

import io

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

sns.set()

st.set_page_config(page_title="Firm-Level Sales Regression Explorer", layout="wide")

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
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="violet")
    if bins:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins)
    else:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")
    return f2


def labeled_barplot(data, feature, figsize=(6, 6), perc=False, n=None):
    total = len(data[feature])
    fig, ax = plt.subplots(figsize=figsize)
    plt.xticks(rotation=90)
    sns.countplot(data=data, x=feature, order=data[feature].value_counts().index[:n], ax=ax)
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


# --------------------------------------------------------------------------
# Sidebar: data loading & settings
# --------------------------------------------------------------------------

st.sidebar.title("⚙️ Settings")

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
    st.title("📊 Firm-Level Sales Regression Explorer")
    st.info("Upload a CSV file in the sidebar (or tick 'Use a small built-in sample dataset') to get started.")
    st.stop()

target_col = st.sidebar.selectbox(
    "Target (dependent) variable",
    options=data.columns.tolist(),
    index=data.columns.get_loc("sales") if "sales" in data.columns else 0,
)

test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=1, step=1)
p_threshold = st.sidebar.slider("Backward-elimination p-value threshold", 0.01, 0.20, 0.05, 0.01)
drop_na_target = st.sidebar.checkbox(f"Drop rows with missing '{target_col}'", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("This app mirrors the EDA → cleaning → OLS regression → diagnostics "
                    "workflow of the original notebook, wrapped in an interactive UI.")

# --------------------------------------------------------------------------
# Main title
# --------------------------------------------------------------------------

st.title("📊 Firm-Level Sales Regression Explorer")

df = data.copy()
if drop_na_target and target_col in df.columns:
    df = df.dropna(subset=[target_col])

tab_overview, tab_eda, tab_prep, tab_model, tab_diag, tab_predict = st.tabs(
    ["Overview", "EDA", "Preprocessing", "Model", "Diagnostics", "Predict"]
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

    st.subheader("Preview")
    n_rows = st.slider("Rows to preview", 5, 50, 10)
    st.dataframe(data.head(n_rows), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Data types")
        info_df = pd.DataFrame(
            {"dtype": data.dtypes.astype(str), "non-null": data.notnull().sum(), "nulls": data.isnull().sum()}
        )
        st.dataframe(info_df, use_container_width=True)
    with col_b:
        st.subheader("Descriptive statistics")
        st.dataframe(data.describe().T, use_container_width=True)

    st.subheader("Unique value counts")
    st.dataframe(data.nunique().rename("unique values"), use_container_width=True)

# --------------------------------------------------------------------------
# EDA tab
# --------------------------------------------------------------------------
with tab_eda:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.subheader("Univariate: Histogram + Boxplot")
    if numeric_cols:
        feat = st.selectbox("Numeric feature", numeric_cols, key="hist_feat")
        kde = st.checkbox("Show KDE curve", value=False)
        fig = histogram_boxplot(df, feat, kde=kde)
        st.pyplot(fig)
        plt.close(fig)

    if cat_cols:
        st.subheader("Categorical feature distribution")
        cfeat = st.selectbox("Categorical feature", cat_cols, key="bar_feat")
        perc = st.checkbox("Show percentages", value=True)
        fig2 = labeled_barplot(df, cfeat, perc=perc)
        st.pyplot(fig2)
        plt.close(fig2)

    if cat_cols and numeric_cols:
        st.subheader("Numeric feature by category")
        c1, c2 = st.columns(2)
        with c1:
            x_cat = st.selectbox("Category (x)", cat_cols, key="box_x")
        with c2:
            y_num = st.selectbox("Numeric (y)", numeric_cols, key="box_y")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=df, x=x_cat, y=y_num, ax=ax3)
        st.pyplot(fig3)
        plt.close(fig3)

    st.subheader("Correlation heatmap")
    if len(numeric_cols) > 1:
        fig4, ax4 = plt.subplots(figsize=(min(15, 1.2 * len(numeric_cols)), 7))
        sns.heatmap(df[numeric_cols].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral", ax=ax4)
        st.pyplot(fig4)
        plt.close(fig4)

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
        st.pyplot(fig5)
        plt.close(fig5)

# --------------------------------------------------------------------------
# Preprocessing tab
# --------------------------------------------------------------------------
with tab_prep:
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

    st.session_state["x_train"] = x_train
    st.session_state["x_test"] = x_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["X_columns"] = X.columns.tolist()

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

    st.subheader("Multicollinearity (VIF)")
    try:
        vif_df = checking_vif(x_train)
        st.dataframe(vif_df.sort_values("VIF", ascending=False), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute VIF: {e}")

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

    st.session_state["olsmodel_final"] = olsmodel2
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

    st.subheader("Fitted vs residuals")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.residplot(data=df_pred, x="Fitted Values", y="Residuals", color="purple", lowess=True, ax=ax)
    ax.set_title("Fitted vs Residual plot")
    st.pyplot(fig)
    plt.close(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Residual distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df_pred, x="Residuals", kde=True, ax=ax2)
        ax2.set_title("Normality of residuals")
        st.pyplot(fig2)
        plt.close(fig2)
    with c2:
        st.subheader("Q-Q plot")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        (osm, osr), (slope, intercept, r) = stats.probplot(df_pred["Residuals"], dist="norm")
        ax3.scatter(osm, osr, s=10)
        ax3.plot(osm, slope * osm + intercept, color="red")
        ax3.set_title("Q-Q plot")
        st.pyplot(fig3)
        plt.close(fig3)

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

    st.subheader("Residuals table")
    st.dataframe(df_pred.head(20), use_container_width=True)

# --------------------------------------------------------------------------
# Predict tab
# --------------------------------------------------------------------------
with tab_predict:
    if "olsmodel_final" not in st.session_state:
        st.warning("Please visit the Model tab first.")
        st.stop()

    model = st.session_state["olsmodel_final"]
    selected_features = st.session_state["selected_features"]

    st.subheader("Predict for a new observation")
    st.caption("Enter values for the features selected by backward elimination.")

    input_vals = {}
    n_cols_grid = 3
    cols = st.columns(n_cols_grid)
    for i, feat in enumerate(selected_features):
        col = cols[i % n_cols_grid]
        with col:
            if feat == "const":
                input_vals[feat] = 1.0
                st.write("const = 1 (intercept)")
            elif feat.endswith("_yes") or set(df.get(feat.rsplit("_", 1)[0], pd.Series(dtype=object)).unique() or []) <= {0, 1}:
                input_vals[feat] = 1.0 if st.checkbox(feat, value=False) else 0.0
            else:
                default_val = float(df[feat].mean()) if feat in df.columns else 0.0
                input_vals[feat] = st.number_input(feat, value=round(default_val, 2))

    if st.button("Predict", type="primary"):
        new_row = pd.DataFrame([input_vals])[selected_features]
        prediction = model.predict(new_row)
        st.success(f"Predicted {target_col}: **{prediction.iloc[0]:.4f}**")
        st.dataframe(new_row, use_container_width=True)
