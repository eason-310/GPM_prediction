import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import warnings
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

st.title("Hybrid Model for GPM Prediction with Feedback Loop")

@st.cache_data
def load_excel(file):
    return pd.read_excel(file)

def spearman_linearity_test(df, features, target, threshold=0.9):
    linear_features = []
    nonlinear_features = []
    for f in features:
        corr, _ = spearmanr(df[f], df[target])
        if np.abs(corr) >= threshold:
            linear_features.append(f)
        else:
            nonlinear_features.append(f)
    return linear_features, nonlinear_features

def train_and_evaluate(df):
    required_columns = [
        "銷售成本(原料)", "銷售成本(人工)", "銷售成本(費用)",
        "銷售成本(報廢)", "銷售成本(其他)", "毛利率"
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    features = required_columns[:-1]
    target = "毛利率"

    x = df[features].astype(float)
    y = df[target].astype(float)

    linear_features, nonlinear_features = spearman_linearity_test(df, features, target)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    linear_model = None
    if linear_features:
        linear_model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])
        linear_model.fit(x_train[linear_features], y_train)

    rf_model = None
    if nonlinear_features:
        rf_model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        rf_model.fit(x_train[nonlinear_features], y_train)

    n = len(y_test)
    linear_preds = np.zeros(n)
    nonlinear_preds = np.zeros(n)

    if linear_model:
        linear_preds = linear_model.predict(x_test[linear_features])

    if rf_model:
        nonlinear_preds = rf_model.predict(x_test[nonlinear_features])

    pred_matrix = np.vstack([linear_preds, nonlinear_preds]).T

    meta_model = LinearRegression()
    meta_model.fit(pred_matrix, y_test)

    final_preds = meta_model.predict(pred_matrix)

    def cross_val_pipeline(x, y, linear_features, nonlinear_features):
        scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(x):
            x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            lm = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())]) if linear_features else None
            rf = Pipeline([("scaler", StandardScaler()), ("regressor", RandomForestRegressor(n_estimators=100))]) if nonlinear_features else None

            if lm: lm.fit(x_tr[linear_features], y_tr)
            if rf: rf.fit(x_tr[nonlinear_features], y_tr)

            lin_pred = lm.predict(x_val[linear_features]) if lm else np.zeros(len(y_val))
            rf_pred = rf.predict(x_val[nonlinear_features]) if rf else np.zeros(len(y_val))

            preds = np.vstack([lin_pred, rf_pred]).T
            meta = LinearRegression()
            meta.fit(preds, y_val)
            score = r2_score(y_val, meta.predict(preds))
            scores.append(score)

        return np.mean(scores)

    cv_r2 = cross_val_pipeline(x, y, linear_features, nonlinear_features)

    explainer = shap.Explainer(rf_model.named_steps["regressor"], x_test) if rf_model else None

    return {
        "linear_model": linear_model,
        "rf_model": rf_model,
        "meta_model": meta_model,
        "linear_features": linear_features,
        "nonlinear_features": nonlinear_features,
        "x_test": x_test,
        "y_test": y_test,
        "final_preds": final_preds,
        "cv_r2": cv_r2,
        "explainer": explainer
    }

if "corrections" not in st.session_state:
    st.session_state["corrections"] = pd.DataFrame(columns=[
        "銷售成本(原料)", "銷售成本(人工)", "銷售成本(費用)",
        "銷售成本(報廢)", "銷售成本(其他)", "毛利率"
    ])

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = load_excel(uploaded_file)

        if not st.session_state["corrections"].empty:
            df = pd.concat([df, st.session_state["corrections"]], ignore_index=True)

        models = train_and_evaluate(df)

        st.subheader("Model Performance")
        st.write(f"Mean Squared Error: {mean_squared_error(models['y_test'], models['final_preds']):.4f}")
        st.write(f"Mean Absolute Error: {mean_absolute_error(models['y_test'], models['final_preds']):.4f}")
        st.write(f"R² Score: {r2_score(models['y_test'], models['final_preds']):.4f}")
        st.write(f"Cross-validated R² Score (5-fold): {models['cv_r2']:.4f}")

        st.sidebar.header("Input your costs to predict 毛利率 (GPM)")
        inputs = {}
        for col in ["銷售成本(原料)", "銷售成本(人工)", "銷售成本(費用)", "銷售成本(報廢)", "銷售成本(其他)"]:
            inputs[col] = st.sidebar.number_input(col, value=0.0)

        linear_pred = 0.0
        rf_pred = 0.0

        if models["linear_model"] and models["linear_features"]:
            df_input_lin = pd.DataFrame([{k: inputs[k] for k in models["linear_features"]}])
            linear_pred = models["linear_model"].predict(df_input_lin)[0]

        if models["rf_model"] and models["nonlinear_features"]:
            df_input_rf = pd.DataFrame([{k: inputs[k] for k in models["nonlinear_features"]}])
            rf_pred = models["rf_model"].predict(df_input_rf)[0]

        final_input = np.array([[linear_pred, rf_pred]])
        final_pred = models["meta_model"].predict(final_input)[0]

        st.subheader(f"Predicted 毛利率 (GPM): {final_pred:.4f}")

        with st.expander("Feedback: Correct the predicted 毛利率 if needed"):
            corrected = st.number_input("Corrected 毛利率", value=float(final_pred), format="%.4f")
            if st.button("Submit Correction"):
                new_data = {**inputs, "毛利率": corrected}
                st.session_state["corrections"] = pd.concat(
                    [st.session_state["corrections"], pd.DataFrame([new_data])],
                    ignore_index=True
                )
                st.success("Correction submitted. The model will be retrained with your feedback.")
                st.experimental_rerun()

        if models["explainer"]:
            with st.expander("Feature Importance (SHAP Summary)"):
                shap_values = models["explainer"](models["x_test"])
                st.set_option("deprecation.showPyplotGlobalUse", False)
                shap.summary_plot(shap_values, models["x_test"], show=False)
                st.pyplot(plt.gcf())
                plt.clf()

    except ValueError as ve:
        st.error(f"Data format issue: {ve}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
else:
    st.info("Please upload an Excel file to begin.")
