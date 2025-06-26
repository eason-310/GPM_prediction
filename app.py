import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import partial_dependence
import shap
import warnings

warnings.filterwarnings("ignore")

st.title("Hybrid Model for GPM Prediction")

@st.cache_data
def load_excel(file):
    return pd.read_excel(file)

@st.cache_resource
def train_models(df):
    required_columns = [
        "銷售成本(原料)", "銷售成本(人工)", "銷售成本(費用)",
        "銷售成本(報廢)", "銷售成本(其他)", "毛利率"
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    x = df[required_columns[:-1]].astype(float)
    y = df["毛利率"].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    temp_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    temp_model.fit(x_train, y_train)

    linear_features, nonlinear_features = [], []

    for feature in x.columns:
        try:
            pd_result = partial_dependence(temp_model.named_steps["regressor"], X=x_test, features=[feature], grid_resolution=100)
            x_vals = pd_result["grid_values"][0].reshape(-1, 1)
            y_vals = pd_result["average"][0].reshape(-1, 1)

            linreg = LinearRegression()
            linreg.fit(x_vals, y_vals)
            r2 = r2_score(y_vals, linreg.predict(x_vals))

            if r2 >= 0.95:
                linear_features.append(feature)
            else:
                nonlinear_features.append(feature)
        except:
            continue

    linear_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]) if linear_features else None

    if linear_model:
        linear_model.fit(x_train[linear_features], y_train)

    rf_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ]) if nonlinear_features else None

    if rf_model:
        rf_model.fit(x_train[nonlinear_features], y_train)

    linear_preds = linear_model.predict(x_test[linear_features]) if linear_model else None
    nonlinear_preds = rf_model.predict(x_test[nonlinear_features]) if rf_model else None

    pred_matrix = np.vstack([
        linear_preds if linear_preds is not None else np.zeros(len(y_test)),
        nonlinear_preds if nonlinear_preds is not None else np.zeros(len(y_test))
    ]).T

    meta_model = LinearRegression()
    meta_model.fit(pred_matrix, y_test)

    final_preds = meta_model.predict(pred_matrix)

    return {
        "linear_model": linear_model,
        "rf_model": rf_model,
        "meta_model": meta_model,
        "linear_features": linear_features,
        "nonlinear_features": nonlinear_features,
        "x_test": x_test,
        "y_test": y_test,
        "final_preds": final_preds,
        "explainer": shap.Explainer(temp_model.named_steps["regressor"], x_test)
    }

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = load_excel(uploaded_file)
        models = train_models(df)

        st.subheader("Model Performance")
        st.write(f"Mean Squared Error: {mean_squared_error(models['y_test'], models['final_preds']):.4f}")
        st.write(f"R^2 Score: {r2_score(models['y_test'], models['final_preds']):.4f}")

        st.sidebar.header("Input your costs to predict 毛利率 (GPM)")
        inputs = {}
        for col in ["銷售成本(原料)", "銷售成本(人工)", "銷售成本(費用)", "銷售成本(報廢)", "銷售成本(其他)"]:
            inputs[col] = st.sidebar.number_input(col, value=0.0)

        preds = []
        if models["linear_model"] and models["linear_features"]:
            df_input_lin = pd.DataFrame([{k: inputs[k] for k in models["linear_features"]}])
            preds.append(models["linear_model"].predict(df_input_lin)[0])

        if models["rf_model"] and models["nonlinear_features"]:
            df_input_rf = pd.DataFrame([{k: inputs[k] for k in models["nonlinear_features"]}])
            preds.append(models["rf_model"].predict(df_input_rf)[0])

        if preds:
            final_input = np.array(preds).reshape(1, -1)
            final_pred = models["meta_model"].predict(final_input)[0]
            st.subheader(f"Predicted 毛利率 (GPM): {final_pred:.2f}")
        else:
            st.warning("No valid prediction — check model training or input.")

        with st.expander("Feature Importance (SHAP Summary)"):
            shap_values = models["explainer"](models["x_test"])
            #st.set_option("deprecation.showPyplotGlobalUse", False)
            shap.summary_plot(shap_values, models["x_test"], show=False)
            st.pyplot()

    except ValueError as ve:
        st.error(f"Data format issue: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Please upload an Excel file to start.")
