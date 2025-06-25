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

st.title("Hybrid Model for GPM Prediction")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    x = df[["銷售成本(原料)", "銷售成本(人工)", "銷售成本(費用)", "銷售成本(報廢)", "銷售成本(其他)"]].astype(float)
    y = df["毛利率"].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Temporary model for partial dependence
    temp_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    temp_model.fit(x_train, y_train)

    linear_features = []
    nonlinear_features = []

    for feature in x_test.columns:
        try:
            pd_result = partial_dependence(
                temp_model.named_steps["regressor"],
                X=x_test.astype(float),
                features=[feature],
                grid_resolution=100
            )
            x_vals = pd_result["grid_values"][0].reshape(-1, 1)
            y_vals = pd_result["average"][0].reshape(-1, 1)

            linreg = LinearRegression()
            linreg.fit(x_vals, y_vals)
            y_pred = linreg.predict(x_vals)

            r2 = r2_score(y_vals, y_pred)

            if r2 >= 0.95:
                linear_features.append(feature)
            else:
                nonlinear_features.append(feature)

        except Exception as e:
            st.warning(f"Skipping feature {feature} due to error: {e}")

    # Train final models
    linear_model = None
    rf_model = None

    if linear_features:
        x_train_linear = x_train[linear_features]
        linear_model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])
        linear_model.fit(x_train_linear, y_train)

    if nonlinear_features:
        x_train_nonlinear = x_train[nonlinear_features]
        rf_model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        rf_model.fit(x_train_nonlinear, y_train)

    # Evaluate on test set
    linear_preds = linear_model.predict(x_test[linear_features]) if linear_model else None
    nonlinear_preds = rf_model.predict(x_test[nonlinear_features]) if rf_model else None

    if linear_preds is not None and nonlinear_preds is not None:
        final_preds = (linear_preds + nonlinear_preds) / 2
    elif linear_preds is not None:
        final_preds = linear_preds
    elif nonlinear_preds is not None:
        final_preds = nonlinear_preds
    else:
        st.error("No models trained — check your data or feature setup.")
        st.stop()

    st.subheader("Hybrid Model Performance on Test Data")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, final_preds):.4f}")
    st.write(f"R^2 Score: {r2_score(y_test, final_preds):.4f}")

    st.sidebar.header("Input your costs to predict 毛利率 (GPM)")
    raw = st.sidebar.number_input("銷售成本(原料)", value=0.0)
    labor = st.sidebar.number_input("銷售成本(人工)", value=0.0)
    expense = st.sidebar.number_input("銷售成本(費用)", value=0.0)
    scrap = st.sidebar.number_input("銷售成本(報廢)", value=0.0)
    other = st.sidebar.number_input("銷售成本(其他)", value=0.0)

    user_input = {
        "銷售成本(原料)": raw,
        "銷售成本(人工)": labor,
        "銷售成本(費用)": expense,
        "銷售成本(報廢)": scrap,
        "銷售成本(其他)": other
    }

    preds = []

    if linear_model and linear_features:
        linear_input = pd.DataFrame([{k: user_input[k] for k in linear_features}])
        preds.append(linear_model.predict(linear_input)[0])

    if rf_model and nonlinear_features:
        nonlinear_input = pd.DataFrame([{k: user_input[k] for k in nonlinear_features}])
        preds.append(rf_model.predict(nonlinear_input)[0])

    if preds:
        final_pred = sum(preds) / len(preds)
        st.subheader(f"Predicted 毛利率 (GPM): {final_pred:.2f}")
    else:
        st.warning("No trained models to make prediction.")

else:
    st.info("Please upload an Excel file to get started.")
