import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
import shap
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

st.title("Hybrid Model Predictor")

def color_metric(value, metric_type):
    if metric_type in ["mse", "mae"]:
        color = "green" if value < 0.01 else "orange" if value < 0.1 else "red"
    else:
        color = "green" if value > 0.8 else "orange" if value > 0.2 else "red"
    return f'<span style="color:{color}; font-weight:bold;">{value:.4f}</span>'

@st.cache_data
def load_excel(file, transpose=False):
    df = pd.read_excel(file)
    if transpose:
        df = df.T
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
    df = df.reset_index(drop=True)
    return df.astype(float)

def hybrid_linearity_test(df, features, target, spearman_threshold=0.3, mi_threshold=0.01):
    linear_feats = []
    nonlinear_feats = []
    spearman_corrs = df[features + [target]].corr(method="spearman")[target].drop(target)
    mi = mutual_info_regression(df[features], df[target])
    for i, f in enumerate(features):
        if abs(spearman_corrs[f]) >= spearman_threshold:
            linear_feats.append(f)
        elif mi[i] >= mi_threshold:
            nonlinear_feats.append(f)
    if not linear_feats and not nonlinear_feats:
        #st.warning("No features passed thresholds. Using all features as nonlinear.")
        nonlinear_feats = features
    return linear_feats, nonlinear_feats, spearman_corrs, pd.Series(mi, index=features)

def train_and_evaluate(df, feature_cols, target_col):
    x, y = df[feature_cols], df[target_col]
    linear_feats, nonlinear_feats, spearman_corrs, mi = hybrid_linearity_test(df, feature_cols, target_col)

    x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.25, random_state=42)

    lm = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())]) if linear_feats else None
    rf = Pipeline([("scaler", StandardScaler()), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))]) if nonlinear_feats else None

    if lm:
        lm.fit(x_train[linear_feats], y_train)
    if rf:
        rf.fit(x_train[nonlinear_feats], y_train)

    lin_pred_val = lm.predict(x_val[linear_feats]) if lm else np.zeros(len(y_val))
    rf_pred_val = rf.predict(x_val[nonlinear_feats]) if rf else np.zeros(len(y_val))
    blend_val = np.vstack([lin_pred_val, rf_pred_val]).T

    meta = XGBRegressor(objective="reg:squarederror", random_state=42)
    meta.fit(blend_val, y_val)

    lin_pred_test = lm.predict(x_test[linear_feats]) if lm else np.zeros(len(y_test))
    rf_pred_test = rf.predict(x_test[nonlinear_feats]) if rf else np.zeros(len(y_test))
    blend_test = np.vstack([lin_pred_test, rf_pred_test]).T

    final_preds = meta.predict(blend_test)

    cv_r2 = None
    if len(x_train_full) >= 5:
        def cv_blend(x, y):
            scores = []
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr, val in kf.split(x):
                xm, xv = x.iloc[tr], x.iloc[val]
                ym, yv = y.iloc[tr], y.iloc[val]

                lm2 = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())]) if linear_feats else None
                rf2 = Pipeline([("scaler", StandardScaler()), ("regressor", RandomForestRegressor(n_estimators=100))]) if nonlinear_feats else None

                if lm2:
                    lm2.fit(xm[linear_feats], ym)
                if rf2:
                    rf2.fit(xm[nonlinear_feats], ym)

                lp = lm2.predict(xv[linear_feats]) if lm2 else np.zeros(len(yv))
                rp = rf2.predict(xv[nonlinear_feats]) if rf2 else np.zeros(len(yv))

                blend = np.vstack([lp, rp]).T
                xgb2 = XGBRegressor(**meta.get_params())
                xgb2.fit(blend, yv)
                scores.append(r2_score(yv, xgb2.predict(blend)))
            return np.mean(scores)
        cv_r2 = cv_blend(x_train_full, y_train_full)

    return {
        "linear_model": lm,
        "rf_model": rf,
        "meta_model": meta,
        "linear_features": linear_feats,
        "nonlinear_features": nonlinear_feats,
        "x_test": x_test,
        "y_test": y_test,
        "final_preds": final_preds,
        "cv_r2": cv_r2,
        "spearman": spearman_corrs,
        "mutual_info": mi,
    }

if "corrections" not in st.session_state:
    st.session_state["corrections"] = pd.DataFrame()

uploaded = st.file_uploader("Upload Excel", type=["xlsx"])
transpose = st.checkbox("資料項目名稱在直行")

if uploaded:
    try:
        df = load_excel(uploaded, transpose=transpose)
        cols = df.columns.tolist()
        target = st.selectbox("Select target (選擇欲預測項目)", cols)
        features = st.multiselect("Select features (選擇用於預測項目)", [c for c in cols if c != target])

        if features:
            if not st.session_state["corrections"].empty:
                df = pd.concat([df, st.session_state["corrections"]], ignore_index=True)

            m = train_and_evaluate(df, features, target)
            if m is None:
                st.stop()

            mse = mean_squared_error(m["y_test"], m["final_preds"])
            mae = mean_absolute_error(m["y_test"], m["final_preds"])
            r2 = r2_score(m["y_test"], m["final_preds"])
            cv_r2 = m["cv_r2"]

            st.subheader("Performance")
            st.markdown(f"MSE: {color_metric(mse,'mse')}", unsafe_allow_html=True)
            st.markdown(f"MAE: {color_metric(mae,'mae')}", unsafe_allow_html=True)
            st.markdown(f"R²: {color_metric(r2,'r2')}", unsafe_allow_html=True)
            st.markdown(f"CV R²: {color_metric(cv_r2,'cv_r2')}", unsafe_allow_html=True)

            st.sidebar.header("Enter feature values")
            inp = {f: st.sidebar.number_input(f, value=0.0) for f in features}

            user_df = pd.DataFrame([inp])
            lp = m["linear_model"].predict(user_df[m["linear_features"]])[0] if m["linear_model"] else 0.0
            rp = m["rf_model"].predict(user_df[m["nonlinear_features"]])[0] if m["rf_model"] else 0.0
            final = m["meta_model"].predict(np.array([[lp, rp]]))[0]

            st.subheader(f"Predicted {target}: **{final:.4f}**")

            with st.expander("Submit correction"):
                corrected = st.number_input("Corrected value", value=float(final))
                if st.button("Submit"):
                    new = {**inp, target: corrected}
                    st.session_state["corrections"] = pd.concat([st.session_state["corrections"], pd.DataFrame([new])], ignore_index=True)
                    st.success("Correction saved, retraining now...")
                    st.experimental_rerun()

        else:
            st.info("Choose at least one feature.")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("請上傳Excel檔案以開始")
