import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
st.title("Hybrid Model Predictor (")

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

def hybrid_linearity_test(df, features, target, spearman_threshold=0.7, mi_threshold=0.1):
    linear_feats = []
    nonlinear_feats = []
    spearman_corrs = df[features + [target]].corr(method="spearman")[target].drop(target)
    mi = mutual_info_regression(df[features], df[target])
    for i, f in enumerate(features):
        if abs(spearman_corrs[f]) >= spearman_threshold:
            linear_feats.append(f)
        elif mi[i] >= mi_threshold:
            nonlinear_feats.append(f)
    return linear_feats, nonlinear_feats, spearman_corrs, pd.Series(mi, index=features)

def train_and_evaluate(df, feature_cols, target_col, use_extended_features=False):
    x, y = df[feature_cols], df[target_col]
    x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    linear_feats, nonlinear_feats, spearman_corrs, mi = hybrid_linearity_test(
        pd.concat([x_train_full, y_train_full], axis=1), feature_cols, target_col
    )

    if not linear_feats and not nonlinear_feats:
        st.error("No features passed the dependency thresholds.")
        return None

    lm = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())]) if linear_feats else None
    rf = Pipeline([("scaler", StandardScaler()), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))]) if nonlinear_feats else None

    def get_stacked_features(x_data):
        lin_pred = lm.predict(x_data[linear_feats]) if lm else np.zeros(len(x_data))
        rf_pred = rf.predict(x_data[nonlinear_feats]) if rf else np.zeros(len(x_data))
        base_preds = np.vstack([lin_pred, rf_pred]).T
        if use_extended_features:
            return np.hstack([x_data, base_preds])
        return base_preds

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    stack_train = []
    meta_targets = []
    for tr, val in kf.split(x_train_full):
        xm, xv = x_train_full.iloc[tr], x_train_full.iloc[val]
        ym, yv = y_train_full.iloc[tr], y_train_full.iloc[val]

        lm_cv = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())]) if linear_feats else None
        rf_cv = Pipeline([("scaler", StandardScaler()), ("regressor", RandomForestRegressor(n_estimators=100))]) if nonlinear_feats else None
        if lm_cv: lm_cv.fit(xm[linear_feats], ym)
        if rf_cv: rf_cv.fit(xm[nonlinear_feats], ym)

        lin_pred = lm_cv.predict(xv[linear_feats]) if lm_cv else np.zeros(len(xv))
        rf_pred = rf_cv.predict(xv[nonlinear_feats]) if rf_cv else np.zeros(len(xv))
        base_preds = np.vstack([lin_pred, rf_pred]).T
        stack_features = np.hstack([xv, base_preds]) if use_extended_features else base_preds

        stack_train.append(stack_features)
        meta_targets.append(yv)

    X_meta = np.vstack(stack_train)
    y_meta = np.hstack(meta_targets)

    xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
    grid = GridSearchCV(xgb, {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1]
    }, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X_meta, y_meta)
    meta_model = grid.best_estimator_

    if lm: lm.fit(x_train_full[linear_feats], y_train_full)
    if rf: rf.fit(x_train_full[nonlinear_feats], y_train_full)

    X_test_meta = get_stacked_features(x_test)
    final_preds = meta_model.predict(X_test_meta)

    cv_r2 = r2_score(y_meta, grid.predict(X_meta))

    return {
        "linear_model": lm, "rf_model": rf, "meta_model": meta_model,
        "linear_features": linear_feats, "nonlinear_features": nonlinear_feats,
        "x_test": x_test, "y_test": y_test, "final_preds": final_preds,
        "cv_r2": cv_r2,
        "spearman": spearman_corrs, "mutual_info": mi
    }

if "corrections" not in st.session_state:
    st.session_state["corrections"] = pd.DataFrame()

uploaded = st.file_uploader("Upload Excel", type=["xlsx"])
transpose = st.checkbox("資料項目名稱在直行")
use_extended = st.checkbox("Use Level-1 Extended Features", value=True)

if uploaded:
    try:
        df = load_excel(uploaded, transpose=transpose)
        cols = df.columns.tolist()
        target = st.selectbox("Select target (選擇欲預測項目)", cols)
        features = st.multiselect("Select features (選擇用於預測項目)", [c for c in cols if c != target])

        if features:
            if not st.session_state["corrections"].empty:
                df = pd.concat([df, st.session_state["corrections"]], ignore_index=True)

            m = train_and_evaluate(df, features, target, use_extended)

            if m is None: st.stop()

            mse = mean_squared_error(m["y_test"], m["final_preds"])
            mae = mean_absolute_error(m["y_test"], m["final_preds"])
            r2 = r2_score(m["y_test"], m["final_preds"])
            cv_r2 = m["cv_r2"]

            st.subheader("Feature Dependency Scores")
            score_df = pd.DataFrame({
                "Spearman Corr": m["spearman"],
                "Mutual Info": m["mutual_info"]
            }).sort_values("Mutual Info", ascending=False)
            st.dataframe(score_df.style.background_gradient(axis=0))

            st.subheader("Performance")
            st.markdown(f"MSE: {color_metric(mse,'mse')}", unsafe_allow_html=True)
            st.markdown(f"MAE: {color_metric(mae,'mae')}", unsafe_allow_html=True)
            st.markdown(f"R²: {color_metric(r2,'r2')}", unsafe_allow_html=True)
            st.markdown(f"CV R²: {color_metric(cv_r2,'cv_r2')}", unsafe_allow_html=True)

            st.sidebar.header("Enter feature values")
            inp = {f: st.sidebar.number_input(f, value=0.0) for f in features}
            lp = m["linear_model"].predict(pd.DataFrame([inp]))[0] if m["linear_model"] else 0.0
            rp = m["rf_model"].predict(pd.DataFrame([inp]))[0] if m["rf_model"] else 0.0
            meta_input = np.array([[*inp.values(), lp, rp]]) if use_extended else np.array([[lp, rp]])
            final = m["meta_model"].predict(meta_input)[0]
            st.subheader(f"Predicted {target}: **{final:.4f}**")

            with st.expander("Submit correction"):
                corrected = st.number_input("Corrected value", value=float(final))
                if st.button("Submit"):
                    new = {**inp, target: corrected}
                    st.session_state["corrections"] = pd.concat(
                        [st.session_state["corrections"], pd.DataFrame([new])], ignore_index=True)
                    st.success("Correction saved, retraining now...")
                    st.experimental_rerun()
        else:
            st.info("Choose at least one feature.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload an Excel file to begin.")
