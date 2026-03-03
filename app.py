import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AQI Regression", page_icon="🌿", layout="wide")

st.markdown("""
<style>
body, [data-testid="stAppViewContainer"] { background-color: #0f1117; color: #ccd6f6; }
[data-testid="stSidebar"] { background-color: #1a1d2e; }
h1, h2, h3 { color: #ccd6f6; }
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 8px;
    padding: 0.6rem 1.2rem; font-weight: 700; width: 100%;
}
.stButton > button:hover { opacity: 0.85; }
div[data-testid="stMetric"] {
    background: #1e2140; border-radius: 10px;
    padding: 0.8rem; border: 1px solid #3a3f6e;
}
div[data-testid="stMetricValue"] { color: #667eea !important; font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

DATA_PATH = "1772272102861_city_day.csv"

try:
    df_raw = load_data(DATA_PATH)
except FileNotFoundError:
    uploaded = st.file_uploader("Upload city_day.csv", type="csv")
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
    else:
        st.warning("Please upload the dataset to continue.")
        st.stop()

# ── Feature columns ───────────────────────────────────────────────────────────
ALL_FEATURES = [c for c in ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]
                if c in df_raw.columns]
TARGET = "AQI"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 AQI Regression")
    st.markdown("---")

    cities = ["All Cities"] + sorted(df_raw["City"].dropna().unique().tolist())
    selected_city = st.selectbox("🏙️ City", cities)

    algo = st.selectbox("🤖 Algorithm", [
        "Linear Regression",
        "K-Nearest Neighbors",
        "Decision Tree",
        "Random Forest"
    ])

    st.markdown("#### Hyperparameters")
    hp = {}
    if algo == "K-Nearest Neighbors":
        hp["k"] = st.slider("K (Neighbors)", 1, 20, 5)
    elif algo == "Decision Tree":
        hp["max_depth"] = st.slider("Max Depth", 2, 30, 5)
        hp["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 2)
    elif algo == "Random Forest":
        hp["n_estimators"] = st.slider("Number of Trees", 10, 200, 100, step=10)
        hp["max_depth"] = st.slider("Max Depth", 2, 30, 10)

    st.markdown("#### Features")
    default_feats = [f for f in ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"] if f in ALL_FEATURES]
    selected_features = st.multiselect("Select Features", ALL_FEATURES, default=default_feats)

    test_pct = st.slider("Test Split %", 10, 40, 20)
    run = st.button("🚀 Train Model")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🌿 AQI Regression Dashboard")
st.markdown("Predict **Air Quality Index** using ML regression models on India city air quality data.")
st.markdown("---")

if not selected_features:
    st.info("👈 Please select at least one feature from the sidebar.")
    st.stop()

# ── Filter & clean ────────────────────────────────────────────────────────────
df = df_raw.copy()
if selected_city != "All Cities":
    df = df[df["City"] == selected_city]

df_model = df[selected_features + [TARGET]].dropna().reset_index(drop=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows", f"{len(df_raw):,}")
c2.metric("Usable Rows", f"{len(df_model):,}")
c3.metric("Features", str(len(selected_features)))
c4.metric("Target", TARGET)

# ── Data preview ──────────────────────────────────────────────────────────────
with st.expander("📋 Data Preview & Statistics"):
    st.dataframe(df_model.head(30), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df_model.describe().round(2), use_container_width=True)
    with col_b:
        st.markdown("**AQI Distribution**")
        fig_h, ax_h = plt.subplots(figsize=(5, 3))
        fig_h.patch.set_facecolor("#1e2140")
        ax_h.set_facecolor("#1e2140")
        ax_h.hist(df_model[TARGET].values, bins=40, color="#667eea", edgecolor="#0f1117")
        ax_h.set_xlabel("AQI", color="#8892b0")
        ax_h.set_ylabel("Count", color="#8892b0")
        ax_h.tick_params(colors="#8892b0")
        for sp in ax_h.spines.values():
            sp.set_edgecolor("#3a3f6e")
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

# ── Training ──────────────────────────────────────────────────────────────────
if run:
    if len(df_model) < 30:
        st.error(f"Not enough data ({len(df_model)} rows). Try 'All Cities' or add more features.")
        st.stop()

    X = df_model[selected_features].values
    y = df_model[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_pct / 100, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Build model
    if algo == "Linear Regression":
        model  = LinearRegression()
        Xtr, Xte = X_train_sc, X_test_sc
        color  = "#667eea"
    elif algo == "K-Nearest Neighbors":
        model  = KNeighborsRegressor(n_neighbors=hp["k"])
        Xtr, Xte = X_train_sc, X_test_sc
        color  = "#f093fb"
    elif algo == "Decision Tree":
        model  = DecisionTreeRegressor(
            max_depth=hp["max_depth"],
            min_samples_split=hp["min_samples_split"],
            random_state=42
        )
        Xtr, Xte = X_train, X_test
        color  = "#4ecdc4"
    else:
        model  = RandomForestRegressor(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            random_state=42
        )
        Xtr, Xte = X_train, X_test
        color  = "#f7dc6f"

    with st.spinner(f"Training {algo}…"):
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_test, y_pred)

    st.markdown("---")
    st.markdown(f"### 📊 Results — {algo}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", f"{r2:.4f}")
    m2.metric("MAE",      f"{mae:.2f}")
    m3.metric("RMSE",     f"{rmse:.2f}")
    m4.metric("MSE",      f"{mse:.2f}")

    # ── 3 Plots ───────────────────────────────────────────────────────────────
    st.markdown("---")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#0f1117")

    for ax in axes:
        ax.set_facecolor("#1e2140")
        ax.tick_params(colors="#8892b0")
        for sp in ax.spines.values():
            sp.set_edgecolor("#3a3f6e")

    # 1. Actual vs Predicted
    lo = min(float(y_test.min()), float(y_pred.min()))
    hi = max(float(y_test.max()), float(y_pred.max()))
    axes[0].scatter(y_test, y_pred, alpha=0.45, color=color, s=14, edgecolors="none")
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5)
    axes[0].set_xlabel("Actual AQI",    color="#8892b0")
    axes[0].set_ylabel("Predicted AQI", color="#8892b0")
    axes[0].set_title("Actual vs Predicted", color="#ccd6f6", fontweight="bold")

    # 2. Residuals
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.45, color=color, s=14, edgecolors="none")
    axes[1].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Predicted AQI", color="#8892b0")
    axes[1].set_ylabel("Residual",      color="#8892b0")
    axes[1].set_title("Residual Plot",  color="#ccd6f6", fontweight="bold")

    # 3. Error distribution
    axes[2].hist(residuals, bins=35, color=color, alpha=0.85, edgecolor="#0f1117")
    axes[2].axvline(0, color="red", lw=1.5, linestyle="--")
    axes[2].set_xlabel("Residual",          color="#8892b0")
    axes[2].set_ylabel("Count",             color="#8892b0")
    axes[2].set_title("Error Distribution", color="#ccd6f6", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Feature importance / coefficients ─────────────────────────────────────
    st.markdown("---")
    if algo in ["Decision Tree", "Random Forest"]:
        st.markdown("### 🌟 Feature Importance")
        fi = pd.DataFrame({
            "Feature":    selected_features,
            "Importance": model.feature_importances_
        }).sort_values("Importance").reset_index(drop=True)

        fig2, ax2 = plt.subplots(figsize=(8, max(3, len(selected_features) * 0.55)))
        fig2.patch.set_facecolor("#0f1117")
        ax2.set_facecolor("#1e2140")
        ax2.tick_params(colors="#8892b0")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#3a3f6e")
        ax2.barh(fi["Feature"], fi["Importance"], color=color, edgecolor="#0f1117")
        ax2.set_xlabel("Importance", color="#8892b0")
        ax2.set_title("Feature Importance", color="#ccd6f6", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    elif algo == "Linear Regression":
        st.markdown("### 📐 Feature Coefficients")
        cf = pd.DataFrame({
            "Feature":     selected_features,
            "Coefficient": model.coef_
        }).sort_values("Coefficient").reset_index(drop=True)

        fig2, ax2 = plt.subplots(figsize=(8, max(3, len(selected_features) * 0.55)))
        fig2.patch.set_facecolor("#0f1117")
        ax2.set_facecolor("#1e2140")
        ax2.tick_params(colors="#8892b0")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#3a3f6e")
        bar_c = [color if v >= 0 else "#ff6b6b" for v in cf["Coefficient"]]
        ax2.barh(cf["Feature"], cf["Coefficient"], color=bar_c, edgecolor="#0f1117")
        ax2.axvline(0, color="white", lw=0.8, linestyle="--")
        ax2.set_xlabel("Coefficient", color="#8892b0")
        ax2.set_title("Feature Coefficients", color="#ccd6f6", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Prediction Table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔢 Sample Predictions (first 25 test rows)")
    n = min(25, len(y_test))
    pred_df = pd.DataFrame({
        "Actual AQI":    np.round(y_test[:n], 2),
        "Predicted AQI": np.round(y_pred[:n], 2),
        "Error":         np.round(y_test[:n] - y_pred[:n], 2),
        "Abs Error":     np.round(np.abs(y_test[:n] - y_pred[:n]), 2),
    })
    st.dataframe(pred_df, use_container_width=True)

    # ── Custom Prediction ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔮 Custom Prediction")
    st.markdown("Enter pollutant values below to get an instant AQI prediction:")

    ncols = min(len(selected_features), 4)
    cols  = st.columns(ncols)
    user_vals = []
    for i, feat in enumerate(selected_features):
        mean_val = float(round(float(df_model[feat].mean()), 2))
        min_val  = float(df_model[feat].min())
        max_val  = float(df_model[feat].max())
        with cols[i % ncols]:
            v = st.number_input(
                feat,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                key=f"inp_{feat}"
            )
            user_vals.append(v)

    user_arr = np.array([user_vals])
    if algo in ["Linear Regression", "K-Nearest Neighbors"]:
        user_in = scaler.transform(user_arr)
    else:
        user_in = user_arr

    pred_custom = float(model.predict(user_in)[0])

    st.markdown(f"""
    <div style="margin-top:1rem; background:#1e2140; border:2px solid #667eea;
                border-radius:14px; padding:1.8rem; text-align:center;">
        <div style="color:#8892b0; font-size:0.85rem; letter-spacing:2px; text-transform:uppercase;">
            Predicted AQI
        </div>
        <div style="color:#667eea; font-size:3.5rem; font-weight:800; line-height:1.1;">
            {pred_custom:.1f}
        </div>
        <div style="color:#8892b0; font-size:0.8rem; margin-top:0.3rem;">using {algo}</div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#1e2140; border:1px dashed #3a3f6e; border-radius:16px;
                padding:3rem; text-align:center; margin-top:1.5rem;">
        <div style="font-size:3rem;">🌿</div>
        <h2 style="color:#ccd6f6; margin:0.5rem 0;">Ready to Train</h2>
        <p style="color:#8892b0;">
            Configure your settings in the sidebar, then click
            <strong style="color:#667eea;">🚀 Train Model</strong>.
        </p>
        <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:1.2rem;">
            <span style="background:#667eea33;color:#667eea;padding:5px 14px;border-radius:20px;border:1px solid #667eea55;font-size:0.85rem;">📈 Linear Regression</span>
            <span style="background:#f093fb33;color:#f093fb;padding:5px 14px;border-radius:20px;border:1px solid #f093fb55;font-size:0.85rem;">🔵 KNN</span>
            <span style="background:#4ecdc433;color:#4ecdc4;padding:5px 14px;border-radius:20px;border:1px solid #4ecdc455;font-size:0.85rem;">🌳 Decision Tree</span>
            <span style="background:#f7dc6f33;color:#f7dc6f;padding:5px 14px;border-radius:20px;border:1px solid #f7dc6f55;font-size:0.85rem;">🌲 Random Forest</span>
        </div>
    </div>
    """, unsafe_allow_html=True)