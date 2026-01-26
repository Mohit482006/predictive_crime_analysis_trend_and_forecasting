import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Predictive Crime Dashboard", layout="wide")

# ---------------- DARK THEME CSS ----------------
st.markdown("""
<style>
body {background-color: #0b1220;}
.block-container {padding-top: 25px;}
.card {
    background: #111827;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0 0 12px rgba(0,0,0,0.4);
    margin-bottom: 15px;
}
.blue {background: #1e3a8a; padding: 15px; border-radius: 12px; color: white; font-size: 18px;}
.green {background: #0f3d2e; padding: 15px; border-radius: 12px; color: white; font-size: 18px;}
.red {background: #3b1d1d; padding: 15px; border-radius: 12px; color: white; font-size: 18px;}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🚔 Predictive Crime Analytics Dashboard ")
st.write("Crime Trends Analysis + Crime Forecasting using Random Forest Regression")

# ---------------- LOAD DATA ----------------
# Change this path if your dataset location is different
DATA_PATH = r"NCRB_Table_1A.1 (1).csv"

try:
    df = pd.read_csv(DATA_PATH)
except:
    st.error("❌ Dataset not found! Please keep the CSV file in same folder as this dashboard.")
    st.stop()

# ---------------- CLEAN DATA ----------------
df.rename(columns={
    "Mid-Year Projected Population (in Lakhs) (2022)": "Population_2022",
    "Rate of Cognizable Crimes (IPC) (2022)": "Crime_Rate_2022",
    "Chargesheeting Rate (2022)": "Chargesheet_Rate_2022"
}, inplace=True)

for col in ["2020", "2021", "2022", "Population_2022"]:
    df[col] = df[col].astype(str).str.replace(",", "")
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# ---------------- MODEL TRAINING ----------------
X = df[["2020", "2021", "Population_2022"]]
y = df["2022"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Convert R2 score to percentage
accuracy_percent = max(0, r2) * 100

# ---------------- DASHBOARD LAYOUT ----------------
col1, col2 = st.columns(2)

# -------- LEFT PANEL --------
with col1:
    st.subheader("📊 Crime Trend Prediction")

    st.markdown(f'<div class="blue">Model Accuracy (R²): {accuracy_percent:.2f}%</div>', unsafe_allow_html=True)

    if accuracy_percent >= 70:
        st.markdown('<div class="green">✅ Model Performance: GOOD</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="red">⚠️ Model Performance: NEED IMPROVEMENT</div>', unsafe_allow_html=True)

    st.markdown("### 📌 Model Metrics")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

# -------- RIGHT PANEL (Prediction Input) --------
with col2:
    st.subheader("🔮 Predict Future Crime")

    c1, c2, c3 = st.columns(3)
    with c1:
        input_2020 = st.number_input("Crimes in 2020", value=50000)
    with c2:
        input_2021 = st.number_input("Crimes in 2021", value=52000)
    with c3:
        input_pop = st.number_input("Population 2022 (Lakhs)", value=350)

    future_df = pd.DataFrame([[input_2020, input_2021, input_pop]],
                             columns=["2020", "2021", "Population_2022"])

    prediction = model.predict(future_df)[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### 🧾 Input Values")
    st.dataframe(future_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="green">📌 Predicted Crimes (2022): {prediction:.0f}</div>', unsafe_allow_html=True)

# ---------------- GRAPHS SECTION ----------------
st.markdown("## 📈 Graphs & Visual Analysis")

g1, g2 = st.columns(2)

# Graph 1: Bar chart
with g1:
    st.markdown("### 🏙️ Crimes by State/UT (2022)")
    fig1 = plt.figure(figsize=(8,4))
    plt.bar(df["State/UT"], df["2022"])
    plt.xticks(rotation=90)
    plt.ylabel("Crime Count")
    plt.title("Crimes by State/UT in 2022")
    st.pyplot(fig1)

# Graph 2: Actual vs Predicted
with g2:
    st.markdown("### 🎯 Actual vs Predicted (2022)")
    fig2 = plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Crimes")
    plt.ylabel("Predicted Crimes")
    plt.title("Actual vs Predicted Crimes")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    st.pyplot(fig2)

# ---------------- DATASET PREVIEW ----------------
st.markdown("## 📂 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.success("✅ Dashboard Loaded Successfully!")
