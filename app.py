import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.parse

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="ERE Digital Twin", layout="wide")

SAFE_TEMP = 52
CRITICAL_TEMP = 65

# -----------------------------
# WHATSAPP FUNCTION
# -----------------------------
def get_whatsapp_link(phone, message):
    encoded_msg = urllib.parse.quote(message)
    return f"https://wa.me/{phone}?text={encoded_msg}"

# -----------------------------
# LOAD MODEL + DATA
# -----------------------------
model = joblib.load("model.pkl")

df = pd.read_csv("engine_failure_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(r"\(.*\)", "", regex=True).str.strip().str.replace(" ", "_")

# Feature engineering
df["Vibration_Total"] = np.sqrt(
    df["Vibration_X"]**2 +
    df["Vibration_Y"]**2 +
    df["Vibration_Z"]**2
)

df["Thermal_Stress"] = df["Temperature"] * df["Power_Output"]
df["Efficiency"] = df["Power_Output"] / df["Power_Output"].max()

# -----------------------------
# NAVIGATION
# -----------------------------
st.sidebar.title("⚡ Navigation")
section = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "⚙ Engine Control", "🧠 AI Prediction", "📊 Analytics"]
)

# -----------------------------
# OVERVIEW
# -----------------------------
if section == "🏠 Overview":
    st.title("⚡ Electromagnetic Engine Digital Twin")

    st.markdown("""
    ### System Features:
    - Digital Twin Simulation  
    - AI Fault Prediction  
    - Thermal Safety Monitoring  
    - WhatsApp Alert System  
    """)

# -----------------------------
# ENGINE CONTROL
# -----------------------------
elif section == "⚙ Engine Control":
    st.title("⚙ Engine Control Panel")

    col1, col2, col3 = st.columns(3)

    with col1:
        voltage = st.slider("Voltage (V)", 5, 24, 12)

    with col2:
        duty_cycle = st.slider("Duty Cycle (%)", 10, 100, 50)

    with col3:
        load = st.slider("Load Factor", 1, 10, 5)

    # Simulation
    rpm = (voltage * duty_cycle) / load * 10
    temperature = (duty_cycle * 0.5) + (load * 5)
    power_output = voltage * duty_cycle * 0.1
    torque = power_output / (rpm + 1)

    st.subheader("🔄 Digital Twin Output")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RPM", round(rpm, 2))
    c2.metric("Temperature (°C)", round(temperature, 2))
    c3.metric("Power Output", round(power_output, 2))
    c4.metric("Torque", round(torque, 2))

# -----------------------------
# AI PREDICTION
# -----------------------------
elif section == "🧠 AI Prediction":
    st.title("🧠 AI-Based Engine Health")

    voltage = st.slider("Voltage (V)", 5, 24, 12)
    duty_cycle = st.slider("Duty Cycle (%)", 10, 100, 50)
    load = st.slider("Load Factor", 1, 10, 5)

    # Simulation
    rpm = (voltage * duty_cycle) / load * 10
    temperature = (duty_cycle * 0.5) + (load * 5)
    power_output = voltage * duty_cycle * 0.1
    torque = power_output / (rpm + 1)
    vibration_total = load * 0.5
    thermal_stress = temperature * power_output
    efficiency = power_output / (power_output + 50)

    # Model input
    input_data = pd.DataFrame([[
        temperature, rpm, torque, power_output,
        vibration_total, thermal_stress, efficiency
    ]], columns=[
        "Temperature", "RPM", "Torque", "Power_Output",
        "Vibration_Total", "Thermal_Stress", "Efficiency"
    ])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data).max()

    status_map = {
        0: "Healthy",
        1: "Minor Issue",
        2: "Moderate Fault",
        3: "Critical Failure"
    }

    st.subheader("🔍 Prediction")
    st.write(f"Status: {status_map[prediction]}")
    st.write(f"Confidence: {round(prob*100,2)}%")

    # -----------------------------
    # THERMAL SAFETY
    # -----------------------------
    st.subheader("🌡 Thermal Status")

    if temperature < SAFE_TEMP:
        st.success(f"Safe ({temperature:.2f} °C)")
    elif SAFE_TEMP <= temperature < CRITICAL_TEMP:
        st.warning(f"High Temperature ({temperature:.2f} °C)")
    else:
        st.error(f"CRITICAL OVERHEATING ({temperature:.2f} °C)")

    # -----------------------------
    # ALERT + WHATSAPP
    # -----------------------------
    if prediction >= 2 or temperature >= SAFE_TEMP:
        st.error("🚨 ALERT: Engine issue detected!")

        phone_number = "919682728712"  # 🔁 replace with your number

        msg = f"""
⚠ Engine Alert!
Status: {status_map[prediction]}
Temperature: {round(temperature,2)} °C
RPM: {round(rpm,2)}
"""

        whatsapp_link = get_whatsapp_link(phone_number, msg)

        st.link_button("📱 Send WhatsApp Alert", whatsapp_link)

    # -----------------------------
    # RECOMMENDATIONS
    # -----------------------------
    st.subheader("💡 Recommendations")

    if temperature >= SAFE_TEMP:
        st.warning("Reduce duty cycle or voltage")

    if temperature >= CRITICAL_TEMP:
        st.error("Immediate shutdown required")

    if rpm < 100:
        st.info("Increase voltage or duty cycle")

# -----------------------------
# ANALYTICS
# -----------------------------
elif section == "📊 Analytics":
    st.title("📊 Data Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=df, x="Temperature", y="Power_Output", ax=ax1)
        ax1.set_title("Power vs Temperature")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x="Temperature", y="RPM", ax=ax2)
        ax2.set_title("RPM vs Temperature")
        st.pyplot(fig2)

    st.subheader("🧠 Feature Importance")

    importances = model.feature_importances_
    features = [
        "Temperature", "RPM", "Torque", "Power_Output",
        "Vibration_Total", "Thermal_Stress", "Efficiency"
    ]

    fig3, ax3 = plt.subplots()
    sns.barplot(x=importances, y=features, ax=ax3)
    st.pyplot(fig3)