import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# 🧠 Page Setup
st.set_page_config(
    page_title="Smart Calorie Predictor 🍽️",
    page_icon="🍔",
    layout="centered"
)

# 🎨 Background Styling
bg_image_path = Path("bc.png")

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("{bg_image_path.as_posix()}");
    background-size: cover;
    background-position: center;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0.3);
}}

div.block-container {{
    background-color: rgba(0, 0, 0, 0.65);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    box-shadow: 0px 0px 20px rgba(255,255,255,0.1);
}}

h1, h2, h3, p, label {{
    color: white !important;
}}

div.stButton > button:first-child {{
    background-color: #ff9800;
    color: white;
    border: none;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
    transition: 0.3s;
}}

div.stButton > button:hover {{
    background-color: #e68900;
    transform: scale(1.05);
}}

input {{
    color: #fff !important;
}}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# 🧩 Load model and scaler
try:
    model = joblib.load("calories_predictor.joblib")
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    st.error(f"⚠️ Error loading model or scaler: {e}")
    st.stop()

# 🏷️ Title & Description
st.title("🍔 Smart Calorie Predictor")
st.write("Enter the nutritional values below to estimate **total calories per serving**.")

# 📊 Input Fields
col1, col2 = st.columns(2)

with col1:
    fat = st.number_input("Fat (g)", min_value=0.0)
    saturated_fats = st.number_input("Saturated Fats (g)", min_value=0.0)
    protein = st.number_input("Protein (g)", min_value=0.0)
    carbohydrates = st.number_input("Carbohydrates (g)", min_value=0.0)

with col2:
    sugars = st.number_input("Sugars (g)", min_value=0.0)
    dietary_fiber = st.number_input("Dietary Fiber (g)", min_value=0.0)
    sodium = st.number_input("Sodium (mg)", min_value=0.0)

# 🔮 Prediction Button
if st.button("🔥 Predict Calories"):
    input_data = np.array([[fat, saturated_fats, protein, carbohydrates, sugars, dietary_fiber, sodium]])
    
    try:
        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        calorie_pred = prediction[0]

        # 💡 Display Result
        st.success(f"✅ Estimated Calories: **{calorie_pred:.2f} kcal**")

        # 🍕 Category feedback
        if calorie_pred < 100:
            st.info("🥗 Low Calorie Food")
        elif calorie_pred < 500:
            st.warning("🍱 Moderate Calorie Food")
        else:
            st.error("🍕 High Calorie Food")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
