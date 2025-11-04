import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import time

# 🧠 Page Setup
st.set_page_config(
    page_title="Smart Calorie Predictor 🍽️",
    page_icon="🍔",
    layout="wide"
)

# 🎨 Background Styling
bg_image_path = Path("C://Users//USER//calorie_app//bc.png")
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("{bg_image_path.as_posix()}");
    background-size: cover;
    background-position: center;
}}
[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
div.block-container {{
    background-color: rgba(0, 0, 0, 0.6);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    box-shadow: 0px 0px 20px rgba(255,255,255,0.1);
}}
h1, h2, h3, p, label {{ color: white !important; }}
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
input {{ color: #fff !important; }}
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

# 🌐 Sidebar Navigation
st.sidebar.title("🍽️ Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Results", "ℹ️ About"])

# 🔁 Session state
if "calorie_pred" not in st.session_state:
    st.session_state.calorie_pred = None
if "nutrients" not in st.session_state:
    st.session_state.nutrients = None
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# 🏠 Home Page
if page == "🏠 Home":
    st.title("🍔 Smart Calorie Predictor")
    st.write("Enter the nutritional values below to estimate **total calories per serving**.")

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

    if st.button("🔥 Predict Calories"):
        try:
            with st.spinner("🔮 Predicting... please wait a moment..."):
                time.sleep(2)  # simulate model loading for better UX
                
                input_data = np.array([[fat, saturated_fats, protein, carbohydrates, sugars, dietary_fiber, sodium]])
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                calorie_pred = prediction[0]

            st.session_state.calorie_pred = calorie_pred
            st.session_state.nutrients = {
                "Fat": fat,
                "Saturated Fats": saturated_fats,
                "Protein": protein,
                "Carbohydrates": carbohydrates,
                "Sugars": sugars,
                "Dietary Fiber": dietary_fiber,
                "Sodium": sodium
            }

            # Navigate to Results
            st.session_state.page = "📊 Results"
            st.experimental_rerun()
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

# 📊 Results Page
elif page == "📊 Results" or st.session_state.page == "📊 Results":
    st.title("📊 Prediction Results")

    if st.session_state.calorie_pred is None:
        st.warning("⚠️ No prediction yet. Please go to the Home page and make a prediction first.")
    else:
        calorie_pred = st.session_state.calorie_pred
        nutrients = st.session_state.nutrients

        st.subheader(f"✅ Estimated Calories: **{calorie_pred:.2f} kcal**")

        # 🍱 Calorie Category
        if calorie_pred < 100:
            st.info("🥗 Low Calorie Food")
        elif calorie_pred < 500:
            st.warning("🍱 Moderate Calorie Food")
        else:
            st.error("🍕 High Calorie Food")

        st.write("---")
        st.subheader("🍎 Nutrient Breakdown")

        # 📈 Bar Chart
        fig1, ax1 = plt.subplots()
        ax1.barh(list(nutrients.keys()), list(nutrients.values()), color="#ff9800")
        ax1.set_xlabel("Amount (g or mg)")
        ax1.set_title("Nutritional Composition")
        st.pyplot(fig1)

        # 🥧 Pie Chart
        st.write("---")
        st.subheader("🥧 Nutrient Proportion")

        nonzero_nutrients = {k: v for k, v in nutrients.items() if v > 0}
        if len(nonzero_nutrients) > 0:
            fig2, ax2 = plt.subplots()
            ax2.pie(
                nonzero_nutrients.values(),
                labels=nonzero_nutrients.keys(),
                autopct="%1.1f%%",
                startangle=90,
                colors=plt.cm.Paired.colors
            )
            ax2.axis("equal")
            st.pyplot(fig2)
        else:
            st.info("No nutrient values entered for pie chart display.")

        st.write("---")

        # 🔁 Try Again Button
        if st.button("🔁 Try Again"):
            st.session_state.page = "🏠 Home"
            st.rerun()

# ℹ️ About Page
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.write("""
    This app uses a **Machine Learning model** to predict total calories from food nutrient composition.
    
    **Technologies Used:**
    - Python 🐍  
    - Streamlit ⚡  
    - Scikit-learn 🤖  
    - Matplotlib 📊  
    
    Created with ❤️ by [Psly](https://github.com/psly1705)
    """)
    st.write("Go to the **Home** page to start predicting!")
