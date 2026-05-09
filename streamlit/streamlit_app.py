import streamlit as st
import requests
import pandas as pd

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Travel MLOps App",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Travel MLOps Application")

# ==========================================
# ✈️ FLIGHT PRICE PREDICTION
# ==========================================
st.header("✈️ Flight Price Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    week_day = st.number_input(
        "Week Day",
        min_value=1,
        max_value=7
    )

with col2:
    week_no = st.number_input(
        "Week Number",
        min_value=1,
        max_value=52
    )

with col3:
    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31
    )

if st.button("Predict Flight Price"):

    payload = {
        "week_day": week_day,
        "week_no": week_no,
        "day": day
    }

    try:

        response = requests.post(
            "http://localhost:8000/predict_price",
            json=payload
        )

        result = response.json()

        if "price" in result:

            st.success(
                f"💰 Predicted Flight Price: ₹ {result['price']:.2f}"
            )

        else:
            st.error(result)

    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

# ==========================================
# 👤 GENDER PREDICTION
# ==========================================
st.header("👤 Gender Prediction")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100
    )

with col2:
    company = st.selectbox(
        "Company",
        [
            "4You",
            "Acme Factory",
            "Monsters CYA",
            "Umbrella LTDA",
            "Wonka Company"
        ]
    )

if st.button("Predict Gender"):

    payload = {
        "age": int(age),
        "company": company
    }

    try:

        response = requests.post(
            "http://localhost:8000/predict_gender",
            json=payload
        )

        result = response.json()

        if "gender" in result:

            st.success(
                f"🧑 Predicted Gender: {result['gender']}"
            )

        else:
            st.error(result)

    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

# ==========================================
# 🏨 HOTEL RECOMMENDATION
# ==========================================
st.header("🏨 Hotel Recommendation System")

user_code = st.number_input(
    "Enter User Code",
    min_value=1
)

if st.button("Recommend Hotels"):

    payload = {
        "userCode": int(user_code)
    }

    try:

        response = requests.post(
            "http://localhost:8000/recommend",
            json=payload
        )

        result = response.json()

        if isinstance(result, list) and len(result) > 0:

            df = pd.DataFrame(result)

            st.success("🎯 Recommended Hotels")

            st.dataframe(
                df,
                use_container_width=True
            )

        else:

            st.warning(
                "No recommendations found"
            )

    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

# ==========================================
# FOOTER
# ==========================================
st.caption(
    "🚀 End-to-End MLOps Project using Flask, Streamlit, Docker, Kubernetes, Airflow & Jenkins"
)