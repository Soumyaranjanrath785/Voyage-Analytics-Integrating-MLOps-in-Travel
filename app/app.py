# =========================
# 📦 IMPORTS
# =========================
import pandas as pd
import joblib
import os
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# =========================
# 📁 BASE PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

# =========================
# 📦 LOAD MODELS & DATA
# =========================

# Flight model
model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))

# Gender model (simple version)
gender_model = joblib.load(os.path.join(MODEL_DIR, "gender_model.pkl"))
company_encoder = joblib.load(os.path.join(MODEL_DIR, "company_encoder.pkl"))

# Recommendation data
user_hotel_matrix = joblib.load(
    os.path.join(MODEL_DIR, "user_hotel_matrix.pkl")
)

similarity_df = joblib.load(
    os.path.join(MODEL_DIR, "similarity_df.pkl")
)

# =========================
# 🏠 HOME
# =========================
@app.route("/")
def home():
    return "MLOps API Running 🚀"

# =========================
# ✈️ FLIGHT PRICE PREDICTION
# =========================
@app.route("/predict_price", methods=["POST"])
def predict_price():
    try:
        data = request.json

        # Ensure correct format
        df = pd.DataFrame([data])
        df = df.reindex(columns=columns, fill_value=0)

        X = scaler.transform(df)
        prediction = model.predict(X)

        return jsonify({"price": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# 👤 GENDER PREDICTION (SIMPLIFIED)
# =========================
@app.route("/predict_gender", methods=["POST"])
def predict_gender():
    try:
        data = request.json

        age = data["age"]
        company = data["company"]

        # Encode company properly
        company_encoded = company_encoder.transform([company])[0]

        X = np.array([[company_encoded, age]])

        pred = gender_model.predict(X)
        gender = "male" if pred[0] == 1 else "female"

        return jsonify({"gender": gender})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# 🏨 HOTEL RECOMMENDATION
# =========================
# =========================
# 🏨 HOTEL RECOMMENDATION
# =========================
@app.route("/recommend", methods=["POST"])
def recommend():

    try:

        user_id = request.json["userCode"]

        # Check user exists
        if user_id not in user_hotel_matrix.index:
            return jsonify([])

        # =========================
        # FIND SIMILAR USERS
        # =========================
        similar_users = similarity_df[user_id] \
            .sort_values(ascending=False)

        similar_users = similar_users.drop(user_id)

        # =========================
        # HOTELS ALREADY VISITED
        # =========================
        visited_hotels = set(
            user_hotel_matrix.loc[user_id][
                user_hotel_matrix.loc[user_id] > 0
            ].index
        )

        recommendations = {}

        # =========================
        # GENERATE RECOMMENDATIONS
        # =========================
        for sim_user in similar_users.index:

            sim_user_hotels = user_hotel_matrix.loc[sim_user]

            for hotel, score in sim_user_hotels.items():

                if score > 0 and hotel not in visited_hotels:

                    if hotel not in recommendations:
                        recommendations[hotel] = 0

                    recommendations[hotel] += score

        # =========================
        # TOP 5 HOTELS
        # =========================
        recommended_hotels = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        result = [
            {
                "hotel": hotel,
                "score": float(score)
            }
            for hotel, score in recommended_hotels
        ]

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    print("🚀 Starting Flask API...")
    app.run(host="0.0.0.0", port=8000, debug=True)