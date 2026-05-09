import pandas as pd
import numpy as np
import os
import joblib

from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, 'data', 'hotels.csv')

df = pd.read_csv(file_path)

# =========================
# USER-HOTEL MATRIX
# =========================
user_hotel_matrix = df.pivot_table(
    index='userCode',
    columns='name',
    values='total',
    aggfunc='sum',
    fill_value=0
)

# =========================
# COSINE SIMILARITY
# =========================
user_similarity = cosine_similarity(user_hotel_matrix)

similarity_df = pd.DataFrame(
    user_similarity,
    index=user_hotel_matrix.index,
    columns=user_hotel_matrix.index
)

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend_hotels(user_id, top_n=5):

    if user_id not in user_hotel_matrix.index:
        return []

    # Similar users
    similar_users = similarity_df[user_id].sort_values(ascending=False)

    similar_users = similar_users.drop(user_id)

    # Hotels already visited
    visited_hotels = set(
        df[df['userCode'] == user_id]['name']
    )

    recommendations = {}

    for sim_user in similar_users.index:

        sim_user_hotels = df[df['userCode'] == sim_user]

        for _, row in sim_user_hotels.iterrows():

            hotel = row['name']

            if hotel not in visited_hotels:

                if hotel not in recommendations:
                    recommendations[hotel] = 0

                recommendations[hotel] += row['total']

    # Top recommendations
    recommended_hotels = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return [
        {"hotel": hotel, "score": score}
        for hotel, score in recommended_hotels
    ]


# =========================
# SAVE FILES
# =========================

MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    user_hotel_matrix,
    os.path.join(MODEL_DIR, "user_hotel_matrix.pkl")
)

joblib.dump(
    similarity_df,
    os.path.join(MODEL_DIR, "similarity_df.pkl")
)

print("✅ Recommendation model saved")


# =========================
# TEST
# =========================
if __name__ == "__main__":

    user_id = df['userCode'].iloc[0]

    print("\n🎯 Recommendations:\n")

    print(recommend_hotels(user_id))