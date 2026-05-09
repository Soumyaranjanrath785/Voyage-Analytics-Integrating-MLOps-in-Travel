import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 📁 Create model folder
os.makedirs("model", exist_ok=True)

# 📂 Load data (FIXED PATH)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, 'data', 'flights.csv')
df = pd.read_csv(file_path)

# Feature engineering
df['date'] = pd.to_datetime(df['date'])
df['week_day'] = df['date'].dt.weekday
df['month'] = df['date'].dt.month
df['week_no'] = df['date'].dt.isocalendar().week.astype(int)
df['year'] = df['date'].dt.year
df['day'] = df['date'].dt.day

df.rename(columns={"to": "destination"}, inplace=True)

df['flight_speed'] = round(df['distance'] / df['time'], 2)

df = pd.get_dummies(df, columns=['from', 'destination', 'flightType', 'agency'])

df.drop(columns=['time', 'flight_speed', 'month', 'year', 'distance'], inplace=True)

# Features & target
X = df.drop('price', axis=1)
y = df['price']

# Rename columns
X.rename(columns={
    'from_Sao Paulo (SP)': 'from_Sao_Paulo (SP)',
    'from_Rio de Janeiro (RJ)': 'from_Rio_de_Janeiro (RJ)',
    'from_Campo Grande (MS)': 'from_Campo_Grande (MS)',
    'destination_Sao Paulo (SP)': 'destination_Sao_Paulo (SP)',
    'destination_Rio de Janeiro (RJ)': 'destination_Rio_de_Janeiro (RJ)',
    'destination_Campo Grande (MS)': 'destination_Campo_Grande (MS)'
}, inplace=True)

# Feature ordering
features_ordering = [
    'from_Florianopolis (SC)','from_Sao_Paulo (SP)','from_Salvador (BH)','from_Brasilia (DF)',
    'from_Rio_de_Janeiro (RJ)','from_Campo_Grande (MS)','from_Aracaju (SE)','from_Natal (RN)','from_Recife (PE)',
    'destination_Florianopolis (SC)','destination_Sao_Paulo (SP)','destination_Salvador (BH)','destination_Brasilia (DF)',
    'destination_Rio_de_Janeiro (RJ)','destination_Campo_Grande (MS)','destination_Aracaju (SE)',
    'destination_Natal (RN)','destination_Recife (PE)',
    'flightType_economic','flightType_firstClass','flightType_premium',
    'agency_Rainbow','agency_CloudFy','agency_FlyingDrops',
    'week_no','week_day','day'
]

X = X[features_ordering]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ MLflow (LOCAL — NO SERVER REQUIRED)
mlflow.set_experiment("flight-price")

with mlflow.start_run():

    param_dict = {
        'n_estimators': [300],
        'max_depth': [15],
        'min_samples_split': [10],
        'max_features': ['sqrt', 27]
    }

    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, param_grid=param_dict, cv=3, scoring='r2')
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print("RMSE:", rmse)
    print("R2:", r2)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, "model")

# 💾 Save for API
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(features_ordering, os.path.join(MODEL_DIR, "columns.pkl"))


print("✅ Model trained + saved")