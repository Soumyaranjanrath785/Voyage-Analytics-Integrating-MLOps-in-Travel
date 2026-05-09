import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, 'data', 'users.csv')
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(file_path)

df = df[df['gender'].isin(['male', 'female'])]

# Encode
gender_encoder = LabelEncoder()
df['gender'] = gender_encoder.fit_transform(df['gender'])

company_encoder = LabelEncoder()
df['company'] = company_encoder.fit_transform(df['company'])

X = df[['company', 'age']]
y = df['gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save
joblib.dump(model, os.path.join(MODEL_DIR, "gender_model.pkl"))
joblib.dump(company_encoder, os.path.join(MODEL_DIR, "company_encoder.pkl"))

print("✅ Gender model trained (simple)")