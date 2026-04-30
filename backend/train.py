import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# =========================
# STEP 1: LOAD DATA (SAFE FOR LARGE FILE)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "rent_data.csv")

# Load only 50k rows to avoid crash (you can increase later)
df = pd.read_csv(file_path, nrows=50000)

print("✅ Data Loaded")
print("Columns:", df.columns)

# =========================
# STEP 2: CLEAN COLUMN NAMES
# =========================
df.columns = df.columns.str.strip().str.lower()

# Rename columns automatically
df.rename(columns={
    "bhk": "bedrooms",
    "size": "area",
    "bathroom": "bathrooms",
    "rent": "rent",
    "city": "city"
}, inplace=True)

# =========================
# STEP 3: HANDLE DIRTY DATA
# =========================

# Convert area like "1100 sqft" → 1100
if "area" in df.columns:
    df["area"] = df["area"].astype(str).str.extract(r'(\d+)').astype(float)

# Convert bedrooms like "2 BHK" → 2
if "bedrooms" in df.columns:
    df["bedrooms"] = df["bedrooms"].astype(str).str.extract(r'(\d+)').astype(int)

# Bathrooms safe conversion
if "bathrooms" in df.columns:
    df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors='coerce')

# Add parking if missing
if "parking" not in df.columns:
    df["parking"] = 1

# =========================
# STEP 4: REMOVE BAD DATA
# =========================
df.dropna(inplace=True)

df = df[df["rent"] > 2000]
df = df[df["rent"] < 200000]

# Keep only required columns
df = df[['city', 'area', 'bedrooms', 'bathrooms', 'parking', 'rent']]

print("\n✅ Cleaned Data Shape:", df.shape)
print(df.head())

# =========================
# STEP 5: ENCODE CITY
# =========================
le = LabelEncoder()
df['city'] = le.fit_transform(df['city'])

# =========================
# STEP 6: SPLIT DATA
# =========================
X = df[['city', 'area', 'bedrooms', 'bathrooms', 'parking']]
y = df['rent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# STEP 7: TRAIN MODEL (ADVANCED)
# =========================
model = RandomForestRegressor(
    n_estimators=120,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\n✅ Model Trained")

# =========================
# STEP 8: CHECK ACCURACY
# =========================
score = model.score(X_test, y_test)
print(f"📊 Accuracy (R² Score): {score:.2f}")

# =========================
# STEP 9: SAVE MODEL
# =========================
model_path = os.path.join(BASE_DIR, "backend", "model.pkl")
encoder_path = os.path.join(BASE_DIR, "backend", "encoder.pkl")

pickle.dump(model, open(model_path, "wb"))
pickle.dump(le, open(encoder_path, "wb"))

print("💾 Model & Encoder Saved")

# =========================
# STEP 10: TEST PREDICTION
# =========================
sample_city = df['city'].iloc[0]

sample = [[sample_city, 1000, 2, 2, 1]]
prediction = model.predict(sample)

print(f"\n🏠 Sample Prediction: ₹{int(prediction[0])}")