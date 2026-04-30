import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Enable CORS (frontend connection fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# =========================
# INPUT MODEL
# =========================
class InputData(BaseModel):
    city: str
    area: float
    bedrooms: int
    bathrooms: int
    parking: int

# =========================
# PREDICT API
# =========================
@app.post("/predict")
def predict(data: InputData):
    try:
        city_encoded = encoder.transform([data.city])[0]

        features = np.array([[city_encoded, data.area, data.bedrooms, data.bathrooms, data.parking]])
        prediction = model.predict(features)[0]

        return {"predicted_rent": int(prediction)}

    except Exception as e:
        return {"error": str(e)}

# =========================
# 🔥 DYNAMIC CITY API
# =========================
@app.get("/cities")
def get_cities():
    return {"cities": list(encoder.classes_)}