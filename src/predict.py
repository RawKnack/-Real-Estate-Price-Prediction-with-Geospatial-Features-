import pandas as pd
from src.features import compute_knn_feature, compute_poi_features, compute_income_feature, compute_distance_from_mean
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = joblib.load(model_path)
def predict_price(input_data):
    lat = float(input_data["latitude"])
    lon = float(input_data["longitude"])

    knn_feature = compute_knn_feature(lat, lon)
    poi_features = compute_poi_features(lat, lon)
    income = compute_income_feature(lat, lon)
    distance = compute_distance_from_mean(lat, lon)

    if income is None:
        income = 60000

    features = {
        "bathroomcnt": float(input_data["bathroomcnt"]),
        "bedroomcnt": float(input_data["bedroomcnt"]),
        "calculatedfinishedsquarefeet": float(input_data["calculatedfinishedsquarefeet"]),
        "latitude": lat,
        "longitude": lon,
        "distance_from_mean": float(distance),
        "weighted_avg_neighbor_price": float(knn_feature),
        "school_count_5km": int(poi_features["school_count_5km"]),
        "restaurant_count_5km": int(poi_features["restaurant_count_5km"]),
        "median_income": float(income)
    }

    df = pd.DataFrame([features])

    df = df[[
        'bathroomcnt',
        'bedroomcnt',
        'calculatedfinishedsquarefeet',
        'latitude',
        'longitude',
        'distance_from_mean',
        'weighted_avg_neighbor_price',
        'school_count_5km',
        'restaurant_count_5km',
        'median_income'
    ]]

    prediction = model.predict(df)[0]

    return float(prediction)