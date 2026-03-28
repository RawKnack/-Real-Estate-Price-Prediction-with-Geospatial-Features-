import joblib
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
from sklearn.metrics.pairwise import haversine_distances

# -------------------------
# BASE DIRECTORY (IMPORTANT)
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------
# PATHS (ALL FIXED)
# -------------------------
knn_path = os.path.join(BASE_DIR, "models", "knn_model.pkl")
prices_path = os.path.join(BASE_DIR, "models", "train_prices.npy")
schools_path = os.path.join(BASE_DIR, "models", "schools.geojson")
restaurants_path = os.path.join(BASE_DIR, "models", "restaurants.geojson")
tracts_path = os.path.join(BASE_DIR, "models", "tracts.geojson")
mean_coords_path = os.path.join(BASE_DIR, "models", "mean_coords.npy")

# -------------------------
# LOAD ARTIFACTS
# -------------------------
knn_model = joblib.load(knn_path)
train_prices = np.load(prices_path)

schools = gpd.read_file(schools_path).to_crs(epsg=3857)
restaurants = gpd.read_file(restaurants_path).to_crs(epsg=3857)
tracts = gpd.read_file(tracts_path).to_crs(epsg=3857)

mean_coords = np.load(mean_coords_path)
mean_lat, mean_lon = mean_coords

# -------------------------
# KNN FEATURE
# -------------------------
def compute_knn_feature(lat, lon, k=50):
    coords = np.array([[lat, lon]])
    coords_rad = np.radians(coords)

    distances, indices = knn_model.kneighbors(coords_rad, n_neighbors=k)

    weights = 1 / (distances + 1e-5)
    neighbor_prices = train_prices[indices]

    weighted_avg = (neighbor_prices * weights).sum(axis=1) / weights.sum(axis=1)

    return weighted_avg[0]

# -------------------------
# POI FEATURES
# -------------------------
def compute_poi_features(lat, lon, radius=5000):
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    point = point.to_crs(epsg=3857)
    point_geom = point.iloc[0]

    school_distances = schools.geometry.distance(point_geom)
    restaurant_distances = restaurants.geometry.distance(point_geom)

    school_count = int((school_distances <= radius).sum())
    restaurant_count = int((restaurant_distances <= radius).sum())

    return {
        "school_count_5km": school_count,
        "restaurant_count_5km": restaurant_count
    }

# -------------------------
# INCOME FEATURE
# -------------------------
def compute_income_feature(lat, lon):
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    point = point.to_crs(epsg=3857)
    point_geom = point.iloc[0]

    match = tracts[tracts.contains(point_geom)]

    if len(match) == 0:
        return None

    return match.iloc[0]["median_income"]

# -------------------------
# DISTANCE FROM MEAN
# -------------------------
def compute_distance_from_mean(lat, lon):
    coords = np.radians([[lat, lon]])
    mean_coords_rad = np.radians([[mean_lat, mean_lon]])

    distance = haversine_distances(coords, mean_coords_rad)

    return (distance * 6371000 / 1000)[0][0]  # km