import joblib
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
# load artifacts (done once when file is imported)
knn_model = joblib.load("../models/knn_model.pkl")
train_prices = np.load("../models/train_prices.npy")


def compute_knn_feature(lat, lon, k=50):
    # 1. convert to radians (VERY IMPORTANT)
    coords = np.array([[lat, lon]])
    coords_rad = np.radians(coords)

    # 2. get neighbors
    distances, indices = knn_model.kneighbors(coords_rad, n_neighbors=k)

    # 3. compute weights
    weights = 1 / (distances + 1e-5)

    # 4. get neighbor prices
    neighbor_prices = train_prices[indices]

    # 5. weighted average
    weighted_avg = (neighbor_prices * weights).sum(axis=1) / weights.sum(axis=1)

    return weighted_avg[0]


# paths
schools_path = "../models/schools.geojson"
restaurants_path = "../models/restaurants.geojson"
# load geo data
schools = gpd.read_file(schools_path)
restaurants = gpd.read_file(restaurants_path)

# ensure correct CRS
schools = schools.to_crs(epsg=3857)
restaurants = restaurants.to_crs(epsg=3857)
def compute_poi_features(lat, lon, radius=5000):
    # 1. create point
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    
    # 2. convert to meters CRS
    point = point.to_crs(epsg=3857)
    point_geom = point.iloc[0]

    # 3. compute distances (vectorized)
    school_distances = schools.geometry.distance(point_geom)
    restaurant_distances = restaurants.geometry.distance(point_geom)

    # 4. count within radius
    school_count = (school_distances <= radius).sum()
    restaurant_count = (restaurant_distances <= radius).sum()

    return {
        "school_count_5km": school_count,
        "restaurant_count_5km": restaurant_count
    }
tracts_path = "../models/tracts.geojson"

tracts = gpd.read_file(tracts_path)

# ensure CRS matches
tracts = tracts.to_crs(epsg=3857)
def compute_income_feature(lat, lon):
    # 1. create point
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    
    # 2. convert CRS
    point = point.to_crs(epsg=3857)
    point_geom = point.iloc[0]

    # 3. find containing tract
    match = tracts[tracts.contains(point_geom)]

    if len(match) == 0:
        return None  # or default value

    return match.iloc[0]["median_income"]
