# 🏡 Real Estate Price Prediction using Geospatial Intelligence

An end-to-end machine learning system that predicts real estate prices using structural features and advanced geospatial intelligence.

---

## 🚀 Live Demo

* 🌍 **API Endpoint:** https://real-estate-api-v07z.onrender.com
* 📘 **API Docs (Swagger):** https://real-estate-api-v07z.onrender.com/docs

---

## 📌 Problem Statement

Accurately predicting real estate prices requires more than just property attributes — **location and neighborhood context play a crucial role**.

This project builds a production-ready ML system that leverages:

* Property features (bedrooms, area, etc.)
* **Geospatial relationships**
* External datasets (Census + POIs)

---

## 🧠 Key Innovation

### 📍 KNN-based Geospatial Feature

Instead of using raw latitude/longitude, we engineered:

> **Weighted Average Neighbor Price**

* Uses K-Nearest Neighbors (KNN)
* Distance-based weighting: `1 / distance`
* Captures **true neighborhood pricing trends**

👉 This significantly improved model performance over baseline models.

---

## 🗂️ Dataset

* Source: Zillow Prize Dataset (2016)
* Target variable: `taxvaluedollarcnt` (property value)

---

## ⚙️ Features Used

### 🏠 Structural Features

* bedroomcnt
* bathroomcnt
* calculatedfinishedsquarefeet

### 🌍 Geospatial Features

* latitude, longitude
* distance_from_mean (Haversine distance)
* weighted_avg_neighbor_price (KNN-based)

### 🏫 Points of Interest (OpenStreetMap)

* school_count_5km
* restaurant_count_5km

### 💰 Socioeconomic Data (US Census)

* median_income

---

## 📈 Model Performance

| Model                 | R² Score  |
| --------------------- | --------- |
| Linear Regression     | ~0.37     |
| + Geospatial Features | ~0.48     |
| LightGBM (Final)      | **~0.55** |

---

## ⚠️ Observations & Limitations

* Performs well on **average-priced homes**
* Struggles with **luxury properties** due to data imbalance
* External features show **diminishing returns beyond a point**

---

## 🏗️ System Architecture

```
User Input
   ↓
Feature Engineering Pipeline
   ↓
KNN Geo Feature + POI + Census
   ↓
LightGBM Model
   ↓
Prediction Output (API)
```

---

## 🧪 API Usage

### Endpoint

```http
POST /predict
```

### Request Body

```json
{
  "bedroomcnt": 3,
  "bathroomcnt": 2,
  "calculatedfinishedsquarefeet": 1500,
  "latitude": 34.05,
  "longitude": -118.25
}
```

### Response

```json
{
  "predicted_price": 270420.42
}
```

---

## 🛠️ Tech Stack

* **Python**
* **LightGBM**
* **scikit-learn**
* **GeoPandas**
* **FastAPI**
* **Docker**
* **Render (Deployment)**

---

## 📦 Project Structure

```
Real_Estate_Price_Prediction/
│
├── models/              # Saved models & artifacts
├── src/
│   ├── features.py      # Feature engineering
│   ├── predict.py       # Prediction pipeline
│   ├── app.py           # FastAPI app
│
├── notebooks/           # Experimentation
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🐳 Docker Setup

### Build Image

```bash
docker build -t real-estate-api .
```

### Run Container

```bash
docker run -p 8000:8000 real-estate-api
```

---

## ☁️ Deployment

* Containerized using Docker
* Deployed on **Render**
* Public API available for real-time predictions

---

## 🔮 Future Improvements

* Handle luxury house predictions (log-transform / separate models)
* Add more POIs (hospitals, parks, transit)
* Optimize geospatial queries (vectorization / indexing)
* Add caching for faster inference
* Build frontend map interface (Folium / React)

---

## 📚 Key Learnings

* Feature engineering > model complexity
* Spatial relationships are critical in real estate
* External data improves performance but with diminishing returns
* Production ML requires **pipeline consistency and deployment readiness**

---

## 👨‍💻 Author

**Raunak Suman**

* GitHub: https://github.com/your-username
* LinkedIn: https://linkedin.com/in/your-profile

---

## ⭐ If you found this useful

Give the repo a ⭐ and feel free to contribute!
