from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_price

app = FastAPI()


class HouseInput(BaseModel):
    bedroomcnt: float
    bathroomcnt: float
    calculatedfinishedsquarefeet: float
    latitude: float
    longitude: float


@app.get("/")
def home():
    return {"message": "Real Estate Price Prediction API is running"}


@app.get("/hi")
def home():
    return {"message": "hi"}

@app.post("/predict")
def predict(data: HouseInput):
    input_dict = data.dict()

    prediction = predict_price(input_dict)

    return {
        "predicted_price": float(prediction)
    }