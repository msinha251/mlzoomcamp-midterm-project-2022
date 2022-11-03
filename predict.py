from fastapi import FastAPI
from pydantic import BaseModel
from ml.data import basic_preprocess
from ml.model import predict_single
import pandas as pd
import pickle
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

app = FastAPI()

class PassengerDetails(BaseModel):
    PassengerId : str
    HomePlanet  : str
    CryoSleep   : bool
    Cabin       : str
    Destination : str
    Age         : float
    VIP         : bool
    RoomService : float
    FoodCourt   : float
    ShoppingMall: float
    Spa         : float
    VRDeck      : float
    Names       : str

    class Config:
        schema_extra = {
            "example": {
                "PassengerId": "0013_01",
                "HomePlanet": "Earth",
                "CryoSleep": True,
                "Cabin": "G/3/S",
                "Destination": "TRAPPIST-1e",
                "Age": 27.0,
                "VIP": False,
                "RoomService": 0.0,
                "FoodCourt": 0.0,
                "ShoppingMall": 0.0,
                "Spa": 0.0,
                "VRDeck": 0.0,
                "Names": "Nelly Carsoning"
            }
        }

#TODO - from config
xgb = True
if xgb:
    model_path = './models/xgb_model.pkl'
else:
    model_path = './models/rf_model.pkl'


@app.on_event("startup")
async def startup_event():
    global model, dv
    with open(model_path, "rb") as f_in:
        logging.info(f"Loading {model_path} Model...")
        model, dv = pickle.load(f_in)
    logging.info("Model and dv LOADED")

@app.get("/")
async def root():
    return "Welcome to the Spaceship Titanic API. Please use '/docs' to see the API documentation or testing out the API"

@app.post("/predict")
def predict(passenger: PassengerDetails):
    logging.info(f"Received passenger details: {passenger}")

    # predict
    message = predict_single(passenger.dict(), model, dv, xgb=xgb)
    return {"message": message}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



    