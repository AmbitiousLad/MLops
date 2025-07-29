import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model
model = mlflow.sklearn.load_model("models/house_price_model")

app = FastAPI()

# Define request body
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict_price(features: HouseFeatures):
    data = pd.DataFrame([features.dict().values()], columns=features.dict().keys())
    prediction = model.predict(data)[0]
    return {"Predicted_Median_House_Value": prediction}
