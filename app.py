from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load trained model
with open("./svm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI
app = FastAPI()

# Define request schema
class InputData(BaseModel):
    SRS_RAW_TOTAL: float
    SRS_AWARENESS: float
    SRS_COGNITION: float
    SRS_COMMUNICATION: float
    SRS_MOTIVATION: float
    SRS_MANNERISMS: float

@app.post("/predict")
def predict(data: InputData):
    # Convert input into numpy array
    features = np.array([[data.SRS_RAW_TOTAL, data.SRS_AWARENESS, data.SRS_COGNITION, 
                          data.SRS_COMMUNICATION, data.SRS_MOTIVATION, data.SRS_MANNERISMS]])
    
    # Get model prediction
    prediction = model.predict(features)[0]

    return {"prediction": int(prediction)}

