from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import MinMaxScaler

# Load trained model
with open("./svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("./svm_dyslexia_model.pkl", "rb") as f:
    model1 = pickle.load(f)

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    print(data)
    # Convert input into numpy array
    features = np.array([[data.SRS_RAW_TOTAL, data.SRS_AWARENESS, data.SRS_COGNITION, 
                          data.SRS_COMMUNICATION, data.SRS_MOTIVATION, data.SRS_MANNERISMS]])
    
    # Get model prediction
    prediction = model.predict(features)[0]

    return {"prediction": int(prediction)}

class InputData1(BaseModel):
    Gender: str
    Nativelang: str
    Otherlang: str
    Age: int
    Clicks1: int
    Hits1: int
    Misses1: int
    Score1: int
    Accuracy1: float
    Missrate1: float
    Clicks2: int
    Hits2: int
    Misses2: int
    Score2: int
    Accuracy2: float
    Missrate2: float
    Clicks3: int
    Hits3: int
    Misses3: int
    Score3: int
    Accuracy3: float
    Missrate3: float
    Clicks4: int
    Hits4: int
    Misses4: int
    Score4: int
    Accuracy4: float
    Missrate4: float
    Clicks5: int
    Hits5: int
    Misses5: int
    Score5: int
    Accuracy5: float
    Missrate5: float
    Clicks6: int
    Hits6: int
    Misses6: int
    Score6: int
    Accuracy6: float
    Missrate6: float
    Clicks7: int
    Hits7: int
    Misses7: int
    Score7: int
    Accuracy7: float
    Missrate7: float
    Clicks8: int
    Hits8: int
    Misses8: int
    Score8: int
    Accuracy8: float
    Missrate8: float
    Clicks9: int
    Hits9: int
    Misses9: int
    Score9: int
    Accuracy9: float
    Missrate9: float
    Clicks10: int
    Hits10: int
    Misses10: int
    Score10: int
    Accuracy10: float
    Missrate10: float
    Clicks11: int
    Hits11: int
    Misses11: int
    Score11: int
    Accuracy11: float
    Missrate11: float
    Clicks12: int
    Hits12: int
    Misses12: int
    Score12: int
    Accuracy12: float
    Missrate12: float
    Clicks13: int
    Hits13: int
    Misses13: int
    Score13: int
    Accuracy13: float
    Missrate13: float
    Clicks14: int
    Hits14: int
    Misses14: int
    Score14: int
    Accuracy14: float
    Missrate14: float
    Clicks15: int
    Hits15: int
    Misses15: int
    Score15: int
    Accuracy15: float
    Missrate15: float
    Clicks16: int
    Hits16: int
    Misses16: int
    Score16: int
    Accuracy16: float
    Missrate16: float
    Clicks17: int
    Hits17: int
    Misses17: int
    Score17: int
    Accuracy17: float
    Missrate17: float
    Clicks18: int
    Hits18: int
    Misses18: int
    Score18: int
    Accuracy18: float
    Missrate18: float
    Clicks19: int
    Hits19: int
    Misses19: int
    Score19: int
    Accuracy19: float
    Missrate19: float
    Clicks20: int
    Hits20: int
    Misses20: int
    Score20: int
    Accuracy20: float
    Missrate20: float
    Clicks21: int
    Hits21: int
    Misses21: int
    Score21: int
    Accuracy21: float
    Missrate21: float
    Clicks22: int
    Hits22: int
    Misses22: int
    Score22: int
    Accuracy22: float
    Missrate22: float
    Clicks23: int
    Hits23: int
    Misses23: int
    Score23: int
    Accuracy23: float
    Missrate23: float
    Clicks24: int
    Hits24: int
    Misses24: int
    Score24: int
    Accuracy24: float
    Missrate24: float
    Clicks25: int
    Hits25: int
    Misses25: int
    Score25: int
    Accuracy25: float
    Missrate25: float
    Clicks26: int
    Hits26: int
    Misses26: int
    Score26: int
    Accuracy26: float
    Missrate26: float
    Clicks27: int
    Hits27: int
    Misses27: int
    Score27: int
    Accuracy27: float
    Missrate27: float
    Clicks28: int
    Hits28: int
    Misses28: int
    Score28: int
    Accuracy28: float
    Missrate28: float
    Clicks29: int
    Hits29: int
    Misses29: int
    Score29: int
    Accuracy29: float
    Missrate29: float
    Clicks30: int
    Hits30: int
    Misses30: int
    Score30: int
    Accuracy30: float
    Missrate30: float
    Clicks31: int
    Hits31: int
    Misses31: int
    Score31: int
    Accuracy31: float
    Missrate31: float
    Clicks32: int
    Hits32: int
    Misses32: int
    Score32: int
    Accuracy32: float
    Missrate32: float
    
   

@app.post("/dyslexia-predict")
def predictDyslexia(data:InputData1):
    p=[]
    keys=[]
    for i in data:
        p.append(i[1])
        keys.append(i[0])
    features = np.array([p])
    X = np.nan_to_num(features)
    
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = np.nan_to_num(X[i][j])


    X[:, 0] = np.where(X[:, 0] == 'Male', 0, 1)

    X[:, 1] = np.where(X[:, 1] == 'Yes', 1, 0)
    X[:, 2] = np.where(X[:, 2] == 'Yes', 1, 0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    for i, feature in enumerate(keys):
        if not feature.startswith('Accuracy'):
            column_values = X[:, i].astype(float).reshape(-1, 1)
            X[:, i] = scaler.fit_transform(column_values).flatten()
    
    predicted_probabilities = model1.predict_proba(X.astype(float))[0]
    return 1 if predicted_probabilities[1] >=0.5 else 0
    
