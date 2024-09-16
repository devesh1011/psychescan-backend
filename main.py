from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",  # React frontend running locally
    "https://psyche-scan.vercel.app",  # If you deploy your frontend
]

# Load the pre-trained models (depression, anxiety, stress)
with open("models/depression_model.pkl", "rb") as f:
    depression_model = pickle.load(f)

with open("models/anxiety_model.pkl", "rb") as f:
    anxiety_model = pickle.load(f)

with open("models/stress_model.pkl", "rb") as f:
    stress_model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows your frontend to communicate with backend
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Define the input data structure
class PsycheScanInput(BaseModel):
    features: list


@app.post("/predict/")
def predict_psyche(input_data: PsycheScanInput):
    # Extract features from the request
    print(input_data)
    features = np.array(input_data.features).reshape(1, -1)

    # Get predictions from each model
    depression_pred = depression_model.predict(features)[0]
    anxiety_pred = anxiety_model.predict(features)[0]
    stress_pred = stress_model.predict(features)[0]

    # Return the results as JSON
    return {
        "depression": depression_pred,
        "anxiety": anxiety_pred,
        "stress": stress_pred,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
