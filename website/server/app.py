from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from uuid import uuid4

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fire Prediction API"}


@app.post("/predict")
def predict_fire(data: List[float]):
    # Placeholder for fire prediction logic
    # In a real application, you would implement the model prediction here
    return {"prediction": "success"}


@app.post("/upload-image")
async def upload_image():

    # Dummy ID for frontend to use
    file_id = str(uuid4())

    return {"file": {"id": file_id}}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
