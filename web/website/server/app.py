from typing import List
from fastapi import FastAPI, File, UploadFile
import uvicorn
from send_mail import send_mail

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
async def upload_image(file: UploadFile = File(...)):

    # Consume the upload stream so we’re sure the full file arrived
    chunk_size = 1024 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
    await file.close()

    mail = send_mail()

    return {"message": "image is taken"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)
