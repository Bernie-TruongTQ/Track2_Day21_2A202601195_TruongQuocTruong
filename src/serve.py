from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
CLOUD_BUCKET = os.environ.get("CLOUD_BUCKET", "mlops-data")
MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu Azure Blob Storage ve may khi server khoi dong.
    """
    if AZURE_STORAGE_CONNECTION_STRING:
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CLOUD_BUCKET, blob=MODEL_KEY)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(blob_client.download_blob().readall())
        print("Model da duoc tai xuong tu Azure Blob Storage.")


if os.environ.get("AUTO_DOWNLOAD_MODEL", "false").lower() == "true":
    download_model()
    model = joblib.load(MODEL_PATH)
elif os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
elif os.path.exists("models/model.pkl"):
    model = joblib.load("models/model.pkl")
else:
    model = None


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    pred = int(model.predict([req.features])[0])
    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = label_map.get(pred, "khong_xac_dinh")

    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
