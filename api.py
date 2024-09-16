import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from data_processing import clean_text

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

app = FastAPI(title="Sarcasm Detection API")

# Loaded once at startup, not per-request — avoids reloading from disk on every call
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


class HeadlineRequest(BaseModel):
    headline: str


class PredictionResponse(BaseModel):
    is_sarcastic: bool
    confidence: float


@app.post("/predict", response_model=PredictionResponse)
def predict(request: HeadlineRequest):
    """
    Predict whether a news headline is sarcastic.

    Example:
        $ curl -X POST http://127.0.0.1:8000/predict \\
            -H "Content-Type: application/json" \\
            -d '{"headline": "Local Man Discovers Sarcasm, Nation Stunned"}'
        {"is_sarcastic": true, "confidence": 0.91}
    """
    cleaned = clean_text(request.headline)
    features = vectorizer.transform([cleaned])

    prediction = model.predict(features)[0]
    # predict_proba returns [P(not sarcastic), P(sarcastic)] — take the predicted class's probability
    confidence = model.predict_proba(features)[0][prediction]

    return PredictionResponse(is_sarcastic=bool(prediction), confidence=round(float(confidence), 4))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
