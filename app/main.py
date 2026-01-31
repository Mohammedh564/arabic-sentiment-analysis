from fastapi import FastAPI
from pydantic import BaseModel

from inference.sentiment_predictor import SentimentPredictor
from preprocessing.arabic_preprocessor import arabicPreprocessor


# -------------------------
# Request schema
# -------------------------
class SentimentRequest(BaseModel):
    text: str


# -------------------------
# App & dependencies
# -------------------------
app = FastAPI(title="Arabic Sentiment Analysis API")

predictor = SentimentPredictor()
preprocessor = arabicPreprocessor()


# -------------------------
# Routes
# -------------------------
@app.post("/predict_sentiment")
def predict_sentiment(request: SentimentRequest):
    processed_text = preprocessor.preprocess(request.text)
    predicted_class_id, label_name, confidence = predictor.predict(processed_text)

    return {
        "predicted_class_id": predicted_class_id,
        "label_name": label_name,
        "confidence": confidence
    }


# -------------------------
# Optional local run
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
# To run the app, use the command:
# uvicorn app.main:app --reload