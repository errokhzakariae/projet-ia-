from pathlib import Path

from src.ml.predict import MLPPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[3]


predictor = MLPPredictor(
    model_path=PROJECT_ROOT / "artifacts" / "models" / "mlp_jurisdiction.pt",
    vectorizer_path=PROJECT_ROOT / "data" / "processed" / "tfidf_vectorizer.pkl",
    label_encoder_path=PROJECT_ROOT / "data" / "processed" / "label_encoder.pkl",
)


def classify_text(text: str) -> dict:
    return predictor.predict(text)