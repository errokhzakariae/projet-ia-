from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy import sparse

from src.ml.train_mlp import JurisdictionMLP


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MLPPredictor:
    def __init__(
        self,
        model_path: Path,
        vectorizer_path: Path,
        label_encoder_path: Path,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(label_encoder_path)

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = JurisdictionMLP(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            output_dim=checkpoint["output_dim"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Le texte fourni est vide.")

        x = self.vectorizer.transform([text])

        if sparse.issparse(x):
            x_array = x.toarray().astype(np.float32)
        else:
            x_array = np.asarray(x, dtype=np.float32)

        features = torch.from_numpy(x_array).to(self.device)

        with torch.no_grad():
            logits = self.model(features)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_id = int(np.argmax(probabilities))
        predicted_label = str(self.label_encoder.inverse_transform([predicted_id])[0])
        confidence = float(probabilities[predicted_id])

        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": {
                str(label): float(probabilities[index])
                for index, label in enumerate(self.label_encoder.classes_)
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prédiction avec le MLP juridique.")

    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "models" / "mlp_jurisdiction.pt",
    )

    parser.add_argument(
        "--vectorizer-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "tfidf_vectorizer.pkl",
    )

    parser.add_argument(
        "--label-encoder-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "label_encoder.pkl",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="La Cour de cassation rejette le pourvoi formé contre l'arrêt attaqué.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictor = MLPPredictor(
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        label_encoder_path=args.label_encoder_path,
    )

    result = predictor.predict(args.text)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()