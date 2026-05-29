from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from src.ml.train_mlp import JurisdictionMLP, SparseTextDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalue le MLP sur le test set.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "models" / "mlp_jurisdiction.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "models" / "test_metrics.json",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    label_encoder = joblib.load(args.data_dir / "label_encoder.pkl")

    test_df = pd.read_csv(args.data_dir / "test.csv")
    x_test = sparse.load_npz(args.data_dir / "features" / "X_test_tfidf.npz")
    test_dataset = SparseTextDataset(x_test, test_df["label_id"].to_numpy())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JurisdictionMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for features, batch_labels in test_loader:
            logits = model(features.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(batch_labels.numpy().tolist())

    report = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(label_encoder.classes_))),
            target_names=list(label_encoder.classes_),
            output_dict=True,
            zero_division=0,
        ),
        "support": int(len(labels)),
        "classes": list(label_encoder.classes_),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Metriques test ecrites: {args.output}")


if __name__ == "__main__":
    main()

