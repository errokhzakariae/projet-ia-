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
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SparseTextDataset(Dataset):
    def __init__(self, features: sparse.csr_matrix, labels: np.ndarray):
        self.features = features.tocsr()
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.features[index].toarray().ravel().astype(np.float32)
        return torch.from_numpy(row), torch.tensor(self.labels[index], dtype=torch.long)


class JurisdictionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraine le MLP PyTorch sur TF-IDF.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "models" / "mlp_jurisdiction.pt",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_features(data_dir: Path) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    x_train = sparse.load_npz(data_dir / "features" / "X_train_tfidf.npz")
    x_val = sparse.load_npz(data_dir / "features" / "X_val_tfidf.npz")
    return x_train, x_val


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            predictions = logits.argmax(dim=1).cpu().numpy().tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average="macro")
    return accuracy, f1_macro, all_labels, all_predictions


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    train_df = pd.read_csv(args.data_dir / "train.csv")
    val_df = pd.read_csv(args.data_dir / "val.csv")
    label_encoder = joblib.load(args.data_dir / "label_encoder.pkl")
    x_train, x_val = load_features(args.data_dir)

    train_dataset = SparseTextDataset(x_train, train_df["label_id"].to_numpy())
    val_dataset = SparseTextDataset(x_val, val_df["label_id"].to_numpy())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JurisdictionMLP(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=len(label_encoder.classes_),
        dropout=args.dropout,
    ).to(device)

    class_counts = train_df["label_id"].value_counts().sort_index().to_numpy()
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_accuracy, val_f1, val_labels, val_predictions = evaluate_model(
            model, val_loader, device
        )
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "val_accuracy": float(val_accuracy),
            "val_f1_macro": float(val_f1),
        }
        history.append(epoch_metrics)
        print(epoch_metrics)

        if val_f1 > best_f1:
            best_f1 = val_f1
            args.model_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": x_train.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "output_dim": len(label_encoder.classes_),
                    "dropout": args.dropout,
                    "classes": list(label_encoder.classes_),
                    "best_val_f1_macro": float(best_f1),
                },
                args.model_output,
            )

    report = classification_report(
        val_labels,
        val_predictions,
        labels=list(range(len(label_encoder.classes_))),
        target_names=list(label_encoder.classes_),
        output_dict=True,
        zero_division=0,
    )
    metrics_output = args.model_output.with_suffix(".metrics.json")
    metrics_output.write_text(
        json.dumps({"history": history, "validation_report": report}, indent=2),
        encoding="utf-8",
    )

    print(f"Meilleur modele ecrit: {args.model_output}")
    print(f"Metriques ecrites: {metrics_output}")


if __name__ == "__main__":
    main()

