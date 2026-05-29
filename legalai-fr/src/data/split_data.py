from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split train/val/test et exporte le vectorizer TF-IDF."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "mlp_dataset.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )

    parser.add_argument(
        "--vectorizer-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "tfidf_vectorizer.pkl",
    )

    parser.add_argument(
        "--label-encoder-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "label_encoder.pkl",
    )

    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "split_report.json",
    )

    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=1.0)

    return parser.parse_args()


def can_stratify(df: pd.DataFrame) -> bool:
    counts = df["label"].value_counts()
    return counts.min() >= 3 and len(df) >= 10


def split_dataset(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_size + val_size + test_size

    if abs(total - 1.0) > 1e-6:
        raise ValueError("Les ratios train/val/test doivent sommer à 1.0.")

    stratify_first = df["label"] if can_stratify(df) else None

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify_first,
    )

    relative_val_size = val_size / (val_size + test_size)

    stratify_second = temp_df["label"] if can_stratify(temp_df) else None

    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        random_state=random_state,
        stratify=stratify_second,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in df["label"].value_counts().sort_index().items()
    }


def save_split_files(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(output_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(output_dir / "test.csv", index=False, encoding="utf-8")


def vectorize_texts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int,
    min_df: int,
    max_df: float,
) -> tuple[TfidfVectorizer, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
        lowercase=True,
    )

    x_train = vectorizer.fit_transform(train_df["text"])
    x_val = vectorizer.transform(val_df["text"])
    x_test = vectorizer.transform(test_df["text"])

    return vectorizer, x_train, x_val, x_test


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Fichier introuvable : {args.input}")

    df = pd.read_csv(args.input)

    required_columns = {"source_id", "label", "text"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Colonnes manquantes : {missing_columns}")

    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])

    train_df, val_df, test_df = split_dataset(
        df=df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    save_split_files(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=args.output_dir,
    )

    vectorizer, x_train, x_val, x_test = vectorize_texts(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
    )

    joblib.dump(vectorizer, args.vectorizer_output)
    joblib.dump(label_encoder, args.label_encoder_output)

    features_dir = args.output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(features_dir / "X_train_tfidf.npz", x_train)
    sparse.save_npz(features_dir / "X_val_tfidf.npz", x_val)
    sparse.save_npz(features_dir / "X_test_tfidf.npz", x_test)

    report = {
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "total": int(len(df)),
        },
        "label_distribution": {
            "full": label_counts(df),
            "train": label_counts(train_df),
            "val": label_counts(val_df),
            "test": label_counts(test_df),
        },
        "label_mapping": {
            label: int(index)
            for index, label in enumerate(label_encoder.classes_)
        },
        "stratification_used": can_stratify(df),
        "tfidf": {
            "max_features": args.max_features,
            "ngram_range": [1, 2],
            "min_df": args.min_df,
            "max_df": args.max_df,
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)

    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Splits écrits dans : {args.output_dir}")
    print(f"Vectorizer TF-IDF écrit : {args.vectorizer_output}")
    print(f"Label encoder écrit : {args.label_encoder_output}")
    print(f"Features TF-IDF écrites dans : {features_dir}")
    print(f"Rapport écrit : {args.report_output}")


if __name__ == "__main__":
    main()