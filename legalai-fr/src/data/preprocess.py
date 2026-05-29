from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nettoie le CSV brut et prépare le dataset final pour le MLP."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "jurisprudence.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "mlp_dataset.csv",
    )

    parser.add_argument(
        "--profile-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "preprocess_profile.json",
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=30,
        help="Longueur minimale du texte après nettoyage.",
    )

    return parser.parse_args()


def clean_text(text: str) -> str:
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"[^a-zàâäéèêëîïôöùûüç0-9\s\.,;:!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    label = label.replace(" ", "_")
    label = label.replace("-", "_")
    return label


def build_profile(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    return {
        "documents_before": int(len(df_before)),
        "documents_after": int(len(df_after)),
        "removed_documents": int(len(df_before) - len(df_after)),
        "labels_after": df_after["label"].value_counts().to_dict(),
        "missing_text_after": int(df_after["text_clean"].isna().sum()),
        "duplicates_after": int(df_after["text_clean"].duplicated().sum()),
        "text_length": {
            "min": int(df_after["text_length"].min()) if len(df_after) else 0,
            "max": int(df_after["text_length"].max()) if len(df_after) else 0,
            "mean": float(df_after["text_length"].mean()) if len(df_after) else 0.0,
            "median": float(df_after["text_length"].median()) if len(df_after) else 0.0,
        },
    }


def preprocess_dataset(input_path: Path, output_path: Path, profile_output: Path, min_chars: int) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    df = pd.read_csv(input_path)
    df_before = df.copy()

    required_columns = {"source_id", "label", "text"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing_columns}")

    df["label"] = df["label"].apply(normalize_label)
    df["text_clean"] = df["text"].fillna("").apply(clean_text)
    df["text_length"] = df["text_clean"].str.len()
    df["word_count"] = df["text_clean"].str.split().str.len()

    df = df[df["text_clean"].str.strip() != ""]
    df = df[df["text_length"] >= min_chars]

    df = df.drop_duplicates(subset=["text_clean"])
    df = df.reset_index(drop=True)

    final_df = df[
        [
            "source_id",
            "label",
            "text_clean",
            "text_length",
            "word_count",
        ]
    ].rename(columns={"text_clean": "text"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8")

    profile = build_profile(df_before, df)
    profile["input_path"] = str(input_path)
    profile["output_path"] = str(output_path)
    profile["min_chars"] = min_chars

    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profile_output.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Dataset nettoyé écrit : {output_path}")
    print(f"Profil preprocessing écrit : {profile_output}")
    print(f"Documents avant : {len(df_before)}")
    print(f"Documents après : {len(final_df)}")


def main() -> None:
    args = parse_args()

    preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
        profile_output=args.profile_output,
        min_chars=args.min_chars,
    )


if __name__ == "__main__":
    main()