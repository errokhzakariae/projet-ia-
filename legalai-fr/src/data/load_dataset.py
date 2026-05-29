from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import get_dataset_split_names, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET = "antoinejeannot/jurisprudence"
DEFAULT_LABELS = ("cour_de_cassation",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Charge le dataset jurisprudence et construit le CSV brut pour le MLP."
    )

    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))

    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "jurisprudence.csv",
    )

    parser.add_argument(
        "--profile-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "raw_profile.json",
    )

    return parser.parse_args()


def take_records(records: Iterable[dict], limit: int | None) -> Iterable[dict]:
    return records if limit is None else islice(records, limit)


def get_available_splits(dataset_name: str) -> list[str]:
    return get_dataset_split_names(dataset_name)


def validate_labels(labels: list[str], available_splits: list[str]) -> None:
    missing_labels = [label for label in labels if label not in available_splits]

    if missing_labels:
        raise ValueError(
            f"Labels introuvables : {missing_labels}\n"
            f"Splits disponibles : {available_splits}"
        )


def load_split_records(
    dataset_name: str,
    split_name: str,
    text_column: str,
    streaming: bool,
    limit: int | None,
) -> list[dict]:
    print(f"Chargement du split : {split_name}")

    dataset = load_dataset(
        dataset_name,
        split=split_name,
        streaming=streaming,
    )

    rows: list[dict] = []

    for index, row in enumerate(take_records(dataset, limit)):
        text = row.get(text_column)

        if text is None or not str(text).strip():
            continue

        rows.append(
            {
                "source_id": row.get("id")
                or row.get("decision_id")
                or row.get("uuid")
                or f"{split_name}-{index}",
                "label": split_name,
                "text": str(text),
            }
        )

    print(f"{len(rows)} documents chargés pour {split_name}")
    return rows


def build_profile(df: pd.DataFrame) -> dict:
    text = df["text"].fillna("").astype(str)
    lengths = text.str.len()

    return {
        "total_documents": int(len(df)),
        "documents_by_label": {
            key: int(value) for key, value in Counter(df["label"]).items()
        },
        "missing_or_empty_texts": int(text.str.strip().eq("").sum()),
        "duplicate_texts": int(text.duplicated().sum()),
        "length_characters": {
            "min": int(lengths.min()) if len(lengths) else 0,
            "max": int(lengths.max()) if len(lengths) else 0,
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "median": float(lengths.median()) if len(lengths) else 0.0,
        },
    }


def save_outputs(
    df: pd.DataFrame,
    output_path: Path,
    profile: dict,
    profile_output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8")

    profile_output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nCSV brut écrit : {output_path}")
    print(f"Profil data understanding écrit : {profile_output_path}")


def main() -> None:
    args = parse_args()

    limit = None if args.full_data else args.max_per_class

    # IMPORTANT :
    # Par défaut on force le streaming pour éviter de télécharger plusieurs Go.
    # Utilise --no-streaming uniquement si tu veux vraiment télécharger les fichiers complets.
    streaming = not args.no_streaming

    available_splits = get_available_splits(args.dataset_name)

    print("Dataset :", args.dataset_name)
    print("Splits disponibles :", available_splits)
    print("Labels sélectionnés :", args.labels)
    print("Streaming :", streaming)
    print("Max par classe :", limit)

    validate_labels(args.labels, available_splits)

    all_rows: list[dict] = []

    for label in args.labels:
        records = load_split_records(
            dataset_name=args.dataset_name,
            split_name=label,
            text_column=args.text_column,
            streaming=streaming,
            limit=limit,
        )
        all_rows.extend(records)

    if not all_rows:
        raise RuntimeError("Aucun document chargé.")

    df = pd.DataFrame(all_rows)

    profile = build_profile(df)
    profile["dataset_name"] = args.dataset_name
    profile["text_column"] = args.text_column
    profile["streaming"] = streaming
    profile["max_per_class"] = limit
    profile["selected_labels"] = args.labels
    profile["available_splits"] = available_splits

    save_outputs(
        df=df,
        output_path=args.output,
        profile=profile,
        profile_output_path=args.profile_output,
    )


if __name__ == "__main__":
    main()