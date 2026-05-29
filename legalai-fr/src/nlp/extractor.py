from __future__ import annotations

import re

import spacy


nlp = spacy.load("fr_core_news_md")


LEGAL_KEYWORDS = [
    "contrat",
    "clause",
    "obligation",
    "responsabilité",
    "paiement",
    "dommages",
    "tribunal",
    "cour",
    "cassation",
    "appel",
]


def extract_entities(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Le texte fourni est vide.")

    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
        )

    legal_keywords_found = []

    lower_text = text.lower()

    for keyword in LEGAL_KEYWORDS:
        if keyword in lower_text:
            legal_keywords_found.append(keyword)

    amounts = re.findall(
        r"\b\d+(?:[\.,]\d+)?\s?(?:€|euros?)\b",
        text,
        flags=re.IGNORECASE,
    )

    articles = re.findall(
        r"article\s+[A-Za-z0-9\-\.\_]+",
        text,
        flags=re.IGNORECASE,
    )

    return {
        "entities": entities,
        "legal_keywords": sorted(list(set(legal_keywords_found))),
        "amounts": amounts,
        "articles": articles,
    }