from src.nlp.extractor import extract_entities


def extract_information(text: str) -> dict:
    return extract_entities(text)