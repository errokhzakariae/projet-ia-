from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.llm.openrouter_client import OpenRouterClient, OpenRouterError
from src.ml.predict import MLPPredictor
from src.nlp.extractor import extract_entities


PROJECT_ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="LegalAI-FR",
    layout="wide",
)


def load_openrouter_secrets() -> None:
    for key in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_APP_TITLE",
    ):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])


@st.cache_resource
def get_predictor() -> MLPPredictor:
    return MLPPredictor(
        model_path=PROJECT_ROOT / "artifacts" / "models" / "mlp_jurisdiction.pt",
        vectorizer_path=PROJECT_ROOT / "data" / "processed" / "tfidf_vectorizer.pkl",
        label_encoder_path=PROJECT_ROOT / "data" / "processed" / "label_encoder.pkl",
    )


def classify_text(text: str) -> None:
    try:
        result = get_predictor().predict(text)
    except Exception as exc:
        st.error(f"Classification indisponible : {exc}")
        return

    st.metric("Juridiction predite", result["predicted_label"])
    st.metric("Confiance", f"{result['confidence']:.2%}")
    st.json(result["probabilities"])


def extract_text(text: str) -> None:
    try:
        result = extract_entities(text)
    except Exception as exc:
        st.error(f"Extraction indisponible : {exc}")
        return

    st.json(result)


def summarize_text(text: str, max_tokens: int, temperature: float) -> None:
    load_openrouter_secrets()

    try:
        result = OpenRouterClient().summarize_legal_text(
            text=text,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except OpenRouterError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Resume indisponible : {exc}")
        return

    st.caption(f"Modele : {result['model']}")
    st.markdown(result["summary"])


st.title("LegalAI-FR")

text = st.text_area(
    "Texte juridique",
    height=260,
    placeholder="Colle ici un extrait de decision, contrat ou document juridique...",
)

left, middle, right = st.columns(3)

with left:
    classify_clicked = st.button("Classifier", use_container_width=True)

with middle:
    extract_clicked = st.button("Extraire", use_container_width=True)

with right:
    summarize_clicked = st.button("Resumer", use_container_width=True)

with st.sidebar:
    st.header("LLM")
    max_tokens = st.slider("Tokens max", min_value=100, max_value=2000, value=700, step=100)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

if not text.strip() and (classify_clicked or extract_clicked or summarize_clicked):
    st.warning("Ajoute un texte avant de lancer l'analyse.")
elif classify_clicked:
    classify_text(text)
elif extract_clicked:
    extract_text(text)
elif summarize_clicked:
    summarize_text(text, max_tokens=max_tokens, temperature=temperature)
