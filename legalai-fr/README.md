# LegalAI-FR

Projet IA juridique francophone.

Cette base couvre la phase CRISP-DM 2, `Data Understanding`, et prépare les
artefacts nécessaires au MLP de classification de juridiction.

## Datasets retenus

| Usage | Dataset |
| --- | --- |
| IA classique MLP | `antoinejeannot/jurisprudence` |
| RAG juridique | `harvard-lil/cold-french-law` |
| Contrats annotés bonus | CUAD Kaggle |
| Open-source NLP | CamemBERT |

## Tache MLP

Classification par juridiction:

- `cour_de_cassation`
- `cour_d_appel`
- `tribunal_judiciaire`

Features:

- `TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3, max_df=0.9)`
- modele: MLP PyTorch

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Pipeline phase 2

Les commandes ci-dessous produisent les livrables attendus dans `data/processed`.
Par defaut, le chargement se fait en streaming avec un echantillon par classe
pour eviter de telecharger plusieurs Go pendant les iterations.

```powershell
python -m src.data.load_dataset --max-per-class 5000
python -m src.data.preprocess
python -m src.data.split_data
```

Pour travailler sur tout le dataset:

```powershell
python -m src.data.load_dataset --full-data --no-streaming
python -m src.data.preprocess --balance downsample
python -m src.data.split_data
```

## Livrables phase 2

- `data/processed/mlp_dataset.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/tfidf_vectorizer.pkl`
- `data/processed/label_encoder.pkl`
- `data/interim/raw_profile.json`
- `data/interim/preprocessing_report.json`
- `data/interim/split_report.json`

## Entrainement MLP

```powershell
python -m src.ml.train_mlp --epochs 10
python -m src.ml.evaluate
```

## Backend FastAPI

```powershell
uvicorn backend.app.main:app --reload
```

Swagger est disponible sur `http://127.0.0.1:8000/docs`.

Endpoints disponibles:

- `POST /ai/classify`
- `POST /ai/extract`
- `POST /ai/summarize`

## Interface Streamlit

En local:

```powershell
streamlit run streamlit_app.py
```

Sur Streamlit Cloud, selectionne ce fichier comme point d'entree:

```txt
streamlit_app.py
```

Dans les secrets Streamlit Cloud, ajoute au minimum:

```toml
OPENROUTER_API_KEY = "sk-or-v1-ta-cle"
OPENROUTER_MODEL = "mistralai/mistral-small-3.2-24b-instruct"
```

## API LLM avec OpenRouter

Crée un fichier `.env` à partir de `.env.example`, puis renseigne ta clé:

```powershell
Copy-Item .env.example .env
```

Variables principales:

- `OPENROUTER_API_KEY`: clé API OpenRouter obligatoire.
- `OPENROUTER_MODEL`: modèle utilisé pour le résumé, par défaut `mistralai/mistral-small-3.2-24b-instruct`.

Exemple de requête:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/ai/summarize `
  -ContentType "application/json" `
  -Body '{"text":"La Cour de cassation rejette le pourvoi forme contre l arret attaque. Le litige porte sur la responsabilite contractuelle.","max_tokens":400,"temperature":0.2}'
```

## Docker

```powershell
docker build -t legalai-fr .
docker run --rm -v ${PWD}\data:/app/data legalai-fr
```
