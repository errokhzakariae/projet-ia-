# CRISP-DM Phase 2 - Data Understanding

## Choix valides

| Element | Decision |
| --- | --- |
| Dataset MLP | `antoinejeannot/jurisprudence` |
| Dataset RAG | `harvard-lil/cold-french-law` |
| Tache MLP | Classification de juridiction |
| Labels | `cour_de_cassation`, `cour_d_appel`, `tribunal_judiciaire` |
| Features | TF-IDF |
| Modele | MLP PyTorch |

## Volume source attendu

D'apres la fiche du dataset `antoinejeannot/jurisprudence`, les trois juridictions
sont publiees sous forme de splits Hugging Face et de fichiers parquet/JSONL.
Le pipeline recalcule localement les volumes exacts dans `data/interim/raw_profile.json`.

| Classe | Role dans la classification |
| --- | --- |
| `cour_de_cassation` | Label MLP |
| `cour_d_appel` | Label MLP |
| `tribunal_judiciaire` | Label MLP |

## Controles qualite

Les scripts mesurent et exportent:

- nombre total de documents;
- nombre de documents par classe;
- textes vides ou manquants;
- doublons;
- textes trop courts ou trop longs;
- longueurs min, max, moyenne et mediane;
- distribution des labels avant et apres equilibrage.

## Pipeline preprocessing

1. Charger dataset.
2. Selectionner `source_id`, `text`, `label`.
3. Supprimer documents vides.
4. Nettoyer le texte.
5. Normaliser labels.
6. Filtrer textes trop courts ou trop longs.
7. Equilibrer classes par sous-echantillonnage.
8. Split train/val/test en 70/15/15.
9. Entrainer TF-IDF sur le train uniquement.
10. Exporter CSV, vectorizer et encoder pour le MLP.

## Commandes reproductibles

```powershell
python -m src.data.load_dataset --max-per-class 5000
python -m src.data.preprocess
python -m src.data.split_data
```

## Livrables

| Fichier | Description |
| --- | --- |
| `data/processed/mlp_dataset.csv` | Dataset nettoye et encode |
| `data/processed/train.csv` | Split train |
| `data/processed/val.csv` | Split validation |
| `data/processed/test.csv` | Split test |
| `data/processed/tfidf_vectorizer.pkl` | Vectorizer TF-IDF entraine sur train |
| `data/processed/label_encoder.pkl` | Encodeur des labels |

