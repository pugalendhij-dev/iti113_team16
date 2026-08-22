# ITI113 Team 16 — Singlish Marker Keyboard

A Singlish-aware smart-keyboard model that predicts the **sentence-final marker** of a
message: a discourse particle (`lah` / `leh` / `lor` ...), a mood-grouped emoji
(`emo_joy` / `emo_playful` / `emo_sad` / `emo_love`), or `none`. Trained on the NUS SMS
Corpus using **Option C** (slot-prediction) labelling.

**Team:** Joseph (s1602) — MLOps & Deployment · Ria — Model Development
**Region:** ap-southeast-1

## Repository layout

| Path | What it is |
|---|---|
| `notebooks/01A_...`, `01B_...` | MLflow app + S3 structure setup |
| `notebooks/03_singlish_pipeline_team16.ipynb` | SageMaker pipeline: preprocess → train → macro-F1 gate → registry; endpoint deploy |
| `notebooks/04_gradio_demo_team16.ipynb` | Live Gradio demo calling the serverless endpoint |
| `notebooks/ITI113_Team16_Combined_ProgressCheck.ipynb` | Progress-check notebook (EDA, labelling, baselines) |
| `src/singlish_labelling.py` | Shared labelling module (the train/serve contract) |
| `src/preprocess.py`, `train.py`, `inference.py` | Pipeline job scripts |
| `tests/` | Unit tests for the labelling module |
| `.github/workflows/ci.yml` | CI: compile scripts, validate notebooks, run tests on every push |

## MLOps summary

- **Pipeline:** `iti113-team16-singlish-pipeline` (preprocess → train → macro-F1 `ConditionStep` → register).
- **Model registry:** `iti113-team16-singlish-models` (versioned Model Package Group).
- **Endpoint:** `iti113-team16-singlish-keyboard` (serverless, real-time top-3 inference).
- **Experiment tracking:** MLflow app `iti113-26s1-team16-mlflow-app`.
- **CI (this repo):** unit tests guard the labelling logic so training and the endpoint never diverge.

## Running the tests locally

```bash
pip install -r requirements-ci.txt
pytest -q
```

## Demo

The Gradio demo (notebook 04) launches a public share link and returns the predicted
marker plus top-3 suggestions. See the report for the current demo URL and the backup
recording.
