# AI Cloud Inference Service

A cloud-native **sentiment analysis API**. A FastAPI web service runs a PyTorch
**DistilBERT** model from Hugging Face
([`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)).
It is packaged as a Docker container, deployed on **Google Cloud Run**, and saves
every prediction to **Google Cloud Firestore**.

**Live URL:** https://ai-inference-service-404745519573.us-central1.run.app
**Interactive API docs (Swagger UI):** https://ai-inference-service-404745519573.us-central1.run.app/docs

---

## Architecture

```
[ Client: Browser /docs, Postman, PowerShell ]
              │  HTTPS POST /predict (JSON)
              ▼
[ Google Cloud Run: serverless container, autoscaling 0→N, managed HTTPS ]
  └─ Docker image (python:3.11-slim) built by Cloud Build, stored in Artifact Registry
     └─ Uvicorn → FastAPI → Pydantic validation → PyTorch DistilBERT model
              │  writes one record per prediction
              ▼
[ Google Cloud Firestore (NoSQL): "predictions" collection ]
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Returns POSITIVE / NEGATIVE sentiment with a confidence score |
| GET | `/docs` | Interactive Swagger UI |

**Example request**

```json
{ "text_input": "Deploying AI to the cloud is awesome", "max_length": 30 }
```

**Example response**

```json
{
  "status": "success",
  "input_text": "Deploying AI to the cloud is awesome",
  "prediction": "POSITIVE (confidence: 99.98%)"
}
```

- `text_input`: the text to classify (required, cannot be blank)
- `max_length`: the most tokens the model reads (8–512, default 50)

Each prediction is stored in Firestore with `input_text`, `label`, `confidence`,
`model` and `created_at`.

## The AI model

- **Model:** DistilBERT base uncased, fine-tuned on the Stanford Sentiment Treebank (SST-2)
- **Architecture:** 6-layer transformer encoder, 768 hidden size, 12 attention heads, about 66M parameters.
  It is a distilled version of BERT: about 40% smaller and 60% faster, while keeping about 97% of BERT's language-understanding performance.
- **Framework:** PyTorch (CPU build), loaded with the Hugging Face `transformers` `pipeline`
- The model loads once when the container starts, and its weights are built into the Docker image for faster cold starts.

## Project files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: endpoints, model inference, Firestore logging |
| `requirements.txt` | Python dependencies (CPU-only PyTorch) |
| `Dockerfile` | Container image definition; downloads the model while the image is built |
| `.dockerignore` / `.gcloudignore` | Keep non-code files out of the image and the Cloud upload |

## Run locally (Python)

```bash
pip install -r requirements.txt
uvicorn main:app --port 8080
```

Open http://127.0.0.1:8080/docs. Firestore logging is **off** locally unless you set `ENABLE_FIRESTORE=true`.

## Run with Docker

```bash
docker build -t ai-inference-service .
docker run -p 8080:8080 ai-inference-service
```

Open http://127.0.0.1:8080/docs

PowerShell test request:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/predict" -ContentType "application/json" -Body '{"text_input": "Hello Cloud Computing", "max_length": 20}'
```

## Deploy to Google Cloud Run

Requires the Google Cloud CLI, a project with billing turned on, these APIs enabled
(Cloud Run, Cloud Build, Artifact Registry, Firestore), a Firestore database in
`us-central1`, and the `roles/datastore.user` role for the Cloud Run service account.

```bash
gcloud run deploy ai-inference-service \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --set-env-vars ENABLE_FIRESTORE=true
```

`--memory 2Gi` is needed because the PyTorch model does not fit in Cloud Run's default 512 MiB.

## Tech stack

Python 3.11 · FastAPI · Pydantic · Uvicorn · PyTorch · Hugging Face Transformers · Docker · Google Cloud Run · Cloud Build · Artifact Registry · Firestore
