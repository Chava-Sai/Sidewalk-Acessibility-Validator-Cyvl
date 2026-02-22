# Sidewalk Accessibility Validator (Cyvl)

AI-powered sidewalk auditing platform with two modes:

1. **Map Audit**: rule-based ADA compliance analysis from street asset data.
2. **Image AI Advisor**: vision classifier (`Good/Fair/Poor`) + LLM recommendations.

The default LLM provider is **Groq (free-first)**. Users can switch to **Gemini** and provide their own key in the frontend.

## Target Repository

- GitHub: [https://github.com/Chava-Sai/Sidewalk-Acessibility-Validator-Cyvl](https://github.com/Chava-Sai/Sidewalk-Acessibility-Validator-Cyvl)

## Core Features

- Sidewalk condition classifier using PyTorch checkpoints.
- Sidewalk presence gate before condition advice.
- Provider-selectable LLM summary:
  - `groq` (default, recommended for free testing)
  - `gemini` (optional)
- User API key override from frontend (no key committed to repo).

## Minimal Project Structure (Important Files)

- `main.py` - FastAPI backend (map endpoints + prediction endpoint)
- `predict.py` - CLI inference helper
- `train.py`, `train_advanced.py` - training scripts
- `download_images.py`, `mask_images.py` - dataset preparation scripts
- `requirements.txt` - backend dependencies
- `frontend/` - Vite + React app
- `.env.example` - environment variable template

Data and large model files are intentionally excluded from Git.

## Data and Model Distribution Strategy

Do **not** commit heavy artifacts to Git history.

Publish heavy assets as **GitHub Release assets**:

- `dataset.zip` (raw dataset)
- `dataset_masked.zip` (masked dataset)
- optional model checkpoints (`*.pt`)
- optional metadata bundle (`pointcloud_coverage.json`, `sam.geojson`, `streetviewImages.geojson`)

This keeps clone/push fast and prevents repository bloat.

## Environment Variables

Copy `.env.example` and set values in your environment.

Key variables:

- `MODEL_PATH` - classifier checkpoint path
- `LLM_PROVIDER` - `groq` or `gemini` (default `groq`)
- `GROQ_API_KEY` - backend default Groq key (optional if user enters key in UI)
- `GEMINI_API_KEY` - backend default Gemini key (optional if user enters key in UI)
- `GROQ_MODEL`, `GROQ_FALLBACK_MODELS`
- `GEMINI_MODEL`, `GEMINI_FALLBACK_MODELS`
- `VITE_API_BASE_URL` (frontend)

Recommended free-first Groq setup for image analysis:

- `GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct`
- `GROQ_FALLBACK_MODELS=`

Note: use a Groq vision model for image analysis. `llama-4-scout` is the safest default.

## Local Run

### Backend

```bash
cd /Users/sai/Documents/sidewalk
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend

```bash
cd /Users/sai/Documents/sidewalk/frontend
npm install
echo 'VITE_API_BASE_URL=http://localhost:8001' > .env.local
npm run dev
```

## API Endpoints

- `GET /sidewalks`
- `GET /summary`
- `GET /violations`
- `POST /predict-sidewalk`

`POST /predict-sidewalk` form fields:

- `image` (required)
- `include_gemini` (`true`/`false`) - toggles recommendations
- `ai_provider` (`groq` or `gemini`)
- `llm_api_key` (optional user key override)
- `ai_model` (optional model override)
- `guidance_prompt` (optional)
- `enforce_sidewalk_check` (`true`/`false`)

## Frontend Behavior

In Image AI Advisor:

- Provider dropdown defaults to **Groq (Recommended Free)**.
- User can paste API key directly in UI.
- If API key is empty, backend env key is used (if configured).
- If no sidewalk is detected, classification summary is skipped.
- If Groq returns `403/1010`, your network is blocking the request; retry from another network/hotspot.

## Deployment (Permanent)

### Recommended Architecture

- **Frontend**: Vercel
- **Backend**: Render / Railway / Fly.io (persistent URL)

Why: this backend uses PyTorch + GeoPandas + model files, which is not ideal for Vercel serverless limits.

### Cost Guidance

- You can start on free tiers for testing.
- Free tiers may sleep or have quotas.
- For permanent stable production (always on), a paid backend plan is usually required.

### Vercel Setup

- Project root: `frontend`
- Env var on Vercel: `VITE_API_BASE_URL=https://<your-backend-domain>`
- Redeploy after env updates.

### Backend Setup (Render/Railway)

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Add env vars from `.env.example`
- Attach model checkpoint file and set `MODEL_PATH`

## Training Workflow

### Download raw images

```bash
python download_images.py --mode full --out-dir dataset
```

### Generate masked images

```bash
python mask_images.py
```

### Train advanced model

```bash
python train_advanced.py \
  --data-dir dataset_masked \
  --split-mode fair \
  --equal-train-per-class \
  --weighted-sampler \
  --arch convnext_tiny \
  --epochs 35 \
  --batch-size 16 \
  --num-workers 0 \
  --model-out sidewalk_classifier_fair.pt \
  --results-out training_results_fair.json
```

## Notes

- Keep secrets out of Git.
- Keep datasets/checkpoints out of Git history.
- Use Releases for heavy assets.
