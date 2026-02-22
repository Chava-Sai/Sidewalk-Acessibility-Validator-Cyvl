# Sidewalk Accessibility Validator (CYVL)

CYVL is a two-part accessibility project:

1. **Map Audit**: rule-based ADA audit on mapped sidewalk assets.
2. **Image AI Advisor**: predicts `Good / Fair / Poor` and generates improvement guidance.

## Public-Ready Setup (Important)

This repo is prepared so **anyone can run it**.

- Model checkpoints are **not** committed to Git history.
- Backend can auto-download the public checkpoint from GitHub Releases using `MODEL_URL`.
- Users use **their own API key** (no private key is stored in code).

Default model URL:

- `https://github.com/Chava-Sai/Sidewalk-Acessibility-Validator-Cyvl/releases/latest/download/sidewalk_classifier_fair.pt`

## Llama First (Free Tier)

The default AI path is **Llama vision model via Groq**.

- Default model: `meta-llama/llama-4-scout-17b-16e-instruct`
- Provider value in API: `groq` (this means Llama via Groq)

Get a key and docs:

- Groq API keys: <https://console.groq.com/keys>
- GroqCloud (start free): <https://groq.com/groqcloud/>
- Groq rate limits: <https://console.groq.com/docs/rate-limits>
- Groq vision docs: <https://console.groq.com/docs/vision>
- Groq models docs: <https://console.groq.com/docs/models>

Gemini is optional and can be selected from the UI.

## Project Structure

- `main.py` - FastAPI backend (map + prediction + AI summary)
- `frontend/` - React + Vite frontend
- `train.py`, `train_advanced.py` - training scripts
- `download_images.py`, `mask_images.py` - dataset preparation
- `requirements.txt` - backend dependencies
- `.env.example` - runtime configuration template
- `images/` - product screenshots

## Quick Start (Local)

### 1. Backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# optional but recommended
export GROQ_API_KEY="your_groq_key"

uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Frontend

```bash
cd frontend
npm install
echo 'VITE_API_BASE_URL=http://localhost:8001' > .env.local
npm run dev
```

Open: `http://localhost:3000` (or the next free port shown by Vite).

## Environment Variables

Copy from `.env.example`:

- `MODEL_PATH` - local `.pt` file path
- `MODEL_URL` - release asset URL used when `MODEL_PATH` is missing
- `LLM_PROVIDER` - `groq` (Llama via Groq) or `gemini`
- `GROQ_API_KEY` - key for Llama via Groq
- `GROQ_MODEL` - vision model name
- `GEMINI_API_KEY`, `GEMINI_MODEL` - optional Gemini path

## Data + Model Release Strategy

Keep heavy files out of Git history. Upload them as Release assets:

- `dataset.zip`
- `dataset_masked.zip`
- `sidewalk_classifier_fair.pt` (public checkpoint)

This keeps cloning fast and still lets everyone run the app.

## Deployment (Permanent)

Recommended architecture:

- **Frontend**: Vercel
- **Backend**: Render / Railway / Fly.io

Why: backend depends on PyTorch + GeoPandas + checkpoint file and is better on a persistent container than serverless limits.

## Product Screenshots

![Image Advisor Upload](images/advisor-upload.jpeg)
![Llama Settings](images/llama-settings.jpeg)
![Local Run Commands](images/local-run-commands.jpeg)

