# Sidewalk Accessibility Validator (CYVL)

## Live Demo

**Final project demo (frontend):**  
<https://sidewalk-acessibility-validator-cyv.vercel.app/>

No setup required for basic use.

## What This Project Does

CYVL has two modules:

1. **Map Audit**: rule-based ADA audit on mapped sidewalk assets.
2. **Image AI Advisor**: classifies uploaded sidewalk images as `Good / Fair / Poor` and provides improvement guidance.

## Production Deployment

- **Frontend (public demo)**: Vercel  
  <https://sidewalk-acessibility-validator-cyv.vercel.app/>
- **Backend API**: Hugging Face Space (Docker, free tier)  
  <https://srinivasasai-sidewalk-backend-hf.hf.space>
- **Backend health check**:  
  <https://srinivasasai-sidewalk-backend-hf.hf.space/summary>

## API Keys (For Full AI Guidance)

The app works without an API key for map + classifier output.

For full AI recommendations, provide a key in the UI:

- Groq keys: <https://console.groq.com/keys>
- Gemini keys: <https://aistudio.google.com/api-keys>

Default free model path is Llama via Groq.

## Local Run (Optional)

### Backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend

```bash
cd frontend
npm install
echo 'VITE_API_BASE_URL=http://localhost:8001' > .env.local
npm run dev
```

## Environment Variables

Main backend variables:

- `MODEL_PATH` - local `.pt` path (recommended on cloud: `/tmp/sidewalk_classifier_fair.pt`)
- `MODEL_URL` - release URL to auto-download checkpoint if file is missing
- `LLM_PROVIDER` - `groq` or `gemini`
- `GROQ_API_KEY` / `GEMINI_API_KEY` - optional provider keys

Main frontend variable:

- `VITE_API_BASE_URL` - backend URL used by the UI

## Repo Structure

- `main.py` - FastAPI backend
- `frontend/` - React + Vite frontend
- `train.py`, `train_advanced.py` - training scripts
- `download_images.py`, `mask_images.py` - data prep scripts
- `.env.example` - runtime configuration template
- `sidewalk_results_cache.json` - precomputed map audit cache

## Model and Data Release Strategy

Heavy files are kept out of Git history and published as release assets:

- `dataset.zip`
- `dataset_masked.zip`
- `sidewalk_classifier_fair.pt`

Default model download URL:

- <https://github.com/Chava-Sai/Sidewalk-Acessibility-Validator-Cyvl/releases/latest/download/sidewalk_classifier_fair.pt>

## Screenshots

![Image Advisor Upload](images/advisor-upload.jpeg)
![Llama Settings](images/llama-settings.jpeg)
![Local Run Commands](images/local-run-commands.jpeg)
