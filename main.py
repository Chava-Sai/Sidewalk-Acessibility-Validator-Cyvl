import base64
import json
import os
import re
import threading
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

IMG_SIZE_DEFAULT = 224
LLM_PROVIDER_DEFAULT = os.getenv("LLM_PROVIDER", "groq").strip().lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview").strip()
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-flash-latest,gemini-2.5-flash-lite").split(",")
    if model.strip()
]
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct").strip()
GROQ_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GROQ_FALLBACK_MODELS",
        "",
    ).split(",")
    if model.strip()
]
DEFAULT_MODEL_URL = (
    "https://github.com/Chava-Sai/Sidewalk-Acessibility-Validator-Cyvl/"
    "releases/latest/download/sidewalk_classifier_fair.pt"
)
MODEL_URL = os.getenv("MODEL_URL", DEFAULT_MODEL_URL).strip()
SIDEWALK_CACHE_PATH = Path(os.getenv("SIDEWALK_CACHE_PATH", "sidewalk_results_cache.json").strip())
HTTP_USER_AGENT = os.getenv(
    "HTTP_USER_AGENT",
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
).strip()
JSON_REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": HTTP_USER_AGENT or "sidewalk-cyvl/1.0",
}


def resolve_model_path():
    env_model = os.getenv("MODEL_PATH", "").strip()
    if env_model:
        return Path(env_model)

    for candidate in ["sidewalk_classifier_fair.pt", "sidewalk_classifier.pt"]:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    return Path("sidewalk_classifier.pt")


MODEL_PATH = resolve_model_path()


def ensure_model_checkpoint(target_path: Path):
    if target_path.exists():
        return
    if not MODEL_URL:
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Model not found at {target_path}. Attempting auto-download from release asset...")
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": HTTP_USER_AGENT or "sidewalk-cyvl/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = response.read()
        if not body:
            raise RuntimeError("Downloaded model file was empty.")
        target_path.write_bytes(body)
        print(f"Downloaded model checkpoint to {target_path}")
    except Exception as exc:
        print(f"Model auto-download failed: {exc}")


def make_classifier_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

if torch.cuda.is_available():
    CLASSIFIER_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    CLASSIFIER_DEVICE = "mps"
else:
    CLASSIFIER_DEVICE = "cpu"

CLASSIFIER_MODEL = None
CLASSIFIER_CLASSES = []
CLASSIFIER_ARCH = ""
CLASSIFIER_IMG_SIZE = IMG_SIZE_DEFAULT
CLASSIFIER_TF = make_classifier_transform(CLASSIFIER_IMG_SIZE)
CLASSIFIER_LOAD_ERROR = ""
CLASSIFIER_LOCK = threading.Lock()


def load_classifier():
    ensure_model_checkpoint(MODEL_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            "Set MODEL_PATH to a local .pt file or set MODEL_URL to a downloadable checkpoint."
        )
    checkpoint = torch.load(MODEL_PATH, map_location=CLASSIFIER_DEVICE)
    classes = checkpoint["classes"]
    arch = checkpoint.get("arch", "efficientnet_b2")
    img_size = int(checkpoint.get("img_size", IMG_SIZE_DEFAULT))

    if arch == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, len(classes)),
        )
    elif arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, len(classes)),
        )
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {arch}")

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(CLASSIFIER_DEVICE)
    model.eval()
    return model, classes, arch, img_size


def ensure_classifier_loaded():
    global CLASSIFIER_MODEL, CLASSIFIER_CLASSES, CLASSIFIER_ARCH, CLASSIFIER_IMG_SIZE, CLASSIFIER_TF, CLASSIFIER_LOAD_ERROR

    if CLASSIFIER_MODEL is not None:
        return

    with CLASSIFIER_LOCK:
        if CLASSIFIER_MODEL is not None:
            return
        try:
            CLASSIFIER_MODEL, CLASSIFIER_CLASSES, CLASSIFIER_ARCH, CLASSIFIER_IMG_SIZE = load_classifier()
            CLASSIFIER_TF = make_classifier_transform(CLASSIFIER_IMG_SIZE)
            CLASSIFIER_LOAD_ERROR = ""
            print(
                f"Loaded model: {MODEL_PATH} | Arch: {CLASSIFIER_ARCH} | "
                f"Classes: {CLASSIFIER_CLASSES} | ImgSize: {CLASSIFIER_IMG_SIZE} | Device: {CLASSIFIER_DEVICE}"
            )
        except Exception as exc:  # pragma: no cover - startup environment issue
            CLASSIFIER_MODEL = None
            CLASSIFIER_CLASSES = []
            CLASSIFIER_ARCH = ""
            CLASSIFIER_IMG_SIZE = IMG_SIZE_DEFAULT
            CLASSIFIER_TF = make_classifier_transform(CLASSIFIER_IMG_SIZE)
            CLASSIFIER_LOAD_ERROR = str(exc)
            print(f"Classifier unavailable: {CLASSIFIER_LOAD_ERROR}")


def predict_sidewalk_quality(image_bytes: bytes):
    ensure_classifier_loaded()
    if CLASSIFIER_MODEL is None:
        raise HTTPException(status_code=503, detail=f"Classifier unavailable: {CLASSIFIER_LOAD_ERROR}")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    tensor = CLASSIFIER_TF(image).unsqueeze(0).to(CLASSIFIER_DEVICE)
    with torch.no_grad():
        logits = CLASSIFIER_MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    probabilities = {CLASSIFIER_CLASSES[i]: round(float(prob), 4) for i, prob in enumerate(probs)}
    predicted_idx = int(max(range(len(probs)), key=lambda idx: probs[idx]))
    predicted_class = CLASSIFIER_CLASSES[predicted_idx]
    confidence = round(float(probs[predicted_idx]), 4)

    return predicted_class, confidence, probabilities


def build_advisor_prompt(predicted_class: str, confidence: float, probabilities: dict, custom_prompt: str):
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    prob_text = ", ".join([f"{label}={score * 100:.1f}%" for label, score in sorted_probs])
    confidence_pct = confidence * 100.0

    base_prompt = (
        "You are an ADA sidewalk accessibility reviewer.\n"
        "Use ONLY the uploaded image for visual evidence.\n"
        "Classifier output is authoritative and must not be changed.\n"
        f"Classifier rating: {predicted_class}\n"
        f"Classifier confidence: {confidence_pct:.1f}%\n"
        f"Class probabilities: {prob_text}\n\n"
        "Return ONLY valid JSON (no markdown, no extra text) with this exact schema:\n"
        '{"rating":"Good|Fair|Poor","confidence_pct":0.0,"why":"...","actions":["...","...","..."],"expected_result":"..."}\n\n'
        "Rules:\n"
        "- rating MUST equal the classifier rating exactly.\n"
        "- why must be 2-3 sentences based only on visible evidence.\n"
        "- if confidence < 55.0 then why must start with 'Uncertain:'.\n"
        "- actions must contain exactly 3 specific repair/maintenance actions.\n"
        "- If rating is Good, actions should be maintenance actions to keep it Good.\n"
        "- expected_result is one concise sentence."
    )

    if custom_prompt.strip():
        base_prompt += f"\nAdditional user request: {custom_prompt.strip()}"

    return base_prompt


def extract_json_object(text: str):
    if not text:
        return None
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def normalize_llm_provider(provider: str):
    value = (provider or "").strip().lower()
    if value in {"groq", "gemini"}:
        return value
    return LLM_PROVIDER_DEFAULT if LLM_PROVIDER_DEFAULT in {"groq", "gemini"} else "groq"


def resolve_llm_api_key(provider: str, api_key_override: str):
    override_key = (api_key_override or "").strip()
    if override_key:
        return override_key

    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY", "").strip()
    if provider == "groq":
        return os.getenv("GROQ_API_KEY", "").strip()
    return ""


def call_gemini_text(
    image_bytes: bytes,
    image_mime_type: str,
    prompt: str,
    api_key_override: str = "",
    model_override: str = "",
    max_output_tokens: int = 320,
    temperature: float = 0.3,
):
    api_key = resolve_llm_api_key("gemini", api_key_override)
    if not api_key:
        return None, "Gemini key missing. Provide GEMINI_API_KEY or enter API key in UI.", None

    primary_model = model_override.strip() or GEMINI_MODEL
    model_candidates = [primary_model] + GEMINI_FALLBACK_MODELS
    tried_models = []
    last_model_error = ""

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": image_mime_type or "image/jpeg",
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    }
                },
            ]
        }],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
    }

    for model_name in model_candidates:
        if not model_name or model_name in tried_models:
            continue
        tried_models.append(model_name)

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={api_key}"
        )
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=JSON_REQUEST_HEADERS,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="ignore")
            if exc.code in {400, 404}:
                last_model_error = error_text[:220]
                continue
            return None, f"Gemini API error ({exc.code}): {error_text[:220]}", model_name
        except Exception as exc:  # pragma: no cover - network/remote failures
            return None, f"Gemini request failed: {exc}", model_name

        candidates = body.get("candidates") or []
        if not candidates:
            return None, "Gemini returned no response candidates.", model_name

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text_chunks = [part.get("text", "").strip() for part in parts if part.get("text")]
        if not text_chunks:
            return None, "Gemini returned an empty response.", model_name

        return "\n".join(text_chunks), None, model_name

    return (
        None,
        f"No available Gemini model for this key/project. Tried: {', '.join(tried_models)}. Last model error: {last_model_error}",
        None,
    )


def call_groq_text(
    image_bytes: bytes,
    image_mime_type: str,
    prompt: str,
    api_key_override: str = "",
    model_override: str = "",
    max_output_tokens: int = 320,
    temperature: float = 0.3,
):
    api_key = resolve_llm_api_key("groq", api_key_override)
    if not api_key:
        return None, "Llama API key missing. Provide GROQ_API_KEY or enter API key in UI.", None

    primary_model = model_override.strip() or GROQ_MODEL
    model_candidates = [primary_model] + GROQ_FALLBACK_MODELS
    tried_models = []
    last_model_error = ""
    image_mime = image_mime_type or "image/jpeg"
    image_data_uri = f"data:{image_mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    for model_name in model_candidates:
        if not model_name or model_name in tried_models:
            continue
        tried_models.append(model_name)

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                }
            ],
            "temperature": temperature,
            "max_completion_tokens": max_output_tokens,
        }

        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                **JSON_REQUEST_HEADERS,
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="ignore")
            lower_error = error_text.lower()
            if exc.code == 403 and ("1010" in lower_error or "access denied" in lower_error):
                return (
                    None,
                    "Llama (via Groq) request was blocked (403/1010). This is usually network or security filtering. "
                    "Try another network and verify the API key works with `curl https://api.groq.com/openai/v1/models`.",
                    model_name,
                )
            if exc.code in {400, 404} and (
                "model" in lower_error
                or "not found" in lower_error
                or "does not exist" in lower_error
                or "unsupported" in lower_error
            ):
                last_model_error = error_text[:220]
                continue
            return None, f"Llama/Groq API error ({exc.code}): {error_text[:220]}", model_name
        except Exception as exc:  # pragma: no cover - network/remote failures
            return None, f"Groq request failed: {exc}", model_name

        choices = body.get("choices") or []
        if not choices:
            return None, "Groq returned no response choices.", model_name

        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            return None, "Groq returned an empty response.", model_name

        return content, None, model_name

    return (
        None,
        f"No available Groq model for this key/project. Tried: {', '.join(tried_models)}. Last model error: {last_model_error}",
        None,
    )


def call_llm_text(
    provider: str,
    image_bytes: bytes,
    image_mime_type: str,
    prompt: str,
    api_key_override: str = "",
    model_override: str = "",
    max_output_tokens: int = 320,
    temperature: float = 0.3,
):
    normalized = normalize_llm_provider(provider)
    if normalized == "gemini":
        return call_gemini_text(
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            prompt=prompt,
            api_key_override=api_key_override,
            model_override=model_override,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    if normalized == "groq":
        return call_groq_text(
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            prompt=prompt,
            api_key_override=api_key_override,
            model_override=model_override,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    return None, f"Unsupported AI provider: {provider}", None


def validate_summary_payload(payload: dict, predicted_class: str, confidence: float):
    if not isinstance(payload, dict):
        return None

    rating = str(payload.get("rating", "")).strip()
    if rating not in {"Good", "Fair", "Poor"}:
        return None
    if rating != predicted_class:
        rating = predicted_class

    confidence_pct = payload.get("confidence_pct")
    try:
        confidence_pct = float(confidence_pct)
    except (TypeError, ValueError):
        confidence_pct = confidence * 100.0

    why = str(payload.get("why", "")).strip()
    if not why:
        return None

    actions = payload.get("actions")
    if not isinstance(actions, list):
        return None
    actions = [str(item).strip() for item in actions if str(item).strip()]
    if len(actions) < 3:
        return None
    actions = actions[:3]

    expected_result = str(payload.get("expected_result", "")).strip()
    if not expected_result:
        return None

    return {
        "rating": rating,
        "confidence_pct": confidence_pct,
        "why": why,
        "actions": actions,
        "expected_result": expected_result,
    }


def format_summary_payload(payload: dict):
    confidence_pct = payload.get("confidence_pct")
    if isinstance(confidence_pct, (int, float)):
        confidence_text = f"{float(confidence_pct):.1f}%"
    else:
        confidence_text = "n/a"

    return "\n".join(
        [
            f"Rating: {payload['rating']} ({confidence_text})",
            f"Why this rating: {payload['why']}",
            "How to improve to Good:",
            f"1. {payload['actions'][0]}",
            f"2. {payload['actions'][1]}",
            f"3. {payload['actions'][2]}",
            f"Expected result: {payload['expected_result']}",
        ]
    )


def fallback_summary_text(predicted_class: str, confidence: float):
    confidence_pct = confidence * 100.0
    uncertain_prefix = "Uncertain: " if confidence_pct < 55.0 else ""

    if predicted_class == "Good":
        why = (
            f"{uncertain_prefix}The surface appears mostly continuous and walkable with limited visible obstructions. "
            "No major trip hazards are clearly dominant in this frame."
        )
        actions = [
            "Schedule routine cleaning and debris removal along the walking path.",
            "Trim vegetation and maintain edge clearances to preserve sidewalk width.",
            "Perform periodic spot sealing/patching where early wear appears.",
        ]
        expected = "Sidewalk quality is maintained at Good with reduced risk of future deterioration."
    elif predicted_class == "Fair":
        why = (
            f"{uncertain_prefix}The sidewalk appears usable but shows moderate surface wear and localized defects that can worsen over time. "
            "These issues can impact comfort and accessibility if not corrected."
        )
        actions = [
            "Patch visible cracks and minor depressions in the main walking line.",
            "Grind or level small vertical offsets to reduce trip risk.",
            "Repaint/restore edge and crossing cues where faded or unclear.",
        ]
        expected = "Defects are reduced and accessibility improves toward a stable Good condition."
    else:
        why = (
            f"{uncertain_prefix}The image indicates significant deterioration and/or hazards affecting safe pedestrian movement. "
            "Current conditions likely present frequent accessibility barriers."
        )
        actions = [
            "Repair or replace severely broken pavement sections in the travel path.",
            "Correct major height differentials and unstable edges near curb transitions.",
            "Reconstruct affected segments to restore a smooth, continuous ADA-friendly surface.",
        ]
        expected = "Major hazards are removed and the corridor can be restored to Good usability."

    return "\n".join(
        [
            f"Rating: {predicted_class} ({confidence_pct:.1f}%)",
            f"Why this rating: {why}",
            "How to improve to Good:",
            f"1. {actions[0]}",
            f"2. {actions[1]}",
            f"3. {actions[2]}",
            f"Expected result: {expected}",
        ]
    )


def build_sidewalk_presence_prompt():
    return (
        "Determine whether this image contains a real, visible pedestrian sidewalk or footpath.\n"
        "Treat logos, icons, screenshots, drawings, diagrams, blank images, and synthetic graphics as NO sidewalk.\n"
        "Return ONLY JSON with this schema:\n"
        '{"has_sidewalk": true|false, "confidence": 0.0-1.0, "reason": "short reason"}'
    )


def detect_sidewalk_presence(
    image_bytes: bytes,
    image_mime_type: str,
    ai_provider: str,
    llm_api_key: str = "",
    ai_model: str = "",
):
    prompt = build_sidewalk_presence_prompt()
    text, error, model = call_llm_text(
        provider=ai_provider,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        prompt=prompt,
        api_key_override=llm_api_key,
        model_override=ai_model,
        max_output_tokens=120,
        temperature=0.0,
    )
    if error:
        return None, error, model

    parsed = extract_json_object(text or "")
    if not parsed or "has_sidewalk" not in parsed:
        return None, f"Could not parse sidewalk-detection output: {text}", model

    has_sidewalk = bool(parsed.get("has_sidewalk"))
    confidence = parsed.get("confidence")
    reason = str(parsed.get("reason", "")).strip()

    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    return {
        "has_sidewalk": has_sidewalk,
        "confidence": confidence,
        "reason": reason,
    }, None, model


def call_ai_summary(
    image_bytes: bytes,
    image_mime_type: str,
    prompt: str,
    predicted_class: str,
    confidence: float,
    ai_provider: str,
    llm_api_key: str = "",
    ai_model: str = "",
):
    text, error, model = call_llm_text(
        provider=ai_provider,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        prompt=prompt,
        api_key_override=llm_api_key,
        model_override=ai_model,
        max_output_tokens=520,
        temperature=0.2,
    )
    if error:
        fallback = fallback_summary_text(predicted_class, confidence)
        return fallback, f"{error} | Used fallback summary template.", model

    parsed = extract_json_object(text or "")
    validated = validate_summary_payload(parsed, predicted_class, confidence)
    if validated is not None:
        return format_summary_payload(validated), None, model

    retry_prompt = (
        prompt
        + "\n\nCRITICAL: Your previous response was invalid. Return ONLY one valid JSON object matching the schema."
    )
    retry_text, retry_error, retry_model = call_llm_text(
        provider=ai_provider,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        prompt=retry_prompt,
        api_key_override=llm_api_key,
        model_override=ai_model,
        max_output_tokens=520,
        temperature=0.1,
    )
    final_model = retry_model or model
    if retry_error:
        fallback = fallback_summary_text(predicted_class, confidence)
        return fallback, f"{retry_error} | Used fallback summary template.", final_model

    retry_parsed = extract_json_object(retry_text or "")
    retry_validated = validate_summary_payload(retry_parsed, predicted_class, confidence)
    if retry_validated is not None:
        return format_summary_payload(retry_validated), None, final_model

    fallback = fallback_summary_text(predicted_class, confidence)
    return fallback, "AI response was incomplete; used fallback summary template.", final_model


OBSTACLE_TYPES = ["BIKE_RACK", "TRASH_BIN", "UTILITY_POLE", "PLANTER", "CABINET", "BOX", "HYDRANT", "SIGNAL_POLE"]
INACCESSIBLE_MATERIALS = ["Gravel"]
POOR_MATERIALS = ["Brick"]
SIDEWALK_RESULTS = []
SUMMARY = {
    "total": 0,
    "compliant": 0,
    "non_compliant": 0,
    "critical": 0,
    "high": 0,
    "medium": 0,
    "missing_sidewalk": 0,
    "poor_condition": 0,
    "obstructed": 0,
    "compliance_rate": 0.0,
}
SIDEWALK_LOAD_ERROR = ""
SIDEWALK_LOCK = threading.Lock()

def find_obstacles(sidewalk_utm, obstacles_utm):
    nearby = obstacles_utm[obstacles_utm.geometry.distance(sidewalk_utm.geometry) < 2.0]
    result = []
    for _, obs in nearby.iterrows():
        result.append({
            "type": str(obs.get('asset_type', 'UNKNOWN')),
            "condition": str(obs.get('condition', 'Unknown')),
            "distance_m": round(float(obs.geometry.distance(sidewalk_utm.geometry)), 2)
        })
    return result

def analyze_sidewalks():
    print("Loading Brookline data...")
    assets = gpd.read_file("aboveGroundAssets.geojson")
    assets_utm = assets.to_crs("EPSG:32619")
    print(f"Loaded {len(assets)} assets")

    sidewalks_utm = assets_utm[assets_utm['asset_type'] == 'SIDEWALK'].copy()
    obstacles_utm = assets_utm[assets_utm['asset_type'].isin(OBSTACLE_TYPES)]
    sidewalks_orig = assets[assets['asset_type'] == 'SIDEWALK']

    results = []
    print(f"Analyzing {len(sidewalks_utm)} sidewalks...")

    for idx, row in sidewalks_utm.iterrows():
        orig = sidewalks_orig.loc[idx]
        sidewalk_type = str(row.get('Type', 'Sidewalk'))
        condition = str(row.get('condition', 'Unknown'))
        material = str(row.get('Material', 'Unknown'))
        image_url = str(orig.get('image_url', ''))

        # Real violations only - no fake measurements
        violations = []
        severity = "compliant"

        # 1. Missing sidewalk entirely
        if sidewalk_type == 'No Sidewalk':
            violations.append("No sidewalk present — pedestrians forced onto road")
            severity = "critical"

        # 2. Poor condition
        elif condition == 'Poor':
            violations.append("Poor condition — surface hazard for wheelchair users")
            severity = "high"

        # 3. Inaccessible material
        if material in INACCESSIBLE_MATERIALS:
            violations.append(f"{material} surface — not ADA compliant")
            severity = "high"

        # 4. Difficult material
        elif material in POOR_MATERIALS and condition != 'Good':
            violations.append(f"{material} surface in {condition} condition — accessibility risk")
            if severity == "compliant":
                severity = "medium"

        # 5. Real obstacles from data
        obstacles = find_obstacles(row, obstacles_utm)
        if obstacles:
            violations.append(f"{len(obstacles)} obstacle(s) within 2m of path")
            if severity == "compliant":
                severity = "medium"

        # 6. Under construction
        if condition == 'Under Construction':
            violations.append("Under construction — temporarily inaccessible")
            if severity == "compliant":
                severity = "medium"

        ada_compliant = len(violations) == 0

        results.append({
            "feature_id": str(row.get('feature_id', idx)),
            "geometry": orig.geometry.__geo_interface__,
            "sidewalk_type": sidewalk_type,
            "condition": condition,
            "material": material,
            "image_url": image_url,
            "ada_compliant": ada_compliant,
            "severity": severity,
            "violations": violations,
            "obstacles": obstacles,
            "obstacle_count": len(obstacles)
        })

    print("Analysis complete!")
    return results


def build_summary(sidewalk_results):
    if not sidewalk_results:
        return {
            "total": 0,
            "compliant": 0,
            "non_compliant": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "missing_sidewalk": 0,
            "poor_condition": 0,
            "obstructed": 0,
            "compliance_rate": 0.0,
        }

    compliant = sum(1 for s in sidewalk_results if s['ada_compliant'])
    critical = sum(1 for s in sidewalk_results if s['severity'] == 'critical')
    high = sum(1 for s in sidewalk_results if s['severity'] == 'high')
    medium = sum(1 for s in sidewalk_results if s['severity'] == 'medium')
    missing = sum(1 for s in sidewalk_results if s['sidewalk_type'] == 'No Sidewalk')
    poor_condition = sum(1 for s in sidewalk_results if s['condition'] == 'Poor')
    obstructed = sum(1 for s in sidewalk_results if s['obstacle_count'] > 0)
    total = len(sidewalk_results)

    return {
        "total": total,
        "compliant": compliant,
        "non_compliant": total - compliant,
        "critical": critical,
        "high": high,
        "medium": medium,
        "missing_sidewalk": missing,
        "poor_condition": poor_condition,
        "obstructed": obstructed,
        "compliance_rate": round(compliant / total * 100, 1),
    }


def ensure_sidewalk_data_loaded():
    global SIDEWALK_RESULTS, SUMMARY, SIDEWALK_LOAD_ERROR

    if SIDEWALK_RESULTS:
        return

    with SIDEWALK_LOCK:
        if SIDEWALK_RESULTS:
            return

        if SIDEWALK_CACHE_PATH.exists():
            try:
                cached = json.loads(SIDEWALK_CACHE_PATH.read_text())
                cached_results = cached.get("sidewalks") or []
                cached_summary = cached.get("summary") or build_summary(cached_results)
                SIDEWALK_RESULTS = cached_results
                SUMMARY = cached_summary
                SIDEWALK_LOAD_ERROR = ""
                print(f"Loaded sidewalk cache from {SIDEWALK_CACHE_PATH} ({len(SIDEWALK_RESULTS)} sidewalks)")
                return
            except Exception as exc:
                print(f"Failed to load sidewalk cache ({SIDEWALK_CACHE_PATH}): {exc}")

        try:
            print("Running sidewalk analysis...")
            SIDEWALK_RESULTS = analyze_sidewalks()
            SUMMARY = build_summary(SIDEWALK_RESULTS)
            SIDEWALK_LOAD_ERROR = ""
            print(f"Summary: {SUMMARY}")
            try:
                SIDEWALK_CACHE_PATH.write_text(json.dumps({"sidewalks": SIDEWALK_RESULTS, "summary": SUMMARY}))
                print(f"Saved sidewalk cache to {SIDEWALK_CACHE_PATH}")
            except Exception as exc:
                print(f"Could not save sidewalk cache: {exc}")
        except Exception as exc:
            SIDEWALK_RESULTS = []
            SUMMARY = build_summary(SIDEWALK_RESULTS)
            SIDEWALK_LOAD_ERROR = str(exc)
            print(f"Sidewalk analysis unavailable: {SIDEWALK_LOAD_ERROR}")

@app.get("/sidewalks")
def get_sidewalks():
    ensure_sidewalk_data_loaded()
    if SIDEWALK_LOAD_ERROR and not SIDEWALK_RESULTS:
        raise HTTPException(status_code=503, detail=f"Sidewalk analysis unavailable: {SIDEWALK_LOAD_ERROR}")
    return {"sidewalks": SIDEWALK_RESULTS, "summary": SUMMARY}

@app.get("/summary")
def get_summary():
    ensure_sidewalk_data_loaded()
    if SIDEWALK_LOAD_ERROR and not SIDEWALK_RESULTS:
        raise HTTPException(status_code=503, detail=f"Sidewalk analysis unavailable: {SIDEWALK_LOAD_ERROR}")
    return SUMMARY

@app.get("/violations")
def get_violations():
    ensure_sidewalk_data_loaded()
    if SIDEWALK_LOAD_ERROR and not SIDEWALK_RESULTS:
        raise HTTPException(status_code=503, detail=f"Sidewalk analysis unavailable: {SIDEWALK_LOAD_ERROR}")
    violations = [s for s in SIDEWALK_RESULTS if not s['ada_compliant']]
    return {"violations": violations, "count": len(violations)}


@app.post("/predict-sidewalk")
async def predict_sidewalk(
    image: UploadFile = File(...),
    include_gemini: bool = Form(True),
    ai_provider: str = Form(LLM_PROVIDER_DEFAULT),
    llm_api_key: str = Form(""),
    ai_model: str = Form(""),
    guidance_prompt: str = Form(""),
    enforce_sidewalk_check: bool = Form(True),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    normalized_provider = normalize_llm_provider(ai_provider)
    ai_summary = None
    ai_error = None
    prompt_used = ""
    ai_model_used = None
    sidewalk_check = {
        "checked": False,
        "has_sidewalk": None,
        "confidence": None,
        "reason": "",
        "error": None,
        "model": None,
    }
    classification_skipped = False

    if enforce_sidewalk_check:
        sidewalk_check["checked"] = True
        check_result, check_error, check_model = detect_sidewalk_presence(
            image_bytes=image_bytes,
            image_mime_type=image.content_type,
            ai_provider=normalized_provider,
            llm_api_key=llm_api_key,
            ai_model=ai_model,
        )
        sidewalk_check["model"] = check_model

        if check_result is None:
            sidewalk_check["error"] = check_error
        else:
            sidewalk_check["has_sidewalk"] = check_result["has_sidewalk"]
            sidewalk_check["confidence"] = check_result["confidence"]
            sidewalk_check["reason"] = check_result["reason"]

            if not check_result["has_sidewalk"]:
                classification_skipped = True
                ai_model_used = check_model
                ai_summary = (
                    "No sidewalk detected in this image, so Good/Fair/Poor classification was skipped. "
                    "Upload a real street-side sidewalk photo to get condition scoring and improvement guidance."
                )
                return {
                    "predicted_class": None,
                    "confidence": None,
                    "probabilities": {},
                    "classification_skipped": classification_skipped,
                    "sidewalk_check": sidewalk_check,
                    "ai_provider": normalized_provider,
                    "ai_summary": ai_summary,
                    "ai_error": ai_error,
                    "ai_model": ai_model_used,
                    "ai_prompt": prompt_used,
                    "gemini_summary": ai_summary,
                    "gemini_error": ai_error,
                    "gemini_model": ai_model_used,
                    "gemini_prompt": prompt_used,
                }

    predicted_class, confidence, probabilities = predict_sidewalk_quality(image_bytes)

    if include_gemini:
        prompt_used = build_advisor_prompt(predicted_class, confidence, probabilities, guidance_prompt)
        ai_summary, ai_error, ai_model_used = call_ai_summary(
            image_bytes=image_bytes,
            image_mime_type=image.content_type,
            prompt=prompt_used,
            predicted_class=predicted_class,
            confidence=confidence,
            ai_provider=normalized_provider,
            llm_api_key=llm_api_key,
            ai_model=ai_model,
        )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "classification_skipped": classification_skipped,
        "sidewalk_check": sidewalk_check,
        "ai_provider": normalized_provider,
        "ai_summary": ai_summary,
        "ai_error": ai_error,
        "ai_model": ai_model_used,
        "ai_prompt": prompt_used,
        "gemini_summary": ai_summary,
        "gemini_error": ai_error,
        "gemini_model": ai_model_used,
        "gemini_prompt": prompt_used,
    }
