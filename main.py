import base64
import binascii
import io
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

MODEL_ID = os.getenv("MODEL_ID", "P82_LPxEy")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/keras_model.h5"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", "models/labels.txt"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))

model = None
class_names: list[str] = []

def read_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        # Teachable Machine labels often use the format "0 Class name".
        line = re.sub(r"^\d+\s+", "", line)
        labels.append(line)
    return labels

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names
    if not MODEL_PATH.exists():
        print(f"WARNING: No se encontró el modelo en {MODEL_PATH}. Descarga el modelo Keras de Teachable Machine y colócalo allí.")
    else:
        try:
            # Import TensorFlow only when a model file exists, so health/docs can still start.
            import tensorflow as tf
            model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
            class_names = read_labels(LABELS_PATH)
            print(f"Modelo {MODEL_ID} cargado desde {MODEL_PATH}")
            if not class_names:
                print(f"WARNING: No hay etiquetas en {LABELS_PATH}; se usarán nombres genéricos.")
        except Exception as exc:
            print(f"ERROR cargando el modelo: {exc}")
            model = None
    yield

app = FastAPI(
    title="Teachable Machine Classification API",
    version="1.0.0",
    description=f"API REST para clasificación de imágenes con el modelo Teachable Machine {MODEL_ID}.",
    lifespan=lifespan,
)

# Para desarrollo local. En producción, limita allow_origins a los dominios reales.
allowed_origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "*").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description="Imagen codificada en Base64; acepta también data:image/...;base64,...")
    filename: Optional[str] = "image.jpg"

@app.get("/")
def root():
    return {"status": "online", "service": "Teachable Machine API", "model_id": MODEL_ID, "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_id": MODEL_ID,
        "model_loaded": model is not None,
        "labels_loaded": len(class_names),
    }

def decode_image(contents: bytes) -> np.ndarray:
    if not contents:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"La imagen supera el límite de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        with Image.open(io.BytesIO(contents)) as source:
            image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            array = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="El archivo no contiene una imagen válida.")
    # Es el preprocesamiento utilizado por los modelos de imagen de Teachable Machine.
    return (array / 127.5 - 1.0)[None, ...]

def classify_bytes(contents: bytes, filename: Optional[str], content_type: Optional[str]):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"El modelo {MODEL_ID} no está cargado. Coloca keras_model.h5 y labels.txt en la carpeta models/ y reinicia FastAPI.",
        )
    if content_type and not content_type.lower().startswith("image/"):
        raise HTTPException(status_code=415, detail="El tipo de archivo debe ser una imagen.")
    tensor = decode_image(contents)
    try:
        output = model.predict(tensor, verbose=0)
        probabilities = np.asarray(output).reshape(-1)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {exc}")
    if probabilities.size == 0:
        raise HTTPException(status_code=500, detail="El modelo no devolvió predicciones.")

    results = []
    for index, probability in enumerate(probabilities):
        label = class_names[index] if index < len(class_names) else f"Clase {index}"
        results.append({"class": label, "label": label, "confidence": round(float(probability) * 100, 2)})
    top_prediction = max(results, key=lambda item: item["confidence"])
    return {
        "success": True,
        "status": "success",
        "model_id": MODEL_ID,
        "filename": filename,
        "top_prediction": top_prediction,
        "predictions": results,
    }

@app.post("/api/v1/classify")
async def classify_multipart(image: UploadFile = File(...)):
    contents = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"La imagen supera el límite de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    return classify_bytes(contents, image.filename, image.content_type)

@app.post("/api/v1/classify/base64")
async def classify_base64(payload: Base64ImageRequest):
    encoded = payload.image_base64.strip()
    if encoded.startswith("data:"):
        if "," not in encoded:
            raise HTTPException(status_code=400, detail="El data URL de la imagen no es válido.")
        header, encoded = encoded.split(",", 1)
        if ";base64" not in header:
            raise HTTPException(status_code=400, detail="La imagen debe estar codificada en Base64.")
        content_type = header[5:].split(";", 1)[0]
    else:
        content_type = "image/jpeg"
    # Check approximate decoded size before allocating the full buffer.
    if len(encoded) > ((MAX_UPLOAD_BYTES + 2) // 3) * 4 + 16:
        raise HTTPException(status_code=413, detail="La imagen Base64 supera el tamaño máximo permitido.")
    try:
        contents = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="El contenido image_base64 no es Base64 válido.")
    return classify_bytes(contents, payload.filename, content_type)

# Same path for both formats: the frontend sends JSON/Base64 or multipart.
@app.post("/api/v1/classify-json")
async def classify_json_alias(payload: Base64ImageRequest):
    return await classify_base64(payload)
