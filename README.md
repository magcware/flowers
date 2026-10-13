# FastAPI — Teachable Machine P82_LPxEy

Este backend recibe imágenes mediante HTTP y devuelve las predicciones del modelo. Incluye dos formatos de entrada:

- `POST /api/v1/classify` — `multipart/form-data`, campo `image`.
- `POST /api/v1/classify/base64` — JSON con `image_base64` (data URL o Base64 puro) y `filename`.
- `POST /api/v1/classify/base64` — alias explícito para JSON/Base64.
- `GET /health` — estado de carga del modelo.
- `GET /docs` — Swagger interactivo.

## 1. Exportar el modelo

El HTML puede usar el modelo alojado de TensorFlow.js, pero TensorFlow/Keras en Python necesita el archivo Keras exportado. En Teachable Machine, abre el proyecto asociado al ID `P82_LPxEy`, selecciona **Export Model → TensorFlow → Keras** y coloca `keras_model.h5` y `labels.txt` en `models/`.

No es posible convertir automáticamente `model.json` de TensorFlow.js a un modelo Keras usando solo esta URL. Si ya no tienes acceso al proyecto original, descarga los archivos Keras desde una copia/exportación que conserves.

## 2. Instalar y ejecutar

Se recomienda Python 3.10 o 3.11 para TensorFlow 2.15.1. Desde esta carpeta:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Comprueba `http://127.0.0.1:8000/health` y abre `http://127.0.0.1:8000/docs`. Si el modelo no se encuentra, la API inicia pero responde `503` en la clasificación hasta que se agreguen los archivos del modelo.

## 3. Conectar el HTML

Abre `index.html` y en la sección **Consumo por API** deja la URL `http://127.0.0.1:8000`. Elige el formato y ejecuta la petición. El HTML ya no simula la respuesta: envía un `fetch()` HTTP al backend y muestra la respuesta o el error real.

Si el HTML y FastAPI están en equipos distintos, cambia la URL por la dirección/IP del servidor. Si se publica en Internet, usa HTTPS y limita `CORS_ORIGINS` a los dominios autorizados. CORS no sustituye autenticación; agrega autenticación y límites adicionales antes de exponerlo públicamente.
