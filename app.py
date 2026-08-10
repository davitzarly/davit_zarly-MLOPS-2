"""Flask API gateway for serving the NEU-DET defect classifier.

This app receives an uploaded JPEG image, base64-encodes the raw bytes to
match the model's ``image_bytes`` serving signature, calls TensorFlow
Serving (localhost:8501 by default) and returns the predicted class name
together with a confidence score.

Prometheus metrics are exposed at /metrics for monitoring.
"""
from __future__ import annotations

import base64
import os
from typing import List

import numpy as np
import requests
from flask import Flask, jsonify, request, Response
from prometheus_client import (
    Counter,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TF_SERVING_URL = os.environ.get(
    "TF_SERVING_URL",
    "http://localhost:8501/v1/models/neu_det_cnn:predict",
)
MODEL_NAME = os.environ.get("MODEL_NAME", "neu_det_cnn")
PORT = int(os.environ.get("PORT", 8080))

CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# --------------------------------------------------------------------------- #
# Prometheus metrics
# --------------------------------------------------------------------------- #
PREDICT_TOTAL = Counter(
    "model_predict_total",
    "Total number of /predict requests received.",
)
PREDICT_ERRORS = Counter(
    "model_predict_errors_total",
    "Total number of /predict requests that failed.",
)
PREDICT_LATENCY = Histogram(
    "model_predict_latency_seconds",
    "Latency of /predict requests in seconds.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
MODEL_INFO = Info(
    "model",
    "Information about the served model.",
)
MODEL_INFO.info({"name": MODEL_NAME, "classes": ",".join(CLASSES)})

# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/predict", methods=["POST"])
@PREDICT_LATENCY.time()
def predict():
    """Run a single prediction on the uploaded image."""
    PREDICT_TOTAL.inc()

    if "image" not in request.files:
        PREDICT_ERRORS.inc()
        return jsonify({"error": "no image provided"}), 400

    file = request.files["image"]
    try:
        raw_bytes = file.read()
        if not raw_bytes:
            raise ValueError("uploaded file is empty")
        image_b64 = base64.b64encode(raw_bytes).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        PREDICT_ERRORS.inc()
        return jsonify({"error": f"failed to read uploaded image: {exc}"}), 400

    # The model's serving signature (see modules/trainer.py::_get_serve_image_bytes_fn)
    # decodes and resizes the JPEG itself, so we just forward the raw bytes,
    # base64-encoded as required by TF Serving's REST API for string tensors.
    payload = {"instances": [{"image_bytes": {"b64": image_b64}}]}
    try:
        resp = requests.post(TF_SERVING_URL, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        PREDICT_ERRORS.inc()
        return jsonify({"error": f"TF Serving request failed: {exc}"}), 502

    try:
        probs = np.asarray(resp.json()["predictions"][0], dtype=np.float32)
    except (KeyError, IndexError, ValueError) as exc:
        PREDICT_ERRORS.inc()
        return jsonify({"error": f"invalid response from TF Serving: {exc}"}), 502

    top_idx = int(np.argmax(probs))
    return jsonify({
        "class_name": CLASSES[top_idx],
        "class_index": top_idx,
        "confidence": float(probs[top_idx]),
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
