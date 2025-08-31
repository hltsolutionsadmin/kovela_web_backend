import os
import json
import base64
import threading
import time
from io import BytesIO
from typing import List, Tuple

from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import faiss

# -----------------------
# Config
# -----------------------
DB_PATH = os.environ.get("FACE_DB_PATH", "./face_db")
INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "./face_index.faiss")
IDS_PATH = os.environ.get("FAISS_IDS_PATH", "./faiss_ids.npy")
MODEL_NAME = os.environ.get("MODEL_NAME", "ArcFace")
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "10"))
DEFAULT_THRESHOLD = float(os.environ.get("THRESHOLD", "0.35"))
ANTI_SPOOF = os.environ.get("ANTI_SPOOF", "false").lower() == "true"

os.makedirs(DB_PATH, exist_ok=True)

app = Flask(__name__)

# -----------------------
# Globals
# -----------------------
MODEL = DeepFace.build_model(MODEL_NAME)
D = 512
INDEX = None
IDS: List[str] = []
INDEX_LOCK = threading.Lock()

def _load_index():
    global INDEX, IDS
    if os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH):
        INDEX = faiss.read_index(INDEX_PATH)
        IDS = np.load(IDS_PATH, allow_pickle=True).tolist()
        if INDEX.ntotal != len(IDS):
            raise RuntimeError("FAISS index and ids.npy are out of sync.")
    else:
        INDEX = faiss.IndexFlatIP(D)

def _save_index():
    faiss.write_index(INDEX, INDEX_PATH)
    np.save(IDS_PATH, np.array(IDS, dtype=object), allow_pickle=True)

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype("float32")
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def _b64_to_image_bytes(b64: str) -> bytes:
    return base64.b64decode(b64)

def _write_image(external_id: str, img_bytes: bytes) -> str:
    path = os.path.join(DB_PATH, f"{external_id}.jpg")
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path

def _read_image_b64(external_id: str) -> str:
    path = os.path.join(DB_PATH, f"{external_id}.jpg")
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _represent_image_bytes(img_bytes: bytes) -> Tuple[np.ndarray, str]:
    temp_path = os.path.join(DB_PATH, "__tmp__.jpg")
    with open(temp_path, "wb") as f:
        f.write(img_bytes)

    try:
        reps = DeepFace.represent(
            img_path=temp_path,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend="opencv",  # safer default than "yunet"
            align=True,
            anti_spoofing=ANTI_SPOOF
        )

        if not reps or "embedding" not in reps[0]:
            raise ValueError("Face/embedding not found or spoof detected.")

        emb = np.array(reps[0]["embedding"], dtype="float32")
        emb = _l2_normalize(emb)

        thumb_b64 = base64.b64encode(open(temp_path, "rb").read()).decode("utf-8")
        return emb, thumb_b64
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

def _search_top_k(query_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    q = query_emb.reshape(1, -1).astype("float32")
    scores, idx = INDEX.search(q, top_k)
    return scores, idx

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "count": INDEX.ntotal})

@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json(force=True)
    b64 = data.get("base64Image") or data.get("base64_image")
    if not b64 or len(b64) < 1000:
        return jsonify({"error": "Base64Image is required/too short"}), 400
    try:
        img_bytes = _b64_to_image_bytes(b64)
        emb, thumb_b64 = _represent_image_bytes(img_bytes)
        return jsonify({"embedding": emb.tolist(), "thumbnail": thumb_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/enroll", methods=["POST"])
def enroll():
    data = request.get_json(force=True)
    b64 = data.get("base64Image") or data.get("base64_image")
    external_id = data.get("externalId") or data.get("external_id")
    if not b64 or len(b64) < 1000:
        return jsonify({"error": "Base64Image is required/too short"}), 400
    if not external_id:
        return jsonify({"error": "externalId is required"}), 400

    try:
        img_bytes = _b64_to_image_bytes(b64)
        emb, thumb_b64 = _represent_image_bytes(img_bytes)

        _write_image(external_id, img_bytes)
        with INDEX_LOCK:
            INDEX.add(emb.reshape(1, -1))
            IDS.append(external_id)
            _save_index()

        return jsonify({
            "message": "Enrolled successfully",
            "externalId": external_id,
            "embedding": emb.tolist(),
            "thumbnail": thumb_b64,
            "count": INDEX.ntotal
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/check", methods=["POST"])
def check():
    data = request.get_json(force=True)
    b64 = data.get("base64Image") or data.get("base64_image")
    top_k = int(data.get("topK") or DEFAULT_TOP_K)
    threshold = float(data.get("threshold") or DEFAULT_THRESHOLD)
    include_thumbs = bool(data.get("includeThumbnails", True))

    if not b64 or len(b64) < 1000:
        return jsonify({"error": "Base64Image is required/too short"}), 400
    if INDEX.ntotal == 0:
        return jsonify({
            "type": "new",
            "message": "No user enrolled yet",
            "matches": [],
            "bestScore": 0.0,
            "threshold": threshold
        })

    try:
        img_bytes = _b64_to_image_bytes(b64)
        emb, _ = _represent_image_bytes(img_bytes)

        with INDEX_LOCK:
            scores, idx = _search_top_k(emb, min(top_k, max(1, INDEX.ntotal)))

        scores = scores[0]
        idx = idx[0]

        matches = []
        best_score = 0.0
        for sc, ix in zip(scores, idx):
            if ix < 0:
                continue
            external_id = IDS[ix]
            best_score = max(best_score, float(sc))
            match = {"externalId": external_id, "score": float(sc)}
            if include_thumbs:
                match["thumbnail"] = _read_image_b64(external_id)
            matches.append(match)

        resp_type = "existing" if best_score >= threshold else "new"

        return jsonify({
            "type": resp_type,
            "bestScore": best_score,
            "threshold": threshold,
            "matches": matches
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    _load_index()
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
