import os
import pickle
import io
import csv
from flask import Blueprint, render_template, request, jsonify, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

bp = Blueprint("main", __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DEMO_DATA_PATH = os.path.join(BASE_DIR, "demo_dataset.csv")


@bp.route("/")
def index():
    return render_template("index.html")


# ---------- Helper functions ----------

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["vectorizer"]


def parse_csv(file_obj):
    """Parse a CSV file-like object.
    Expected columns: label, text OR label, message.
    label should be 'spam' or 'ham' (case-insensitive).
    """
    file_obj.seek(0)
    reader = csv.DictReader(io.StringIO(file_obj.read().decode("utf-8", errors="ignore")))
    rows = list(reader)
    if not rows:
        raise ValueError("CSV file is empty or has no rows.")

    # detect text column
    fieldnames = [c.lower() for c in reader.fieldnames or []]
    if "text" in fieldnames:
        text_col = reader.fieldnames[fieldnames.index("text")]
    elif "message" in fieldnames:
        text_col = reader.fieldnames[fieldnames.index("message")]
    elif "sms" in fieldnames:
        text_col = reader.fieldnames[fieldnames.index("sms")]
    else:
        raise ValueError("CSV must have a 'text' or 'message' column.")

    if "label" in fieldnames:
        label_col = reader.fieldnames[fieldnames.index("label")]
    elif "category" in fieldnames:
        label_col = reader.fieldnames[fieldnames.index("category")]
    else:
        raise ValueError("CSV must have a 'label' or 'category' column.")

    texts = []
    labels = []

    for r in rows:
        txt = (r.get(text_col) or "").strip()
        lab_raw = (r.get(label_col) or "").strip().lower()
        if not txt:
            continue
        if lab_raw in ("spam", "1", "true"):
            lab = 1
        else:
            lab = 0
        texts.append(txt)
        labels.append(lab)

    if not texts:
        raise ValueError("No valid rows found in CSV.")

    return texts, labels


def train_model_from_texts(texts, labels):
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    model = MultinomialNB()
    model.fit(X, labels)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vec}, f)
    return len(texts)


# ---------- API endpoints ----------

@bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("message") or "").strip()
    if not text:
        return jsonify({"error": "Message is empty"}), 400

    model, vectorizer = load_model()
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not trained yet. Please use 'Train Demo Data' or 'Train on file' first."}), 400

    X = vectorizer.transform([text])
    y = model.predict(X)[0]
    label = "Spam" if int(y) == 1 else "Ham"

    print(f"[PREDICT] text={text!r} -> {label}")
    return jsonify({"prediction": label})


@bp.route("/train-demo", methods=["POST"])
def train_demo():
    if not os.path.exists(DEMO_DATA_PATH):
        return jsonify({"error": "Demo dataset not found on server."}), 500

    with open(DEMO_DATA_PATH, "rb") as f:
        try:
            texts, labels = parse_csv(f)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    n = train_model_from_texts(texts, labels)
    print(f"[TRAIN-DEMO] Trained on {n} samples.")
    return jsonify({"status": "ok", "samples": n})


@bp.route("/train-file", methods=["POST"])
def train_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded."}), 400

    try:
        texts, labels = parse_csv(file)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    n = train_model_from_texts(texts, labels)
    print(f"[TRAIN-FILE] Trained on {n} samples from uploaded file.")
    return jsonify({"status": "ok", "samples": n})


@bp.route("/download-model", methods=["GET"])
def download_model():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet."}), 404
    return send_file(MODEL_PATH, as_attachment=True, download_name="model.pkl")
