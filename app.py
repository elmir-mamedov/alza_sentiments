from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "model/bert"  # local path after copying from Drive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

print(f"Model loaded on {device}")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = F.softmax(logits, dim=-1)

    confidence, predicted_class = torch.max(probs, dim=-1)

    return jsonify({
        "label":      "positive" if predicted_class.item() == 1 else "negative",
        "confidence": round(confidence.item() * 100, 1),
        "probs": {
            "negative": round(probs[0][0].item() * 100, 1),
            "positive": round(probs[0][1].item() * 100, 1),
        }
    })

if __name__ == "__main__":
    app.run(debug=True)