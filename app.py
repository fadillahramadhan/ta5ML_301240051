import os, json
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "model", "model_b.joblib"))
meta = json.load(open(os.path.join(BASE, "model", "meta.json")))
FEATURES = meta["features"]

app = Flask(__name__)


def build_row(form):
    """Ubah input form menjadi satu baris fitur sesuai urutan kolom saat training."""
    age = form.get("age", "").strip()
    row = {
        "pclass": int(form["pclass"]),
        "age": float(age) if age else meta["median_age"],
        "sibsp": int(form.get("sibsp") or 0),
        "parch": int(form.get("parch") or 0),
        "fare": float(form.get("fare") or 0),
    }
    sex, emb = form["sex"], form["embarked"]
    for col in FEATURES:
        if col.startswith("sex_"):
            row[col] = int(col == f"sex_{sex}")
        elif col.startswith("embarked_"):
            row[col] = int(col == f"embarked_{emb}")
    return pd.DataFrame([row])[FEATURES]


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, values = None, None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            X = build_row(request.form)
            proba = model.predict_proba(X)[0]
            pred = int(model.predict(X)[0])
            result = {
                "label": "Selamat" if pred == 1 else "Tidak Selamat",
                "pred": pred,
                "p_selamat": round(float(proba[1]) * 100, 1),
                "p_tidak": round(float(proba[0]) * 100, 1),
            }
        except (ValueError, KeyError):
            error = "Input tidak valid. Periksa kembali isian formulir."
    return render_template("index.html", result=result, error=error, v=values, meta=meta)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
