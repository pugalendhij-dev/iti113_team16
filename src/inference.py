"""SageMaker inference handler - single Pipeline, text in / top-3 out."""
import os
import json
import joblib
import numpy as np

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(body)
    # accept {"text": "..."} or {"context": "..."} or a bare list of strings
    if isinstance(payload, dict):
        text = payload.get("text", payload.get("context", ""))
        return [text]
    if isinstance(payload, list):
        return [str(t) for t in payload]
    return [str(payload)]

def predict_fn(inputs, model):
    preds = model.predict(inputs)
    out = []
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(inputs)
        classes = model.classes_
        for i, p in enumerate(preds):
            order = np.argsort(proba[i])[::-1][:3]
            top3 = [{"marker": str(classes[j]), "score": round(float(proba[i][j]), 4)}
                    for j in order]
            out.append({"prediction": str(p), "top3": top3})
    else:
        for p in preds:
            out.append({"prediction": str(p), "top3": []})
    return out

def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
