"""SageMaker Training Job - fits one sklearn Pipeline (TF-IDF + classifier).

The whole preprocessing+model is a single Pipeline, saved as model.joblib.
--model-type selects the classifier so the same script serves both models.
"""
import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--model-type", type=str, default="logreg")  # logreg | svm | tree
parser.add_argument("--random-state", type=int, default=42)
parser.add_argument("--team-id",   type=str, default=os.environ.get("TEAM_ID", "team16"))
parser.add_argument("--student-id",type=str, default=os.environ.get("STUDENT_ID", "s1602"))
parser.add_argument("--run-name",  type=str, default="sagemaker_pipeline_run")
parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
parser.add_argument("--test",  type=str, default=os.environ.get("SM_CHANNEL_TEST",  "/opt/ml/input/data/test"))
args = parser.parse_args()

os.makedirs(args.model_dir, exist_ok=True)

X_train = pd.read_csv(os.path.join(args.train, "train_features.csv"))["context"].fillna("")
y_train = pd.read_csv(os.path.join(args.train, "train_labels.csv"))["label"]
X_test  = pd.read_csv(os.path.join(args.test,  "test_features.csv"))["context"].fillna("")
y_test  = pd.read_csv(os.path.join(args.test,  "test_labels.csv"))["label"]
print(f"Train: {len(X_train)} | Test: {len(X_test)} | model-type: {args.model_type}")

# --- choose the classifier (this is how one pipeline serves both models) ---
def make_classifier(kind):
    if kind == "svm":
        # LinearSVC has no predict_proba; wrap for top-3 support.
        return CalibratedClassifierCV(LinearSVC(class_weight="balanced"))
    if kind == "tree":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=args.random_state, n_jobs=-1)
    # default: logreg baseline
    return LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=args.random_state)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000, min_df=3, ngram_range=(1, 3),
        token_pattern=r"(?u)\b\w+\b|EMOJI_\w+")),
    ("clf", make_classifier(args.model_type)),
])
pipe.fit(X_train, y_train)

def top_k_acc(model, X, y, k=3):
    if not hasattr(model, "predict_proba"):
        return float("nan")
    proba = model.predict_proba(X)
    classes = model.classes_
    topk = classes[np.argsort(proba, axis=1)[:, -k:]]
    return float(np.mean([yt in row for yt, row in zip(y, topk)]))

y_pred = pipe.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
acc      = accuracy_score(y_test, y_pred)
top3     = top_k_acc(pipe, X_test, y_test, k=3)

# Print in a form SageMaker metric_definitions can capture.
print(f"test_macro_f1: {macro_f1:.4f}")
print(f"test_accuracy: {acc:.4f}")
print(f"test_top3: {top3:.4f}")
print("=== classification report ===")
print(classification_report(y_test, y_pred, zero_division=0))

# --- save the single Pipeline artefact ---
joblib.dump(pipe, os.path.join(args.model_dir, "model.joblib"))
print("Saved model.joblib (single Pipeline: TF-IDF + classifier)")
