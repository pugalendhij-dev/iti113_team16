"""SageMaker Processing Job - Singlish marker data preparation (Option C).

Reads the raw NUS SMS Corpus JSON, applies Option C labelling via the shared
singlish_labelling module, removes duplicates, scrubs PII, and writes
train/test splits. No scaler is fitted here: TF-IDF lives in the model Pipeline.
"""
import os
import sys
import json
import argparse
import pandas as pd

# The shared module is shipped as a processing input mounted here.
sys.path.insert(0, "/opt/ml/processing/input/module")
import singlish_labelling as SL

parser = argparse.ArgumentParser()
parser.add_argument("--test-size",    type=float, default=0.20)
parser.add_argument("--random-state", type=int,   default=42)
parser.add_argument("--min-class-count", type=int, default=100)
args = parser.parse_args()

input_dir  = "/opt/ml/processing/input"
output_dir = "/opt/ml/processing/output"
os.makedirs(output_dir, exist_ok=True)

# --- locate the corpus json in the input folder ---
json_name = None
for f in os.listdir(input_dir):
    if f.endswith(".json"):
        json_name = f
        break
if json_name is None:
    raise FileNotFoundError(f"No .json corpus found in {input_dir}")
raw = json.loads(open(os.path.join(input_dir, json_name), encoding="utf-8").read())

# --- extract message text (same structure as Notebook 01) ---
def get_messages(raw):
    node = raw
    for key in ("smsCorpus", "message"):
        if isinstance(node, dict) and key in node:
            node = node[key]
    return node if isinstance(node, list) else [node]

def extract_text(m):
    if isinstance(m, dict):
        t = m.get("text")
        if isinstance(t, dict) and "$" in t:
            return str(t["$"])
        if isinstance(t, str):
            return t
    return str(m) if isinstance(m, str) else ""

df = pd.DataFrame({"text": [extract_text(m) for m in get_messages(raw)]})
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0].reset_index(drop=True)
print(f"Loaded {len(df):,} messages")

# --- scrub PII, then Option C label (dedup handled inside build_labelled_frame) ---
df["scrubbed"] = df["text"].apply(SL.scrub_pii)
labelled = SL.build_labelled_frame(
    df.rename(columns={"scrubbed": "text_for_labelling"}),
    text_col="text_for_labelling",
    use_emoji=True,
    drop_duplicates=True,
)
data = labelled[["context", "label"]].copy()
print("Rows after dedup+label:", len(data))

# --- drop rare PARTICLE classes (emoji classes protected), same rule as the notebook ---
counts = data["label"].value_counts()
drop_particles = [c for c in counts[counts < args.min_class_count].index
                  if c != "none" and not str(c).startswith("emo_")]
if drop_particles:
    print("Dropping rare particle classes:", drop_particles)
    data = data[~data["label"].isin(drop_particles)].reset_index(drop=True)

# --- also drop any class with < 2 examples so stratify works ---
vc = data["label"].value_counts()
too_rare = vc[vc < 2].index.tolist()
if too_rare:
    data = data[~data["label"].isin(too_rare)].reset_index(drop=True)

from sklearn.model_selection import train_test_split
X = data["context"].fillna("")
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

# --- write splits (X is text, Y is label) ---
pd.DataFrame({"context": X_train}).to_csv(os.path.join(output_dir, "train_features.csv"), index=False)
pd.DataFrame({"label":   y_train}).to_csv(os.path.join(output_dir, "train_labels.csv"),   index=False)
pd.DataFrame({"context": X_test}).to_csv(os.path.join(output_dir,  "test_features.csv"),  index=False)
pd.DataFrame({"label":   y_test}).to_csv(os.path.join(output_dir,  "test_labels.csv"),    index=False)

print(f"Wrote train ({len(X_train)}) and test ({len(X_test)}) splits to {output_dir}")
print("Classes:", sorted(y.unique()))
