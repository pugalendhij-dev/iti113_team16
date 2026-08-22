"""
singlish_labelling.py
=====================
Shared preprocessing and labelling module for the ITI113 Singlish keyboard project (Team 16).

This is the JOINT-OWNED interface between Ria's data/modelling work and Joseph's
pipeline/serving work. It is imported by:
  - Notebook 01 (EDA & data prep)   -> to build the labelled training table
  - Notebook 02 (baseline)          -> reuses the same functions, no divergence
  - src/preprocess.py (pipeline)    -> same logic inside the SageMaker Processing job
  - src/inference.py (endpoint)     -> strip_for_inference() only

Design decision: Option C ("slot prediction").
Given a message, we detect a sentence-final discourse particle in the LAST clause,
strip it, and use it as the label. Messages with no sentence-final particle are
labelled 'none'. This recovers particles that strict last-token labelling misses
(e.g. "Meet after lunch la...") while correctly ignoring mid-clause particles
(e.g. "great world la e bugis").

Keeping ONE copy of this logic is what prevents train/serve skew: training and the
live endpoint compute the context string identically.

NOTE ON LABEL NOISE: 'one' and 'ah' are genuinely ambiguous ("the ubi one" is a
particle; "number one" is not). Sentence-final position reduces but does not remove
this. Expect some 'one'/'ah' confusion in the matrix — it is the data, not the model.
"""

import re

# --------------------------------------------------------------------------
# Canonical particle map. Spelling variants collapse to one canonical label,
# because Singlish has no fixed orthography (la/lah are the same particle).
# Ria: finalise this set from the EDA counts. Rare forms (<10) were dropped;
# forms <150 were merged/kept per the label-set decision in the design doc.
# --------------------------------------------------------------------------
PARTICLE_CANON = {
    "la": "lah", "lah": "lah",
    "leh": "leh",
    "lor": "lor",
    "meh": "meh",
    "hor": "hor",
    "ah": "ah",
    "mah": "mah",
    "liao": "liao",
    "one": "one",
    "eh": "eh",
    "sia": "sia", "siah": "sia",
}
PARTICLES = set(PARTICLE_CANON)

# Emoticon families (Option: affective markers as a grouped class).
# Kept separate from particles; a message can carry both — see label_message().
EMOTICON_FAMILIES = {
    "joy":        [":)", ":-)", "=)", ":d", ":-d", "=d", "^^", "^_^"],
    "playful":    [":p", ":-p", "=p", ";)", ";-)"],
    "sad":        [":(", ":-(", "=(", ":'(", "t_t"],
    "love":       ["<3"],
}
# flatten for detection
_EMO_LOOKUP = {e: fam for fam, lst in EMOTICON_FAMILIES.items() for e in lst}
_EMO_RE = re.compile(
    r"(?::|;|=)-?[\)\(dpo]|\^\^|\^_\^|<3|:'\(|t_t", re.IGNORECASE
)

# Canonical EMOJI_* tokens (produced by upstream emoticon canonicalisation) mapped
# to families, so labelling works whether the text has raw emoticons or canonical tokens.
_CANON_EMOJI_FAMILY = {
    "EMOJI_HAPPY": "joy", "EMOJI_LAUGH": "joy",
    "EMOJI_PLAYFUL": "playful",
    "EMOJI_SAD": "sad", "EMOJI_CRY": "sad", "EMOJI_FRUSTRATED": "sad",
    "EMOJI_LOVE": "love", "EMOJI_SKEPTIC": "playful",
}
_CANON_EMOJI_RE = re.compile(r"EMOJI_[A-Z]+")

_TRAIL_RE = re.compile(r"[\s.\-!?,~]+$")
_CLAUSE_SPLIT_RE = re.compile(r"[.!?]+")


def clean_trailing(text):
    """Remove trailing punctuation / ellipsis / whitespace used to end SMS lines."""
    return _TRAIL_RE.sub("", str(text))


def detect_emoticon_family(text):
    """Return the first emoticon family present, or None.
    Recognises both canonical EMOJI_* tokens and raw emoticons."""
    m_canon = _CANON_EMOJI_RE.search(str(text))
    if m_canon and m_canon.group(0) in _CANON_EMOJI_FAMILY:
        return _CANON_EMOJI_FAMILY[m_canon.group(0)]
    m = _EMO_RE.search(str(text))
    if not m:
        return None
    tok = m.group(0).lower()
    return _EMO_LOOKUP.get(tok, None) or _closest_family(tok)


def _closest_family(tok):
    if tok.endswith(("d", "^", ")")):
        return "joy"
    if tok.endswith(("p",)):
        return "playful"
    if tok.endswith(("(",)):
        return "sad"
    return None


def extract_slot(text):
    """
    Option C core. Returns (context, particle_label).
    - context: the message with the sentence-final particle removed (model input X)
    - particle_label: canonical particle, or 'none'
    """
    t = clean_trailing(text)
    # A trailing emoticon (raw or canonical EMOJI_* token) can sit after the particle
    # ("dunno leh :(" or "dunno leh EMOJI_SAD"). Remove a trailing emoticon before
    # locating the sentence-final particle. The emoticon is captured separately by
    # detect_emoticon_family() for co-occurrence and priority handling.
    t = _CANON_EMOJI_RE.sub("", t).strip() if _CANON_EMOJI_RE.search(t) else t
    t = clean_trailing(_EMO_RE.sub("", t + " ").strip()) if _EMO_RE.search(t) else t
    if not t:
        return t, "none"
    clauses = [c for c in _CLAUSE_SPLIT_RE.split(t) if c.strip()]
    if not clauses:
        return t, "none"
    last = clauses[-1].strip()
    toks = last.split()
    if not toks:
        return t, "none"
    cand = re.sub(r"[^a-zA-Z]", "", toks[-1]).lower()
    if cand in PARTICLES:
        label = PARTICLE_CANON[cand]
        idx = t.lower().rfind(cand)
        context = (t[:idx] + t[idx + len(cand):]).strip()
        return clean_trailing(context), label
    return t, "none"


def label_message(text, use_emoji=True):
    """
    Full labelling for one message. Returns a dict:
      context      -> model input X (particle stripped)
      label        -> the target Y
      has_emoticon -> bool
    Label priority: a sentence-final particle wins; else an emoticon family (if
    use_emoji); else 'none'. Priority is a design choice — particles and emoji
    compete for the same slot, and we label the particle when both are present,
    recording co-occurrence via has_emoticon for later analysis.
    """
    context, particle = extract_slot(text)
    fam = detect_emoticon_family(text)
    has_emo = fam is not None
    if particle != "none":
        label = particle
    elif use_emoji and fam is not None:
        label = f"emo_{fam}"
    else:
        label = "none"
    return {"context": context, "label": label, "has_emoticon": has_emo}


def strip_for_inference(text):
    """
    Endpoint-side helper. At inference we do NOT know the true marker; we only
    need the context string exactly as training produced it. Returns the context
    with any trailing particle removed, so the live features match training.
    """
    context, _ = extract_slot(text)
    return context


# --------------------------------------------------------------------------
# Dataframe-level helpers used by Notebook 01 / preprocess.py
# --------------------------------------------------------------------------
def build_labelled_frame(df, text_col="text", use_emoji=True, drop_duplicates=True):
    """
    Takes a DataFrame with a text column, returns it with context/label/has_emoticon
    columns added. Deduplicates first (the EDA found 12.9% exact duplicates).
    """
    import pandas as pd
    out = df.copy()
    out[text_col] = out[text_col].astype(str).str.strip()
    out = out[out[text_col].str.len() > 0]
    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates(subset=[text_col]).reset_index(drop=True)
        print(f"Dropped {before - len(out):,} duplicate messages "
              f"({(before - len(out))/before*100:.1f}%)")
    labelled = out[text_col].apply(lambda t: label_message(t, use_emoji=use_emoji))
    out["context"] = [r["context"] for r in labelled]
    out["label"] = [r["label"] for r in labelled]
    out["has_emoticon"] = [r["has_emoticon"] for r in labelled]
    return out


# --------------------------------------------------------------------------
# PII scrubbing — governance requirement. Replace, do not drop, so context is
# preserved for the model while identifiers are removed.
# --------------------------------------------------------------------------
_PII_PATTERNS = [
    (re.compile(r"\b[89]\d{7}\b"), " <PHONE> "),
    (re.compile(r"\b\d{6,}\b"), " <NUM> "),
    (re.compile(r"\b[\w.]+@[\w.]+\.\w+\b"), " <EMAIL> "),
    (re.compile(r"https?://\S+|www\.\S+"), " <URL> "),
]


def scrub_pii(text):
    t = str(text)
    for pat, repl in _PII_PATTERNS:
        t = pat.sub(repl, t)
    return re.sub(r"\s+", " ", t).strip()


if __name__ == "__main__":
    # smoke test mirroring real corpus patterns
    samples = [
        "Meet after lunch la...",
        "so expensive meh",
        "great world la e bugis",
        "Bugis oso near wat...",
        "call 67441233 look for irene",
        "ok can :)",
        "dunno leh :(",
    ]
    for s in samples:
        r = label_message(s)
        print(f"{s!r:40s} -> label={r['label']:9s} ctx={r['context']!r}")
