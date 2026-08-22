"""
Unit tests for singlish_labelling.py — the joint-owned train/serve contract.

Why these matter for CI: preprocess.py (training) and inference.py (endpoint)
both compute the context string through this module. If the labelling logic
changes, training and serving would silently diverge (train/serve skew). These
tests pin the behaviour the corpus was labelled with, so any change that would
break that consistency fails the build before it reaches SageMaker.

Run locally:  pytest -q
"""
import pandas as pd
import pytest

import singlish_labelling as SL


# --- Option C: sentence-final particle detection + stripping ------------------

@pytest.mark.parametrize("text,label,context", [
    ("Meet after lunch la...", "lah", "Meet after lunch"),  # spelling variant la->lah, ellipsis stripped
    ("so expensive meh",       "meh", "so expensive"),
    ("number one",             "one", "number"),
])
def test_sentence_final_particle_is_stripped_and_canonicalised(text, label, context):
    r = SL.label_message(text)
    assert r["label"] == label
    assert r["context"] == context


def test_mid_clause_particle_is_not_labelled():
    # "la" sits mid-clause, not sentence-final -> must stay 'none' (Option C intent)
    r = SL.label_message("great world la e bugis")
    assert r["label"] == "none"
    assert "la" in r["context"]  # not stripped


def test_message_with_no_particle_is_none():
    r = SL.label_message("Bugis oso near wat...")
    assert r["label"] == "none"


# --- emoji families -----------------------------------------------------------

def test_emoticon_family_becomes_emo_label():
    r = SL.label_message("ok can :)")
    assert r["label"] == "emo_joy"
    assert r["has_emoticon"] is True
    assert r["context"] == "ok can"


def test_love_family():
    assert SL.label_message("I love you <3")["label"] == "emo_love"


def test_particle_wins_over_emoticon_but_cooccurrence_is_recorded():
    # design decision: particle takes the slot; emoticon still flagged for analysis
    r = SL.label_message("dunno leh :(")
    assert r["label"] == "leh"
    assert r["has_emoticon"] is True


def test_emoji_labels_stay_within_the_declared_families():
    allowed = {f"emo_{f}" for f in SL.EMOTICON_FAMILIES}
    assert allowed == {"emo_joy", "emo_playful", "emo_sad", "emo_love"}


# --- inference-side consistency ----------------------------------------------

def test_strip_for_inference_matches_training_context():
    # the endpoint must produce the same X as training for the same raw text
    assert SL.strip_for_inference("ok can lah") == "ok can"
    assert SL.strip_for_inference("ok can") == "ok can"


# --- PII scrubbing (governance) ----------------------------------------------

def test_phone_number_is_masked():
    assert SL.scrub_pii("call me at 91234567 now") == "call me at <PHONE> now"


def test_email_and_url_are_masked():
    out = SL.scrub_pii("email a@b.com see www.x.com")
    assert "<EMAIL>" in out and "<URL>" in out


# --- dataframe helper ---------------------------------------------------------

def test_build_labelled_frame_dedupes_and_labels():
    df = pd.DataFrame({"text": ["ok lah", "ok lah", "so happy :)"]})
    out = SL.build_labelled_frame(df)
    assert len(out) == 2  # one exact duplicate removed
    assert set(out["label"]) == {"lah", "emo_joy"}
    assert {"context", "label", "has_emoticon"}.issubset(out.columns)
