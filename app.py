"""
Provenance Guard — Milestone 4 backend.

A small Flask service that accepts text submissions, runs two independent
AI-detection signals (a Groq LLM signal and a pure-Python stylometric signal),
combines them into a single confidence score, assigns a provenance attribution,
and appends a structured entry to a persistent append-only audit log.

Endpoints
---------
POST /submit   Accept a JSON submission, classify it, and log the result.
GET  /log      Return every structured audit log entry as JSON.

Environment
-----------
GROQ_API_KEY   Required. Your Groq API key (loaded from the environment or .env).

Usage
-----
    # Test the Groq signal on its own, without starting the server:
    python app.py test "The quick brown fox jumps over the lazy dog."

    # Test the stylometric heuristic signal on its own:
    python app.py heuristic "The quick brown fox jumps over the lazy dog."

    # Start the Flask server on localhost:5000:
    python app.py
"""

import json
import os
import re
import statistics
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from groq import Groq

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
LOG_PATH = Path(__file__).with_name("log.json")

# Scoring-engine weights. The combined confidence score is:
#     (groq_score * GROQ_WEIGHT) + (heuristic_score * HEURISTIC_WEIGHT)
GROQ_WEIGHT = 0.6
HEURISTIC_WEIGHT = 0.4

# Attribution threshold applied to the combined confidence score.
ATTRIBUTION_THRESHOLD = 0.5

# Milestone 5 placeholder. The human-facing transparency label is derived from
# the confidence score in M5; until then we log a fixed placeholder.
PLACEHOLDER_LABEL = "pending_m5_transparency_label"

app = Flask(__name__)


# --------------------------------------------------------------------------- #
# Detection signal 1: Groq LLM
# --------------------------------------------------------------------------- #

def evaluate_groq_signal(text: str) -> float:
    """Score how likely `text` is AI-generated using the Groq LLM.

    Asks the model to judge overall flow, word choice, and stylistic
    characteristics, and to reply with ONLY a raw float in [0.0, 1.0], where
    0.0 is confidently human-written and 1.0 is confidently AI-generated.

    Returns the parsed score, clamped to [0.0, 1.0]. Returns -1.0 if the call
    or parse fails so callers can distinguish "no signal" from a real score.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")

    client = Groq(api_key=api_key)

    system_prompt = (
        "You are an expert AI-content detector. You evaluate a passage's overall "
        "flow, word choice, sentence rhythm, and stylistic characteristics to "
        "estimate the likelihood that it was written by an AI language model "
        "rather than a human.\n\n"
        "Respond with ONLY a single raw floating-point number between 0.0 and 1.0, "
        "where 0.0 means confidently human-written and 1.0 means confidently "
        "AI-generated. Do not include any words, labels, punctuation, or "
        "explanation — output the number and nothing else."
    )

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = completion.choices[0].message.content.strip()
    except Exception as exc:  # network error, auth error, rate limit, etc.
        app.logger.error("Groq API call failed: %s", exc)
        return -1.0

    return _parse_score(raw)


def _parse_score(raw: str) -> float:
    """Extract the first float from a model reply and clamp it to [0.0, 1.0]."""
    match = re.search(r"[-+]?\d*\.?\d+", raw)
    if not match:
        app.logger.error("Could not parse a score from Groq reply: %r", raw)
        return -1.0

    try:
        score = float(match.group())
    except ValueError:
        app.logger.error("Parsed token was not a valid float: %r", raw)
        return -1.0

    return max(0.0, min(1.0, score))


# --------------------------------------------------------------------------- #
# Detection signal 2: stylometric heuristics (pure Python, no external calls)
# --------------------------------------------------------------------------- #

# Sentence-length variability is measured with the coefficient of variation
# (std / mean). Human prose is "bursty" — a CV at or above this cap reads as
# fully chaotic; a CV of 0 (perfectly uniform sentences) reads as fully rigid.
CV_CAP = 1.0

# Punctuation density is punctuation marks per word. Human writing tends to be
# richer in expressive punctuation (dashes, semicolons, parentheticals); this
# density is treated as "fully human". Sparser punctuation reads as more rigid.
PUNCTUATION_DENSITY_CAP = 0.30

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"\b\w+\b")
_PUNCTUATION_RE = re.compile(r"""[.,;:!?…\-—–'"()\[\]{}]""")


def evaluate_heuristic_signal(text: str) -> float:
    """Score how AI-like `text` is from its style alone, no model call.

    Combines two stylometric measures, each mapped so that a rigid, uniform
    style scores high and a varied, chaotic style scores low:

      * Sentence-length uniformity — the inverse of the coefficient of
        variation of per-sentence word counts. Uniform sentence lengths look
        machine-like; wildly varying lengths look human.
      * Punctuation sparsity — the inverse of punctuation density (marks per
        word). Rich, frequent punctuation looks human; sparse punctuation
        looks rigid.

    The two measures are averaged equally into a single score in [0.0, 1.0],
    where 0.0 is a highly variable, chaotic human style and 1.0 is a highly
    uniform, rigid AI style.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    word_counts = [len(_WORD_RE.findall(s)) for s in sentences]
    word_counts = [c for c in word_counts if c > 0]
    total_words = sum(word_counts)

    # --- Measure 1: sentence-length uniformity ---
    if len(word_counts) >= 2 and statistics.mean(word_counts) > 0:
        mean_len = statistics.mean(word_counts)
        cv = statistics.pstdev(word_counts) / mean_len
        uniformity = 1.0 - min(cv / CV_CAP, 1.0)
    else:
        # Too little structure to judge variability; stay neutral.
        uniformity = 0.5

    # --- Measure 2: punctuation sparsity ---
    if total_words > 0:
        punctuation_count = len(_PUNCTUATION_RE.findall(text))
        density = punctuation_count / total_words
        sparsity = 1.0 - min(density / PUNCTUATION_DENSITY_CAP, 1.0)
    else:
        sparsity = 0.5

    score = (uniformity + sparsity) / 2.0
    return max(0.0, min(1.0, score))


# --------------------------------------------------------------------------- #
# Scoring engine
# --------------------------------------------------------------------------- #

def calculate_confidence_score(groq_score: float, heuristic_score: float) -> float:
    """Combine the two signals into a single confidence score in [0.0, 1.0].

    Formula: (groq_score * 0.6) + (heuristic_score * 0.4).
    """
    combined = (groq_score * GROQ_WEIGHT) + (heuristic_score * HEURISTIC_WEIGHT)
    return max(0.0, min(1.0, combined))


# --------------------------------------------------------------------------- #
# Persistent audit log (append-only log.json)
# --------------------------------------------------------------------------- #

def read_log() -> list:
    """Return all log entries, or an empty list if the log does not exist yet."""
    if not LOG_PATH.exists():
        return []
    try:
        with LOG_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        app.logger.error("Failed to read log file: %s", exc)
        return []


def append_log(entry: dict) -> None:
    """Append a single structured entry to the persistent log."""
    entries = read_log()
    entries.append(entry)
    with LOG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.post("/submit")
def submit():
    """Classify an incoming text submission and record it in the audit log."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object."), 400

    text = payload.get("text")
    creator_id = payload.get("creator_id")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="Field 'text' is required and must be a non-empty string."), 400
    if not isinstance(creator_id, str) or not creator_id.strip():
        return jsonify(error="Field 'creator_id' is required and must be a non-empty string."), 400

    content_id = str(uuid.uuid4())
    llm_score = evaluate_groq_signal(text)

    if llm_score < 0:
        return jsonify(error="Detection signal unavailable; submission was not classified."), 502

    heuristic_score = evaluate_heuristic_signal(text)
    confidence = calculate_confidence_score(llm_score, heuristic_score)

    attribution = "likely_ai" if confidence > ATTRIBUTION_THRESHOLD else "likely_human"

    entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attribution": attribution,
        "confidence": confidence,  # real combined score (groq*0.6 + heuristic*0.4)
        "llm_score": llm_score,
        "heuristic_score": heuristic_score,
        "label": PLACEHOLDER_LABEL,  # placeholder — transparency label lands in M5
        "status": "classified",
    }
    append_log(entry)

    return jsonify(entry), 201


@app.get("/log")
def get_log():
    """Return every structured audit log entry as JSON."""
    return jsonify(read_log()), 200


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

_DEFAULT_SAMPLE = (
    "In today's fast-paced digital landscape, leveraging synergistic "
    "solutions is paramount to unlocking transformative value."
)


def _run_signal_test(text: str) -> None:
    """Exercise evaluate_groq_signal from the terminal without the server."""
    print(f"Text under test:\n  {text!r}\n")
    score = evaluate_groq_signal(text)
    if score < 0:
        print("Groq signal FAILED (see logs above).")
        return
    label = "likely_ai" if score > ATTRIBUTION_THRESHOLD else "likely_human"
    print(f"Groq score : {score:.3f}")
    print(f"Attribution: {label}")


def _run_heuristic_test(text: str) -> None:
    """Exercise evaluate_heuristic_signal from the terminal without the server."""
    print(f"Text under test:\n  {text!r}\n")
    score = evaluate_heuristic_signal(text)
    label = "likely_ai" if score > ATTRIBUTION_THRESHOLD else "likely_human"
    print(f"Heuristic score: {score:.3f}  (0.0 = chaotic/human, 1.0 = rigid/AI)")
    print(f"Style read     : {label}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    sample = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_SAMPLE

    if mode == "test":
        _run_signal_test(sample)
    elif mode == "heuristic":
        _run_heuristic_test(sample)
    else:
        app.run(host="127.0.0.1", port=5000, debug=True)
