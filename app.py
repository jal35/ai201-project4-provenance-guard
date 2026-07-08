"""
Provenance Guard — Milestone 3 backend.

A small Flask service that accepts text submissions, runs a first AI-detection
signal through the Groq LLM API, assigns a provenance attribution, and appends a
structured entry to a persistent append-only audit log.

Endpoints
---------
POST /submit   Accept a JSON submission, classify it, and log the result.
GET  /log      Return every structured audit log entry as JSON.

Environment
-----------
GROQ_API_KEY   Required. Your Groq API key (loaded from the environment or .env).

Usage
-----
    # Test the Groq signal function on its own, without starting the server:
    python app.py test "The quick brown fox jumps over the lazy dog."

    # Start the Flask server on localhost:5000:
    python app.py
"""

import json
import os
import re
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

# Milestone 3 placeholders. These are wired in for the API contract now and will
# be replaced with real values once the additional detection signals land.
PLACEHOLDER_CONFIDENCE = 0.5
ATTRIBUTION_THRESHOLD = 0.5

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

    attribution = "likely_ai" if llm_score > ATTRIBUTION_THRESHOLD else "likely_human"

    entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attribution": attribution,
        "confidence": PLACEHOLDER_CONFIDENCE,  # placeholder — combined score TBD
        "llm_score": llm_score,
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sample = sys.argv[2] if len(sys.argv) > 2 else (
            "In today's fast-paced digital landscape, leveraging synergistic "
            "solutions is paramount to unlocking transformative value."
        )
        _run_signal_test(sample)
    else:
        app.run(host="127.0.0.1", port=5000, debug=True)
