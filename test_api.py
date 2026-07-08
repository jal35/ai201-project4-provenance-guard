"""
Smoke test for the running Provenance Guard server.

Start the server first (`python app.py`), then in another terminal run:

    python test_api.py
    python test_api.py "Some custom text to classify"

Uses only the standard library so it needs no extra dependencies.
"""

import json
import sys
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:5000"


def post_submit(text: str, creator_id: str = "test-user") -> None:
    body = json.dumps({"text": text, "creator_id": creator_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/submit",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"POST /submit  ->  text={text!r}")
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"  {resp.status} {resp.reason}")
            print("  " + json.dumps(json.load(resp), indent=2).replace("\n", "\n  "))
    except urllib.error.HTTPError as exc:
        print(f"  {exc.code} {exc.reason}")
        print("  " + exc.read().decode("utf-8"))


def get_log() -> None:
    print("GET /log")
    with urllib.request.urlopen(f"{BASE_URL}/log") as resp:
        entries = json.load(resp)
        print(f"  {resp.status} {resp.reason}  ({len(entries)} entries)")
        print("  " + json.dumps(entries, indent=2).replace("\n", "\n  "))


if __name__ == "__main__":
    sample = sys.argv[1] if len(sys.argv) > 1 else (
        "In today's fast-paced digital landscape, leveraging synergistic "
        "solutions is paramount to unlocking transformative value."
    )
    try:
        post_submit(sample)
        print()
        get_log()
    except urllib.error.URLError as exc:
        print(f"Could not reach the server at {BASE_URL} — is it running?  ({exc})")
