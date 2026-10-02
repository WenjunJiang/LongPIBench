"""Smoke test for openai/privacy-filter on CPU.

Loads the model once, then times redaction of a few English and Korean
samples plus one longer document. Run from the privacy-filter venv:

    python ../smoke_test.py
"""

import json
import time

from opf import OPF

SAMPLES = [
    "Alice was born on 1990-01-02.",
    "Hi, I'm John Smith. Call me at +1 415-555-0198 or email john.smith@example.com. "
    "My account number is 4111 1111 1111 1111 and I live at 221B Baker Street, London.",
    "안녕하세요, 저는 김민수입니다. 전화번호는 010-1234-5678이고 이메일은 "
    "minsu.kim@example.co.kr 입니다. 주소는 서울특별시 강남구 테헤란로 123입니다.",
    "I was diagnosed with depression last March and my therapist is Dr. Lee.",
    "Use API key sk-test-51HxQ2eZvKYlo2C9a8bT3 for the staging server.",
]

# ~1,000-word document with a few PII items buried in benign text.
FILLER = (
    "The quarterly review covered vendor onboarding, budget planning and the "
    "migration schedule for the internal tooling platform. "
)
LONG_DOC = (
    FILLER * 30
    + "Please send the signed contract to Maria Garcia at maria.garcia@example.org "
    "or call +44 20 7946 0958. "
    + FILLER * 30
)


def main() -> None:
    t0 = time.perf_counter()
    redactor = OPF(device="cpu")
    redactor.redact("warm up")  # first call loads the checkpoint
    print(f"model load + warm-up: {time.perf_counter() - t0:.1f} s\n")

    for text in SAMPLES + [LONG_DOC]:
        t = time.perf_counter()
        result = redactor.redact(text)
        ms = (time.perf_counter() - t) * 1000
        spans = [(s["label"], s["text"]) for s in result.to_dict()["detected_spans"]]
        if text is LONG_DOC:
            label = f"[long doc, {len(text)} chars]"
        else:
            label = text if len(text) <= 70 else text[:67] + "..."
        print(f"{ms:7.0f} ms | {label}")
        print(f"          spans: {json.dumps(spans, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
