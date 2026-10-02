"""Check that GLiNER runs on CPU and finds PII in one long sentence.

Run from the GLiNER venv:

    python ../pii_test.py
"""

import time
import warnings

from gliner import GLiNER

warnings.filterwarnings("ignore")

MODELS = [
    "urchade/gliner_multi_pii-v1",
    "knowledgator/gliner-pii-base-v1.0",
]

LABELS = [
    "person",
    "phone number",
    "email",
    "address",
    "date of birth",
    "credit card number",
    "passport number",
    "bank account number",
    "medical condition",
    "organization",
]

# One long sentence with PII of several types spread across it.
LONG_SENTENCE = (
    "Yesterday afternoon, after the quarterly budget meeting ran over by almost an hour, "
    "our new client Sarah Johnson, who was born on 14 March 1987 and currently lives at "
    "742 Evergreen Terrace, Springfield, called from +1 (555) 123-4567 to ask whether we "
    "could send the revised contract to sarah.johnson@example.com instead of her work "
    "address, mentioned that her passport number X12345678 would be needed for the visa "
    "paperwork, asked us to refund the deposit to her bank account 0123456789 at First "
    "National Bank rather than to her credit card 4532 0151 1283 0366, and added, almost "
    "in passing, that she had recently been diagnosed with type 2 diabetes and might need "
    "to reschedule the site visit with Dr. Michael Chen next week."
)

KOREAN_SENTENCE = (
    "어제 회의가 끝난 뒤 고객 김민수 님이 010-1234-5678로 전화해서 계약서를 "
    "minsu.kim@example.co.kr로 보내 달라고 했고, 주소는 서울특별시 강남구 테헤란로 123이며 "
    "최근 당뇨 진단을 받았다고 말했습니다."
)


def run(model_name: str) -> None:
    t0 = time.perf_counter()
    model = GLiNER.from_pretrained(model_name, map_location="cpu")
    print(f"=== {model_name} (load {time.perf_counter() - t0:.1f} s)")
    for name, text in [("English long sentence", LONG_SENTENCE), ("Korean sentence", KOREAN_SENTENCE)]:
        t = time.perf_counter()
        entities = model.predict_entities(text, LABELS, threshold=0.4)
        ms = (time.perf_counter() - t) * 1000
        print(f"--- {name}: {len(text)} chars, {len(entities)} entities, {ms:.0f} ms")
        for e in entities:
            print(f"    {e['label']:<20} {e['score']:.2f}  {e['text']}")
    print()


if __name__ == "__main__":
    for m in MODELS:
        run(m)
