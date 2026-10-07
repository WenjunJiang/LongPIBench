from pathlib import Path

import pytest
from omegaconf import OmegaConf

from pii_mmbert.data import Document, LabelMapper, make_windows, token_bies_labels
from pii_mmbert.labels import IGNORE_INDEX, LabelSpace, labels_to_spans

CONF = Path(__file__).resolve().parents[1] / "conf" / "label_map" / "nemotron_to_opf_v2.yaml"


@pytest.fixture(scope="module")
def mapper():
    return LabelMapper(OmegaConf.load(CONF))


def _span(text, sub, label, start=0):
    i = text.index(sub, start)
    return {"label": label, "start": i, "end": i + len(sub)}


def test_first_and_last_name_merge(mapper):
    t = "I, Ethan Connelly, apply."
    spans, _ = mapper(t, [_span(t, "Ethan", "first_name"), _span(t, "Connelly", "last_name")])
    assert spans == [("private_person", 3, 17)]


def test_address_components_attach_only_next_to_street(mapper):
    t = "Lives at 87 Main St, Springfield, IL 62704. Born in Canada."
    src = [_span(t, "87 Main St", "street_address"), _span(t, "Springfield", "city"),
           _span(t, "IL", "state"), _span(t, "62704", "postcode"), _span(t, "Canada", "country")]
    spans, _ = mapper(t, src)
    assert spans == [("private_address", t.index("87"), t.index("62704") + 5)]


def test_background_labels_and_sensitive(mapper):
    t = "Dana, a nurse, is Buddhist."
    spans, sens = mapper(t, [_span(t, "Dana", "first_name"), _span(t, "nurse", "occupation"),
                             _span(t, "Buddhist", "religious_belief")])
    assert spans == [("private_person", 0, 4)]
    assert sens == [(t.index("Buddhist"), t.index("Buddhist") + 8)]


def test_unknown_label_fails(mapper):
    with pytest.raises(ValueError, match="missing from label_map"):
        mapper("x", [{"label": "not_a_label", "start": 0, "end": 1}])


def test_bies_round_trip():
    space = LabelSpace(("private_person", "private_phone"))
    offsets = [(0, 2), (3, 8), (8, 9), (10, 13), (13, 14), (14, 17)]  # "Hi Ethan , 010 - 123"
    spans = [("private_person", 3, 8), ("private_phone", 10, 17)]
    labels = token_bies_labels(offsets, spans, space)
    assert [space.names[i] for i in labels.tolist()] == ["O", "S-private_person", "O",
                                                "B-private_phone", "I-private_phone", "E-private_phone"]
    assert labels_to_spans(labels.tolist(), space) == [("private_person", 1, 2), ("private_phone", 3, 6)]


def test_windows_cover_every_token_with_overlap():
    n = 50
    doc = Document("u", "x" * n, [], [], list(range(10, 10 + n)), [(i, i + 1) for i in range(n)], [0] * n)
    wins = make_windows(doc, 0, max_length=22, stride=5, bos_id=2, eos_id=1)
    covered = set()
    for w in wins:
        assert w.input_ids[0] == 2 and w.input_ids[-1] == 1 and w.input_ids.dtype.kind == "i"
        assert w.labels[0] == IGNORE_INDEX and w.labels[-1] == IGNORE_INDEX
        covered |= set(range(w.token_start, w.token_start + len(w.input_ids) - 2))
    assert covered == set(range(n))
