from pathlib import Path

import pytest
from omegaconf import OmegaConf

from pii_mmbert.data import (DOC_FEATURES, LabelMapper, _encode_batch, _windows_batch, doc_windows,
                             token_bies_labels, window_bounds)
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
    ids, starts, ends, labs = list(range(10, 10 + n)), list(range(n)), list(range(1, n + 1)), [0] * n
    covered = set()
    for st, w_ids, w_labs, _ in doc_windows(ids, starts, ends, labs, [], [], max_length=22, stride=5,
                                            bos_id=2, eos_id=1):
        assert w_ids[0] == 2 and w_ids[-1] == 1 and w_ids.dtype.kind == "i"
        assert w_labs[0] == IGNORE_INDEX and w_labs[-1] == IGNORE_INDEX
        covered |= set(range(st, st + len(w_ids) - 2))
    assert covered == set(range(n))
    assert window_bounds(10, 22, 5) == [(0, 10)]


def test_encode_and_window_with_datasets_map(mapper):
    """Runs the real datasets.map pipeline on two tiny rows with the mmBERT tokenizer."""
    datasets = pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    except Exception as exc:  # offline machines
        pytest.skip(f"tokenizer unavailable: {exc}")
    t1 = "I, Ethan Connelly, live at 87 Main St, Springfield. I am Buddhist."
    t2 = "No personal data here."
    rows = {"uid": ["a", "b"], "locale": ["us", "us"], "text": [t1, t2],
            "spans": [str([_span(t1, "Ethan", "first_name"), _span(t1, "Connelly", "last_name"),
                           _span(t1, "87 Main St", "street_address"), _span(t1, "Springfield", "city"),
                           _span(t1, "Buddhist", "religious_belief")]), "[]"]}
    space = LabelSpace(tuple(OmegaConf.load(CONF).target_types))
    ds = datasets.Dataset.from_dict(rows).map(
        _encode_batch, batched=True, remove_columns=list(rows), features=DOC_FEATURES,
        fn_kwargs={"mapper": mapper, "tokenizer": tok, "space": space})
    r = ds[0]
    assert list(zip(r["span_type"], r["span_start"], r["span_end"])) == [
        ("private_person", t1.index("Ethan"), t1.index(",", 4)),
        ("private_address", t1.index("87"), t1.index("Springfield") + len("Springfield"))]
    labels = [space.names[i] for i in r["token_labels"] if i]
    assert labels[0] == "B-private_person" and labels[-1] == "E-private_address"
    assert ds[1]["span_type"] == [] and not any(ds[1]["token_labels"])
    win = ds.map(_windows_batch, batched=True, with_indices=True, remove_columns=ds.column_names,
                 fn_kwargs={"max_length": 16, "stride": 4, "bos_id": 2, "eos_id": 1})
    assert len(win) > 2 and set(win["doc_index"]) == {0, 1}
    assert max(win["sensitive"]) == 1.0 and win["sensitive"][-1] == 0.0
