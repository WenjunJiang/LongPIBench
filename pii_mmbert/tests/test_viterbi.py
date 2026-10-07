import random

import pytest
import torch

from pii_mmbert.labels import LabelSpace, labels_to_spans
from pii_mmbert.viterbi import BIAS_KEYS, ViterbiDecoder

TYPES = ("account_number", "private_address", "private_date", "private_email",
         "private_person", "private_phone", "private_url", "secret")


def _is_legal(path, space):
    prev = None
    for lid in path:
        tag, typ = space.tag_of(lid), space.type_of(lid)
        if prev is None:
            if tag not in (None, "B", "S"):
                return False
        else:
            ptag, ptyp = space.tag_of(prev), space.type_of(prev)
            if ptag in ("B", "I") and not (tag in ("I", "E") and typ == ptyp):
                return False
            if ptag in (None, "E", "S") and tag not in (None, "B", "S"):
                return False
        prev = lid
    return space.tag_of(prev) in (None, "E", "S")


def test_label_space_matches_opf_v2_size():
    assert LabelSpace(TYPES).num_labels == 33


def test_paths_are_legal_and_fill_gaps():
    space = LabelSpace(("injection",))
    lp = torch.full((6, space.num_labels), -6.0)
    for t, best in enumerate(["B-injection", "I-injection", "O", "I-injection", "E-injection", "O"]):
        lp[t, space.index[best]] = -0.3
    lp[2, space.index["I-injection"]] = -1.2
    path = ViterbiDecoder(space).decode(lp)
    assert _is_legal(path, space)
    assert labels_to_spans(path, space) == [("injection", 0, 5)]


def test_matches_opf_decoder_on_random_inputs():
    opf_dec = pytest.importorskip("opf._core.decoding")
    opf_seq = pytest.importorskip("opf._core.sequence_labeling")
    space = LabelSpace(TYPES)
    info = opf_seq.build_label_info(list(space.names))
    rng = random.Random(0)
    for trial in range(30):
        biases = {k: rng.uniform(-3, 3) for k in BIAS_KEYS} if trial % 2 else {}
        ours = ViterbiDecoder(space, biases)
        theirs = opf_dec.ViterbiCRFDecoder(label_info=info, **biases)
        lp = torch.log_softmax(torch.randn(rng.randint(1, 40), space.num_labels) * 3, dim=-1)
        assert ours.decode(lp) == theirs.decode(lp)
