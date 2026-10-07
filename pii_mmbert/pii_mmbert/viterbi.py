"""Constrained BIES Viterbi decoder.

Follows the semantics of the OpenAI Privacy Filter decoder (opf/_core/decoding.py, Apache-2.0):
valid starts are O/B/S, valid ends are O/E/S, B/I may only continue to I/E of the same type,
E/S/O may only go to O/B/S, illegal transitions score -1e9, and six transition biases (same key
names as OPF's viterbi_calibration.json) are added to the legal transitions.
"""

from __future__ import annotations

import torch

from .labels import LabelSpace

NEG_INF = -1e9
BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)


def _is_valid(space: LabelSpace, prev: int, nxt: int) -> bool:
    prev_tag, prev_type = space.tag_of(prev), space.type_of(prev)
    next_tag, next_type = space.tag_of(nxt), space.type_of(nxt)
    if prev_tag in (None, "E", "S"):
        return next_tag in (None, "B", "S")
    # prev is B or I: must continue the same span.
    return next_type == prev_type and next_tag in ("I", "E")


def _bias(space: LabelSpace, prev: int, nxt: int, b: dict[str, float]) -> float:
    prev_tag, next_tag = space.tag_of(prev), space.tag_of(nxt)
    if prev_tag is None:
        if next_tag is None:
            return b["transition_bias_background_stay"]
        return b["transition_bias_background_to_start"]  # next is B or S
    if prev_tag in ("B", "I"):
        if next_tag == "I":
            return b["transition_bias_inside_to_continue"]
        return b["transition_bias_inside_to_end"]  # next is E
    if next_tag is None:
        return b["transition_bias_end_to_background"]
    return b["transition_bias_end_to_start"]  # E/S -> B/S


class ViterbiDecoder:
    def __init__(self, space: LabelSpace, biases: dict[str, float] | None = None):
        biases = {k: 0.0 for k in BIAS_KEYS} | dict(biases or {})
        unknown = set(biases) - set(BIAS_KEYS)
        if unknown:
            raise ValueError(f"unknown Viterbi bias keys: {sorted(unknown)}")
        n = space.num_labels
        self.space = space
        self.start = torch.full((n,), NEG_INF)
        self.end = torch.full((n,), NEG_INF)
        self.trans = torch.full((n, n), NEG_INF)
        for i in range(n):
            tag = space.tag_of(i)
            if tag in (None, "B", "S"):
                self.start[i] = 0.0
            if tag in (None, "E", "S"):
                self.end[i] = 0.0
            for j in range(n):
                if _is_valid(space, i, j):
                    self.trans[i, j] = _bias(space, i, j, biases)

    def decode(self, log_probs: torch.Tensor) -> list[int]:
        """Best legal label path for one [seq_len, num_labels] log-probability tensor."""
        if log_probs.ndim != 2:
            raise ValueError("log_probs must have shape [seq_len, num_labels]")
        T = log_probs.shape[0]
        if T == 0:
            return []
        lp = log_probs.float().cpu()
        scores = lp[0] + self.start
        back = torch.empty((T - 1, lp.shape[1]), dtype=torch.long)
        for t in range(1, T):
            best, arg = (scores.unsqueeze(1) + self.trans).max(dim=0)
            scores = best + lp[t]
            back[t - 1] = arg
        last = int((scores + self.end).argmax())
        path = [last]
        for t in range(T - 2, -1, -1):
            last = int(back[t, last])
            path.append(last)
        return path[::-1]
