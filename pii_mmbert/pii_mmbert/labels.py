"""Token-level BIES label space built from a list of span types.

Mirrors the OpenAI Privacy Filter layout: index 0 is the background label "O", followed by
B/I/E/S for each span type in the configured order (8 types -> 33 labels for the OPF v2 taxonomy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

BACKGROUND = "O"
BOUNDARY_TAGS = ("B", "I", "E", "S")
IGNORE_INDEX = -100


@dataclass(frozen=True)
class LabelSpace:
    span_types: tuple[str, ...]
    names: tuple[str, ...] = field(init=False)
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if BACKGROUND in self.span_types:
            raise ValueError("span_types must not contain the background label 'O'")
        names = [BACKGROUND] + [f"{t}-{s}" for s in self.span_types for t in BOUNDARY_TAGS]
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "index", {n: i for i, n in enumerate(names)})

    @property
    def num_labels(self) -> int:
        return len(self.names)

    def id(self, tag: str, span_type: str) -> int:
        return self.index[f"{tag}-{span_type}"]

    def tag_of(self, label_id: int) -> str | None:
        """Boundary tag (B/I/E/S) of a label id, or None for background."""
        return None if label_id == 0 else self.names[label_id].split("-", 1)[0]

    def type_of(self, label_id: int) -> str | None:
        """Span type of a label id, or None for background."""
        return None if label_id == 0 else self.names[label_id].split("-", 1)[1]


def labels_to_spans(label_ids: list[int], space: LabelSpace) -> list[tuple[str, int, int]]:
    """Convert a valid BIES label sequence into (type, start_token, end_token_exclusive) spans."""
    spans: list[tuple[str, int, int]] = []
    start: int | None = None
    current: str | None = None
    for i, lid in enumerate(label_ids):
        tag, typ = space.tag_of(lid), space.type_of(lid)
        if tag is None:
            start, current = None, None
        elif tag == "S":
            spans.append((typ, i, i + 1))
            start, current = None, None
        elif tag == "B":
            start, current = i, typ
        elif tag == "E" and start is not None and typ == current:
            spans.append((typ, start, i + 1))
            start, current = None, None
        elif tag == "I" and (start is None or typ != current):
            # Only reachable with unconstrained (argmax) decoding; treat as a fresh span start.
            start, current = i, typ
    return spans
