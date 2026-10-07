"""Nemotron-PII loading, label mapping to the OPF taxonomy, tokenization and windowing."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from .labels import IGNORE_INDEX, LabelSpace

log = logging.getLogger(__name__)


@dataclass
class Document:
    """One text with gold spans; per-token data is stored as compact numpy arrays."""
    uid: str
    text: str
    spans: list[tuple[str, int, int]]          # gold (type, char_start, char_end) in target taxonomy
    sensitive_spans: list[tuple[int, int]]     # char ranges of sensitive source labels
    token_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    offsets: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))
    token_labels: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int16))

    def __post_init__(self) -> None:
        self.token_ids = np.asarray(self.token_ids, dtype=np.int32)
        self.offsets = np.asarray(self.offsets, dtype=np.int32).reshape(-1, 2)
        self.token_labels = np.asarray(self.token_labels, dtype=np.int16)


@dataclass
class Window:
    doc_index: int
    token_start: int                           # index into Document.token_ids
    input_ids: np.ndarray                      # with <bos>/<eos>
    labels: np.ndarray | None                  # IGNORE_INDEX on special tokens; None for inference
    sensitive: float


# ----------------------------------------------------------------------------- label mapping

class LabelMapper:
    """Maps Nemotron source spans to target-type char spans, following the label_map config."""

    def __init__(self, label_map_cfg):
        self.target_types = tuple(label_map_cfg.target_types)
        self.mapping = dict(label_map_cfg.mapping)
        self.attach = dict(label_map_cfg.attach or {})
        overlap = set(self.mapping) & set(self.attach)
        if overlap:
            raise ValueError(f"labels listed in both mapping and attach: {sorted(overlap)}")
        for src, tgt in list(self.mapping.items()) + list(self.attach.items()):
            if tgt is not None and tgt not in self.target_types:
                raise ValueError(f"{src} maps to unknown target type {tgt!r}")
        merge = label_map_cfg.merge
        self.merge_enabled = bool(merge.enabled)
        self.max_gap = int(merge.max_gap_chars)
        self.gap_re = re.compile(merge.gap_pattern)
        self.sensitive = set(label_map_cfg.sensitive_source_labels or [])

    def known(self, label: str) -> bool:
        return label in self.mapping or label in self.attach

    def __call__(self, text: str, source_spans: list[dict]):
        unknown = sorted({s["label"] for s in source_spans if not self.known(s["label"])})
        if unknown:
            raise ValueError(f"source labels missing from label_map: {unknown}")
        sensitive = [(s["start"], s["end"]) for s in source_spans if s["label"] in self.sensitive]

        # (type, start, end, is_core)
        cand = []
        for s in source_spans:
            if s["label"] in self.mapping:
                tgt = self.mapping[s["label"]]
                if tgt is not None:
                    cand.append((tgt, s["start"], s["end"], True))
            else:
                cand.append((self.attach[s["label"]], s["start"], s["end"], False))
        cand.sort(key=lambda c: (c[1], -c[2]))

        groups: list[list] = []  # [type, start, end, has_core]
        for typ, st, en, core in cand:
            if groups:
                g = groups[-1]
                gap = text[g[2]:st] if st >= g[2] else ""
                same = g[0] == typ
                if st < g[2] and same:            # overlaps a span of the same type: extend
                    g[2], g[3] = max(g[2], en), g[3] or core
                    continue
                if st < g[2]:                     # overlaps a different type: keep the earlier one
                    continue
                if (self.merge_enabled and same and len(gap) <= self.max_gap
                        and self.gap_re.match(gap)):
                    g[2], g[3] = en, g[3] or core
                    continue
            groups.append([typ, st, en, core])
        spans = [(t, s, e) for t, s, e, core in groups if core]
        return spans, sensitive


# ----------------------------------------------------------------------------- tokenization

def token_bies_labels(offsets, spans, space: LabelSpace) -> np.ndarray:
    """BIES label per token: a token belongs to a span if its char range overlaps the span."""
    offsets = np.asarray(offsets, dtype=np.int32).reshape(-1, 2)
    a, b = offsets[:, 0], offsets[:, 1]
    labels = np.zeros(len(offsets), dtype=np.int16)
    taken = np.zeros(len(offsets), dtype=bool)
    for typ, s, e in spans:
        idx = np.flatnonzero((b > a) & (a < e) & (b > s) & ~taken)
        if idx.size == 0:
            continue
        if idx.size == 1:
            labels[idx[0]] = space.id("S", typ)
        else:
            labels[idx[0]] = space.id("B", typ)
            labels[idx[1:-1]] = space.id("I", typ)
            labels[idx[-1]] = space.id("E", typ)
        taken[idx] = True
    return labels


def make_windows(doc: Document, doc_index: int, max_length: int, stride: int,
                 bos_id: int, eos_id: int, with_labels: bool = True) -> list[Window]:
    body = max_length - 2
    if body <= stride:
        raise ValueError("data.max_length - 2 must be larger than data.stride")
    n = len(doc.token_ids)
    starts = [0] if n <= body else list(range(0, n - stride, body - stride))
    if starts[-1] + body < n:
        starts.append(n - body)
    windows = []
    for st in starts:
        en = min(st + body, n)
        ids = np.concatenate([[bos_id], doc.token_ids[st:en], [eos_id]]).astype(np.int32)
        labs = (np.concatenate([[IGNORE_INDEX], doc.token_labels[st:en], [IGNORE_INDEX]]).astype(np.int16)
                if with_labels else None)
        c0 = int(doc.offsets[st, 0]) if en > st else 0
        c1 = int(doc.offsets[en - 1, 1]) if en > st else 0
        sens = float(any(a < c1 and b > c0 for a, b in doc.sensitive_spans))
        windows.append(Window(doc_index, st, ids, labs, sens))
    return windows


# ----------------------------------------------------------------------------- loading

def _read_split(path: str, locales, max_rows) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["uid", "locale", "text", "spans"])
    df = df[df.locale.isin(list(locales))]
    if max_rows is not None:
        df = df.iloc[: int(max_rows)]
    return df.reset_index(drop=True)


def _uid_is_val(uid: str, seed: int, fraction: float) -> bool:
    h = zlib.crc32(f"{seed}:{uid}".encode()) & 0xFFFFFFFF
    return h / 2**32 < fraction


def _build_docs(df: pd.DataFrame, mapper: LabelMapper, tokenizer, space: LabelSpace,
                batch_size: int = 1000) -> list[Document]:
    docs = []
    texts, uids, raws = list(df.text), list(df.uid), list(df.spans)
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i: i + batch_size], add_special_tokens=False, return_offsets_mapping=True)
        for j, (ids, offs) in enumerate(zip(enc["input_ids"], enc["offset_mapping"])):
            text, raw = texts[i + j], raws[i + j]
            source = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
            # Offsets are authoritative; the span "text" field sometimes differs in case only.
            source = [{"label": s["label"], "start": int(s["start"]), "end": int(s["end"])} for s in source]
            spans, sensitive = mapper(text, source)
            doc = Document(uids[i + j], text, spans, sensitive, ids, offs)
            doc.token_labels = token_bies_labels(doc.offsets, spans, space)
            docs.append(doc)
        if (i // batch_size) % 20 == 0:
            log.info("  tokenized %d / %d", min(i + batch_size, len(texts)), len(texts))
    return docs


def _cache_key(cfg) -> str:
    from omegaconf import OmegaConf
    payload = {
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "label_map": OmegaConf.to_container(cfg.label_map, resolve=True),
        "model": cfg.model.name,
        "seed": cfg.seed,
        "version": 1,
    }
    for k in ("max_length", "stride"):   # windows are built on the fly; they do not change the cache
        payload["data"].pop(k, None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_or_build(cfg, resolve_path) -> dict:
    """Returns {"train": [...], "val": [...], "test": [...], "space": LabelSpace, "stats": {...}}."""
    cache = Path(resolve_path(cfg.paths.cache_dir)) / f"nemotron_{_cache_key(cfg)}.pt"
    space = LabelSpace(tuple(cfg.label_map.target_types))
    if cache.exists():
        log.info("loading preprocessed data from %s", cache)
        blob = torch.load(cache, weights_only=False)
        blob["space"] = space
        return blob

    mapper = LabelMapper(cfg.label_map)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    train_df = _read_split(resolve_path(cfg.data.train_file), cfg.data.locales, cfg.data.max_train_docs)
    test_df = _read_split(resolve_path(cfg.data.test_file), cfg.data.locales, cfg.data.max_test_docs)
    is_val = [_uid_is_val(u, cfg.seed, cfg.data.val_fraction) for u in train_df.uid]
    log.info("tokenizing %d train rows and %d test rows", len(train_df), len(test_df))
    train_docs = _build_docs(train_df, mapper, tokenizer, space)
    test_docs = _build_docs(test_df, mapper, tokenizer, space)
    blob = {
        "train": [d for d, v in zip(train_docs, is_val) if not v],
        "val": [d for d, v in zip(train_docs, is_val) if v],
        "test": test_docs,
    }
    blob["stats"] = _stats(blob, space)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, cache)           # atomic: an interrupted run never leaves a truncated cache
    log.info("saved preprocessed data to %s", cache)
    blob["space"] = space
    return blob


def _stats(blob, space: LabelSpace) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        docs = blob[split]
        by_type = {t: 0 for t in space.span_types}
        for d in docs:
            for t, _, _ in d.spans:
                by_type[t] += 1
        n_tok = int(sum(len(d.token_ids) for d in docs))
        n_pii = int(sum(np.count_nonzero(d.token_labels) for d in docs))
        out[split] = {
            "docs": len(docs),
            "uids": len({d.uid for d in docs}),
            "tokens": n_tok,
            "pii_token_fraction": n_pii / n_tok if n_tok else 0.0,
            "docs_with_sensitive": sum(1 for d in docs if d.sensitive_spans),
            "spans_by_type": by_type,
        }
    return out
