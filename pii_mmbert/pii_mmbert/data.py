"""Nemotron-PII loading, label mapping to the OPF taxonomy, tokenization and windowing.

All preprocessed data is stored as Hugging Face `datasets` (Arrow files on disk). Datasets are
memory-mapped when loaded, so training, evaluation and Ray Tune trials read rows on demand instead of
holding the corpus in Python objects.

Document dataset columns (one row per Nemotron row):
    uid, text,
    span_type / span_start / span_end      gold spans in the target taxonomy (char offsets)
    sens_start / sens_end                  char ranges of sensitive source labels
    input_ids, offset_start, offset_end    tokenizer output without special tokens
    token_labels                           BIES label id per token

Window dataset columns (one row per training window):
    input_ids (with <bos>/<eos>), labels (IGNORE_INDEX on specials), sensitive, doc_index, token_start
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import shutil
import zlib
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer

from .labels import IGNORE_INDEX, LabelSpace

log = logging.getLogger(__name__)

DOC_FEATURES = Features({
    "uid": Value("string"),
    "text": Value("string"),
    "span_type": Sequence(Value("string")),
    "span_start": Sequence(Value("int32")),
    "span_end": Sequence(Value("int32")),
    "sens_start": Sequence(Value("int32")),
    "sens_end": Sequence(Value("int32")),
    "input_ids": Sequence(Value("int32")),
    "offset_start": Sequence(Value("int32")),
    "offset_end": Sequence(Value("int32")),
    "token_labels": Sequence(Value("int16")),
})

WINDOW_FEATURES = Features({
    "input_ids": Sequence(Value("int32")),
    "labels": Sequence(Value("int16")),
    "sensitive": Value("float32"),
    "doc_index": Value("int32"),
    "token_start": Value("int32"),
})

SPLITS = ("train", "val", "test")


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

        cand = []  # (type, start, end, is_core)
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


# ----------------------------------------------------------------------------- tokens and windows

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


def window_bounds(n_tokens: int, max_length: int, stride: int) -> list[tuple[int, int]]:
    """[start, end) token ranges covering a document; consecutive windows share `stride` tokens."""
    body = max_length - 2
    if body <= stride:
        raise ValueError("data.max_length - 2 must be larger than data.stride")
    if n_tokens <= body:
        return [(0, n_tokens)]
    starts = list(range(0, n_tokens - stride, body - stride))
    if starts[-1] + body < n_tokens:
        starts.append(n_tokens - body)
    return [(s, min(s + body, n_tokens)) for s in starts]


def doc_windows(input_ids, offset_start, offset_end, token_labels, sens_start, sens_end,
                max_length: int, stride: int, bos_id: int, eos_id: int, with_labels: bool = True):
    """Yield (token_start, input_ids, labels or None, sensitive) for one document."""
    ids = np.asarray(input_ids, dtype=np.int32)
    labs = np.asarray(token_labels, dtype=np.int16) if with_labels else None
    for st, en in window_bounds(len(ids), max_length, stride):
        win_ids = np.concatenate([[bos_id], ids[st:en], [eos_id]]).astype(np.int32)
        win_labs = (np.concatenate([[IGNORE_INDEX], labs[st:en], [IGNORE_INDEX]]).astype(np.int16)
                    if with_labels else None)
        c0 = int(offset_start[st]) if en > st else 0
        c1 = int(offset_end[en - 1]) if en > st else 0
        sens = float(any(a < c1 and b > c0 for a, b in zip(sens_start, sens_end)))
        yield st, win_ids, win_labs, sens


def _windows_batch(batch, indices, max_length, stride, bos_id, eos_id):
    out = {k: [] for k in WINDOW_FEATURES}
    for j, idx in enumerate(indices):
        for st, ids, labs, sens in doc_windows(
                batch["input_ids"][j], batch["offset_start"][j], batch["offset_end"][j],
                batch["token_labels"][j], batch["sens_start"][j], batch["sens_end"][j],
                max_length, stride, bos_id, eos_id):
            out["input_ids"].append(ids)
            out["labels"].append(labs)
            out["sensitive"].append(sens)
            out["doc_index"].append(idx)
            out["token_start"].append(st)
    return out


def _encode_batch(batch, mapper: LabelMapper, tokenizer, space: LabelSpace):
    out = {k: [] for k in DOC_FEATURES}
    enc = tokenizer(batch["text"], add_special_tokens=False, return_offsets_mapping=True)
    for j, text in enumerate(batch["text"]):
        raw = batch["spans"][j]
        source = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
        # Offsets are authoritative; the span "text" field sometimes differs in case only.
        source = [{"label": s["label"], "start": int(s["start"]), "end": int(s["end"])} for s in source]
        spans, sensitive = mapper(text, source)
        offsets = np.asarray(enc["offset_mapping"][j], dtype=np.int32).reshape(-1, 2)
        out["uid"].append(batch["uid"][j])
        out["text"].append(text)
        out["span_type"].append([t for t, _, _ in spans])
        out["span_start"].append([s for _, s, _ in spans])
        out["span_end"].append([e for _, _, e in spans])
        out["sens_start"].append([s for s, _ in sensitive])
        out["sens_end"].append([e for _, e in sensitive])
        out["input_ids"].append(np.asarray(enc["input_ids"][j], dtype=np.int32))
        out["offset_start"].append(offsets[:, 0])
        out["offset_end"].append(offsets[:, 1])
        out["token_labels"].append(token_bies_labels(offsets, spans, space))
    return out


def gold_spans(row) -> list[tuple[str, int, int]]:
    return list(zip(row["span_type"], row["span_start"], row["span_end"]))


# ----------------------------------------------------------------------------- building and caching

def _cache_key(cfg) -> str:
    from omegaconf import OmegaConf
    payload = {
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "label_map": OmegaConf.to_container(cfg.label_map, resolve=True),
        "model": cfg.model.name,
        "seed": cfg.seed,
        "version": 2,
    }
    for k in ("max_length", "stride", "num_proc"):   # do not change the document datasets
        payload["data"].pop(k, None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _save_atomic(ds: Dataset, target: Path) -> None:
    """Write a dataset so an interrupted run never leaves a half-written directory behind."""
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    ds.save_to_disk(str(tmp))
    shutil.rmtree(target, ignore_errors=True)
    os.replace(tmp, target)


def _read_split(path: str, locales, max_rows, num_proc) -> Dataset:
    ds = Dataset.from_parquet(path, columns=["uid", "locale", "text", "spans"])
    keep = set(locales)
    ds = ds.filter(lambda b: [x in keep for x in b["locale"]], batched=True, num_proc=num_proc)
    if max_rows is not None:
        ds = ds.select(range(min(int(max_rows), len(ds))))
    return ds


def cache_root(cfg, resolve_path) -> Path:
    return Path(resolve_path(cfg.paths.cache_dir)) / f"nemotron_{_cache_key(cfg)}"


def load_or_build(cfg, resolve_path) -> dict:
    """Build (once) and memory-map the train / val / test document datasets.

    Returns {"train": Dataset, "val": Dataset, "test": Dataset, "space": LabelSpace,
             "stats": dict, "root": Path}.
    """
    root = cache_root(cfg, resolve_path)
    space = LabelSpace(tuple(cfg.label_map.target_types))
    done = root / "stats.json"
    if not done.exists():
        _build(cfg, resolve_path, root, space)
    else:
        log.info("loading preprocessed datasets from %s", root)
    out = {split: load_from_disk(str(root / split)) for split in SPLITS}
    out.update(space=space, stats=json.loads(done.read_text()), root=root)
    return out


def _build(cfg, resolve_path, root: Path, space: LabelSpace) -> None:
    num_proc = cfg.data.num_proc
    mapper = LabelMapper(cfg.label_map)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    root.mkdir(parents=True, exist_ok=True)
    raw = {
        "train": _read_split(resolve_path(cfg.data.train_file), cfg.data.locales, cfg.data.max_train_docs, num_proc),
        "test": _read_split(resolve_path(cfg.data.test_file), cfg.data.locales, cfg.data.max_test_docs, num_proc),
    }
    encoded = {}
    for name, ds in raw.items():
        log.info("encoding %d %s rows", len(ds), name)
        encoded[name] = ds.map(_encode_batch, batched=True, batch_size=500, num_proc=num_proc,
                               remove_columns=ds.column_names, features=DOC_FEATURES,
                               fn_kwargs={"mapper": mapper, "tokenizer": tokenizer, "space": space},
                               desc=f"encode {name}")
    seed, frac = cfg.seed, cfg.data.val_fraction

    def is_val(batch):
        return [(zlib.crc32(f"{seed}:{u}".encode()) & 0xFFFFFFFF) / 2**32 < frac for u in batch["uid"]]

    flags = encoded["train"].map(lambda b: {"_val": is_val(b)}, batched=True, num_proc=num_proc,
                                 desc="split")
    splits = {
        "train": flags.filter(lambda b: [not v for v in b["_val"]], batched=True).remove_columns("_val"),
        "val": flags.filter(lambda b: b["_val"], batched=True).remove_columns("_val"),
        "test": encoded["test"],
    }
    for name, ds in splits.items():
        _save_atomic(ds, root / name)
    stats = {name: _split_stats(load_from_disk(str(root / name)), space) for name in SPLITS}
    (root / "stats.json").write_text(json.dumps(stats, indent=2))  # written last: marks the cache complete
    log.info("saved preprocessed datasets to %s", root)


def _split_stats(ds: Dataset, space: LabelSpace) -> dict:
    by_type = {t: 0 for t in space.span_types}
    n_tok = n_pii = n_sens = 0
    uids = set()
    for batch in ds.select_columns(["uid", "span_type", "token_labels", "sens_start"]).iter(batch_size=2000):
        uids.update(batch["uid"])
        for types in batch["span_type"]:
            for t in types:
                by_type[t] += 1
        for labs in batch["token_labels"]:
            arr = np.asarray(labs)
            n_tok += arr.size
            n_pii += int(np.count_nonzero(arr))
        n_sens += sum(1 for s in batch["sens_start"] if len(s))
    return {"docs": len(ds), "uids": len(uids), "tokens": n_tok,
            "pii_token_fraction": n_pii / n_tok if n_tok else 0.0,
            "docs_with_sensitive": n_sens, "spans_by_type": by_type}


def windows_dir(cfg, data: dict, split: str) -> Path:
    return data["root"] / f"windows_{split}_L{cfg.data.max_length}_S{cfg.data.stride}"


def load_windows(cfg, data: dict, splits, bos_id: int, eos_id: int) -> Dataset:
    """Training windows for the given splits, built once per (max_length, stride) and memory-mapped."""
    parts = []
    for split in splits:
        target = windows_dir(cfg, data, split)
        if not target.exists():
            docs = data[split]
            win = docs.map(_windows_batch, batched=True, batch_size=500, with_indices=True,
                           num_proc=cfg.data.num_proc, remove_columns=docs.column_names,
                           features=WINDOW_FEATURES, desc=f"windows {split}",
                           fn_kwargs={"max_length": int(cfg.data.max_length), "stride": int(cfg.data.stride),
                                      "bos_id": bos_id, "eos_id": eos_id})
            _save_atomic(win, target)
        parts.append(load_from_disk(str(target)))
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)
