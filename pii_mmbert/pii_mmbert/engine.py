"""Training loop and document-level evaluation shared by the tune and final stages."""

from __future__ import annotations

import logging
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader

from .data import doc_windows, gold_spans, subset
from .labels import IGNORE_INDEX, LabelSpace, labels_to_spans
from .metrics import SpanScorer
from .model import PiiTagger, class_weights, token_loss
from .viterbi import ViterbiDecoder

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def available_cpus(cgroup_root: str = "/sys/fs/cgroup") -> int:
    """CPU cores this process may use: its CPU affinity, capped by a cgroup CPU quota.

    Containers (e.g. Kubernetes notebooks) often limit CPU with a quota while still showing every host
    core; torch's default thread count then follows the host and oversubscribes the few cores granted,
    which makes many small CPU tensor ops (Viterbi) orders of magnitude slower.
    """
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    root = Path(cgroup_root)
    try:                                                    # cgroup v2: "max 100000" or "400000 100000"
        quota, period = (root / "cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, math.ceil(int(quota) / int(period))))
        return n
    except (OSError, ValueError):
        pass
    try:                                                    # cgroup v1: quota -1 = unlimited
        quota = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            n = min(n, max(1, math.ceil(quota / period)))
    except (OSError, ValueError):
        pass
    return n


def configure_threads(num_threads=None) -> int:
    """Set torch's CPU thread count: train.num_threads if given, else the cores actually available."""
    n = int(num_threads) if num_threads else available_cpus()
    torch.set_num_threads(n)
    return n


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def special_ids(model_name: str) -> tuple[int, int, int]:
    """(bos, eos, pad) ids used to wrap and pad windows."""
    tok = AutoTokenizer.from_pretrained(model_name)
    with_special = tok("x")["input_ids"]
    bare = tok("x", add_special_tokens=False)["input_ids"]
    if len(with_special) != len(bare) + 2:
        raise ValueError(f"expected one leading and one trailing special token for {model_name}")
    return with_special[0], with_special[-1], tok.pad_token_id


def collate(rows: list[dict], pad_id: int):
    """Pad a list of window rows ({"input_ids", "labels" (optional)}) into tensors."""
    n = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), n), dtype=torch.long)
    labels = torch.full((len(rows), n), IGNORE_INDEX, dtype=torch.long)
    for i, r in enumerate(rows):
        k = len(r["input_ids"])
        ids[i, :k] = torch.as_tensor(np.asarray(r["input_ids"], dtype=np.int64))
        mask[i, :k] = 1
        if r.get("labels") is not None:
            labels[i, :k] = torch.as_tensor(np.asarray(r["labels"], dtype=np.int64))
    return ids, mask, labels


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@dataclass
class DocPrediction:
    spans: list = field(default_factory=list)       # [(type, char_start, char_end)]
    scores: list = field(default_factory=list)      # raw span confidence: mean prob of the decoded labels
    token_spans: list = field(default_factory=list) # [(token_start, token_end_exclusive)] per span
    path: list = field(default_factory=list)        # token label ids


@torch.no_grad()
def batch_log_probs(model, input_ids_list, space: LabelSpace, cfg, device, bos_id: int, eos_id: int,
                    pad_id: int) -> list[torch.Tensor]:
    """[n_tokens, num_labels] log-probs for each document; overlapping windows are averaged per token.

    The windows of all documents are sorted by length and run in batches of eval.batch_size, so short
    documents share a forward pass and padding stays small.
    """
    windows = []  # (doc index, token_start, window ids)
    for d, input_ids in enumerate(input_ids_list):
        if len(input_ids):
            for st, ids, _ in doc_windows(input_ids, None, cfg.data.max_length, cfg.data.stride,
                                          bos_id, eos_id, with_labels=False):
                windows.append((d, st, ids))
    acc = [torch.full((len(ids), space.num_labels), -math.inf) for ids in input_ids_list]
    count = [torch.zeros(len(ids)) for ids in input_ids_list]
    windows.sort(key=lambda w: len(w[2]))
    bs = max(1, int(cfg.eval.get("batch_size") or cfg.train.batch_size))
    for i in range(0, len(windows), bs):
        chunk = windows[i: i + bs]
        ids, mask, _ = collate([{"input_ids": w} for _, _, w in chunk], pad_id)
        with _autocast(device, cfg.train.amp):
            logits = model(ids.to(device), mask.to(device))
        lp = F.log_softmax(logits.float(), dim=-1).cpu()
        for (d, st, w), row in zip(chunk, lp):
            m = len(w) - 2
            sl = slice(st, st + m)
            acc[d][sl] = torch.logaddexp(acc[d][sl], row[1: 1 + m])
            count[d][sl] += 1
    return [a - torch.log(c.clamp(min=1)).unsqueeze(1) for a, c in zip(acc, count)]


def doc_log_probs(model, input_ids, space: LabelSpace, cfg, device, bos_id: int, eos_id: int,
                  pad_id: int) -> torch.Tensor:
    """[n_tokens, num_labels] log-probs of one document."""
    return batch_log_probs(model, [input_ids], space, cfg, device, bos_id, eos_id, pad_id)[0]


def iter_log_probs(model, docs, space: LabelSpace, cfg, device, bos_id, eos_id, pad_id, max_docs=None,
                   what: str = "evaluating"):
    """Yield (rows, log_probs) per group of eval.batch_docs documents of a document Dataset."""
    model.eval()
    docs = subset(docs, max_docs, int(cfg.seed))
    group = max(1, int(cfg.eval.get("batch_docs") or 1))
    progress = Progress(what, len(docs), cfg.eval.get("log_every", 0))
    for start in range(0, len(docs), group):
        rows = docs.select(range(start, min(start + group, len(docs)))).to_list()
        yield rows, batch_log_probs(model, [r["input_ids"] for r in rows], space, cfg, device,
                                    bos_id, eos_id, pad_id)
        progress.step(len(rows))


def spans_from_path(path: list[int], log_probs: torch.Tensor, text: str, offset_start, offset_end,
                    space: LabelSpace, trim_whitespace: bool) -> DocPrediction:
    """Character spans and raw confidences for a decoded label path."""
    out = DocPrediction(path=list(path))
    if not path:
        return out
    path_prob = log_probs.gather(1, torch.tensor(path).unsqueeze(1)).squeeze(1).exp()
    for typ, a, b in labels_to_spans(path, space):
        st, en = int(offset_start[a]), int(offset_end[b - 1])
        if trim_whitespace:
            while st < en and text[st].isspace():
                st += 1
            while en > st and text[en - 1].isspace():
                en -= 1
        if en > st:
            out.spans.append((typ, st, en))
            out.scores.append(float(path_prob[a:b].mean()))
            out.token_spans.append((a, b))
    return out


def predict_document(model, text: str, input_ids, offset_start, offset_end, space: LabelSpace,
                     decoder: ViterbiDecoder, cfg, device, bos_id: int, eos_id: int, pad_id: int
                     ) -> DocPrediction:
    """Average overlapping windows' log-probs per token, then one Viterbi pass over the document."""
    lp = doc_log_probs(model, input_ids, space, cfg, device, bos_id, eos_id, pad_id)
    return spans_from_path(decoder.decode(lp), lp, text, offset_start, offset_end, space,
                           cfg.decode.trim_whitespace)


class Progress:
    """Logs "<what>: i/n documents, s/doc, ~min left" every `every` documents (0 = never)."""

    def __init__(self, what: str, total: int, every: int):
        self.what, self.total, self.every = what, total, int(every or 0)
        self.done, self.t0 = 0, time.time()

    def step(self, n: int = 1) -> None:
        before, self.done = self.done, self.done + n
        if self.every and (self.done // self.every > before // self.every or self.done == self.total):
            rate = (time.time() - self.t0) / self.done
            log.info("%s: %d/%d documents, %.3f s/doc, ~%.1f min left", self.what, self.done, self.total,
                     rate, rate * (self.total - self.done) / 60)


def iter_predictions(model, docs, space: LabelSpace, cfg, device, bos_id, eos_id, pad_id, max_docs=None,
                     what: str = "evaluating"):
    """Yield (row, DocPrediction) for a document Dataset, batched over eval.batch_docs documents."""
    decoder = ViterbiDecoder(space, dict(cfg.decode.biases))
    for rows, lps in iter_log_probs(model, docs, space, cfg, device, bos_id, eos_id, pad_id, max_docs, what):
        for row, lp, path in zip(rows, lps, decoder.decode_batch(lps)):
            yield row, spans_from_path(path, lp, row["text"], row["offset_start"], row["offset_end"], space,
                                       cfg.decode.trim_whitespace)


def score_prediction(scorer: SpanScorer, row, pred: DocPrediction, calibrator=None) -> None:
    """Add one document to the scorer; with a calibrator, spans below threshold are dropped."""
    spans, path = pred.spans, pred.path
    if calibrator is not None:
        keep = {i for i, _ in calibrator.keep(pred.spans, pred.scores)}
        spans = [sp for i, sp in enumerate(pred.spans) if i in keep]
        path = list(pred.path)
        for i, (a, b) in enumerate(pred.token_spans):
            if i not in keep:
                path[a:b] = [0] * (b - a)
    scorer.add(gold_spans(row), spans, row["token_labels"], path)


def evaluate(model, docs, space: LabelSpace, cfg, device, bos_id, eos_id, pad_id, what: str = "evaluating") -> dict:
    """Span/token metrics on a document Dataset with the decode biases from cfg."""
    scorer = SpanScorer(space.span_types)
    for row, pred in iter_predictions(model, docs, space, cfg, device, bos_id, eos_id, pad_id,
                                      cfg.eval.max_docs, what=what):
        score_prediction(scorer, row, pred)
    return scorer.result()


def build_model_and_optim(cfg, space: LabelSpace, steps: int, device):
    model = PiiTagger(cfg.model.name, space.num_labels, cfg.model.dropout).to(device)
    no_decay = ("bias", "norm", "LayerNorm")

    def groups(module, lr):
        named = list(module.named_parameters())
        return [
            {"params": [p for n, p in named if not any(k in n for k in no_decay)],
             "weight_decay": cfg.train.weight_decay, "lr": lr},
            {"params": [p for n, p in named if any(k in n for k in no_decay)],
             "weight_decay": 0.0, "lr": lr},
        ]

    params = groups(model.encoder, cfg.train.lr)
    params += groups(model.tag_head, cfg.train.head_lr)
    optim = torch.optim.AdamW([g for g in params if g["params"]])
    sched = get_linear_schedule_with_warmup(optim, int(cfg.train.warmup_ratio * steps), steps)
    return model, optim, sched


def train(cfg, windows, space: LabelSpace, eval_docs=None, eval_prefix="val", report=None):
    """Train on a window Dataset for cfg.train.epochs.

    After each epoch, evaluate on the document Dataset `eval_docs` (if given) and call `report`.
    """
    set_seed(cfg.seed)
    if cfg.train.num_threads:
        torch.set_num_threads(int(cfg.train.num_threads))
    device = pick_device(cfg.train.device)
    bos_id, eos_id, pad_id = special_ids(cfg.model.name)
    bs = int(cfg.train.batch_size)
    loader = DataLoader(
        windows.with_format("numpy", columns=["input_ids", "labels"]),
        batch_size=bs, shuffle=True, drop_last=False, num_workers=int(cfg.train.num_workers),
        generator=torch.Generator().manual_seed(int(cfg.seed)),
        collate_fn=partial(collate, pad_id=pad_id),
    )
    steps_per_epoch = len(loader)
    total = steps_per_epoch * int(cfg.train.epochs)
    model, optim, sched = build_model_and_optim(cfg, space, total, device)
    weights = class_weights(space, cfg.loss.background_weight)
    log.info("training on %d windows, %d steps, device=%s", len(windows), total, device)

    metrics: dict = {}
    for epoch in range(int(cfg.train.epochs)):
        model.train()
        t0, run_loss = time.time(), 0.0
        for step, (ids, mask, labels) in enumerate(loader):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            with _autocast(device, cfg.train.amp):
                logits = model(ids, mask)
            loss = token_loss(logits, labels, weights)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optim.step()
            sched.step()
            run_loss += loss.item()
            if (step + 1) % int(cfg.train.log_every) == 0:
                log.info("epoch %d step %d/%d loss %.4f", epoch + 1, step + 1, steps_per_epoch,
                         run_loss / (step + 1))
        metrics = {
            "epoch": epoch + 1,
            "train_loss": run_loss / steps_per_epoch,
            "epoch_seconds": time.time() - t0,
        }
        if eval_docs is not None:
            res = evaluate(model, eval_docs, space, cfg, device, bos_id, eos_id, pad_id)
            metrics.update({f"{eval_prefix}_{k}": v for k, v in res.items()})
        log.info("epoch %d: %s", epoch + 1, {k: round(v, 4) if isinstance(v, float) else v
                                             for k, v in metrics.items()})
        if report is not None:
            report(metrics)
    return model, metrics
