"""Training loop and document-level evaluation shared by the tune and final stages."""

from __future__ import annotations

import logging
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader

from .data import doc_windows, gold_spans
from .labels import IGNORE_INDEX, LabelSpace, labels_to_spans
from .metrics import SpanScorer
from .model import PiiTagger, class_weights, pii_trace_loss
from .viterbi import ViterbiDecoder

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    """Pad a list of window rows ({"input_ids", "labels" (optional), "sensitive"}) into tensors."""
    n = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), n), dtype=torch.long)
    labels = torch.full((len(rows), n), IGNORE_INDEX, dtype=torch.long)
    sens = torch.tensor([float(r.get("sensitive", 0.0)) for r in rows])
    for i, r in enumerate(rows):
        k = len(r["input_ids"])
        ids[i, :k] = torch.as_tensor(np.asarray(r["input_ids"], dtype=np.int64))
        mask[i, :k] = 1
        if r.get("labels") is not None:
            labels[i, :k] = torch.as_tensor(np.asarray(r["labels"], dtype=np.int64))
    return ids, mask, labels, sens


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
    sens_logit: float = float("-inf")               # max sensitivity logit over the document's windows


@torch.no_grad()
def predict_document(model, text: str, input_ids, offset_start, offset_end, space: LabelSpace,
                     decoder: ViterbiDecoder, cfg, device, bos_id: int, eos_id: int, pad_id: int
                     ) -> DocPrediction:
    """Average overlapping windows' log-probs per token, then one Viterbi pass over the document."""
    n = len(input_ids)
    if n == 0:
        return DocPrediction()
    windows = [{"token_start": st, "input_ids": ids}
               for st, ids, _, _ in doc_windows(input_ids, offset_start, offset_end, None, [], [],
                                                cfg.data.max_length, cfg.data.stride, bos_id, eos_id,
                                                with_labels=False)]
    acc = torch.full((n, space.num_labels), -math.inf)
    count = torch.zeros(n)
    sens_logit = float("-inf")
    bs = max(1, int(cfg.train.batch_size))
    for i in range(0, len(windows), bs):
        chunk = windows[i: i + bs]
        ids, mask, _, _ = collate(chunk, pad_id)
        with _autocast(device, cfg.train.amp):
            logits, sens = model(ids.to(device), mask.to(device))
        sens_logit = max(sens_logit, float(sens.float().max()))
        lp = F.log_softmax(logits.float(), dim=-1).cpu()
        for w, row in zip(chunk, lp):
            m = len(w["input_ids"]) - 2
            sl = slice(w["token_start"], w["token_start"] + m)
            acc[sl] = torch.logaddexp(acc[sl], row[1: 1 + m])
            count[sl] += 1
    avg = acc - torch.log(count.clamp(min=1)).unsqueeze(1)
    path = decoder.decode(avg)
    path_prob = avg.gather(1, torch.tensor(path).unsqueeze(1)).squeeze(1).exp()
    out = DocPrediction(path=path, sens_logit=sens_logit)
    for typ, a, b in labels_to_spans(path, space):
        st, en = int(offset_start[a]), int(offset_end[b - 1])
        if cfg.decode.trim_whitespace:
            while st < en and text[st].isspace():
                st += 1
            while en > st and text[en - 1].isspace():
                en -= 1
        if en > st:
            out.spans.append((typ, st, en))
            out.scores.append(float(path_prob[a:b].mean()))
            out.token_spans.append((a, b))
    return out


def iter_predictions(model, docs, space: LabelSpace, cfg, device, bos_id, eos_id, pad_id, max_docs=None):
    """Yield (row, DocPrediction) for a document Dataset (memory-mapped; rows read one at a time)."""
    model.eval()
    decoder = ViterbiDecoder(space, dict(cfg.decode.biases))
    if max_docs is not None:
        docs = docs.select(range(min(int(max_docs), len(docs))))
    for row in docs:
        yield row, predict_document(model, row["text"], row["input_ids"], row["offset_start"],
                                    row["offset_end"], space, decoder, cfg, device, bos_id, eos_id, pad_id)


def evaluate(model, docs, space: LabelSpace, cfg, device, bos_id, eos_id, pad_id, calibrator=None) -> dict:
    """Span/token metrics on a document Dataset; with a calibrator, spans below threshold are dropped."""
    scorer = SpanScorer(space.span_types)
    for row, pred in iter_predictions(model, docs, space, cfg, device, bos_id, eos_id, pad_id,
                                      cfg.eval.max_docs):
        spans, path = pred.spans, pred.path
        if calibrator is not None:
            keep = {i for i, _ in calibrator.keep(pred.spans, pred.scores)}
            spans = [sp for i, sp in enumerate(pred.spans) if i in keep]
            path = list(pred.path)
            for i, (a, b) in enumerate(pred.token_spans):
                if i not in keep:
                    path[a:b] = [0] * (b - a)
        scorer.add(gold_spans(row), spans, row["token_labels"], path)
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
    params += groups(model.tag_head, cfg.train.head_lr) + groups(model.sens_head, cfg.train.head_lr)
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
        windows.with_format("numpy", columns=["input_ids", "labels", "sensitive"]),
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
        t0, run_loss, run_tag, run_sens = time.time(), 0.0, 0.0, 0.0
        for step, (ids, mask, labels, sens) in enumerate(loader):
            ids, mask, labels, sens = ids.to(device), mask.to(device), labels.to(device), sens.to(device)
            with _autocast(device, cfg.train.amp):
                logits, sens_logit = model(ids, mask)
            loss, tag_l, sens_l = pii_trace_loss(logits, sens_logit, labels, sens, weights,
                                                 cfg.loss.lambda_tok, cfg.loss.lambda_sens)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optim.step()
            sched.step()
            run_loss += loss.item(); run_tag += tag_l.item(); run_sens += sens_l.item()
            if (step + 1) % int(cfg.train.log_every) == 0:
                log.info("epoch %d step %d/%d loss %.4f", epoch + 1, step + 1, steps_per_epoch,
                         run_loss / (step + 1))
        metrics = {
            "epoch": epoch + 1,
            "train_loss": run_loss / steps_per_epoch,
            "train_tag_loss": run_tag / steps_per_epoch,
            "train_sens_loss": run_sens / steps_per_epoch,
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
