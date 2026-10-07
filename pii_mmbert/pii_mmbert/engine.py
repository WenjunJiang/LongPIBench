"""Training loop and document-level evaluation shared by the tune and final stages."""

from __future__ import annotations

import logging
import math
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import Document, Window, make_windows
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


def _collate(windows: list[Window], pad_id: int):
    n = max(len(w.input_ids) for w in windows)
    ids = torch.full((len(windows), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(windows), n), dtype=torch.long)
    labels = torch.full((len(windows), n), IGNORE_INDEX, dtype=torch.long)
    sens = torch.tensor([w.sensitive for w in windows])
    for i, w in enumerate(windows):
        ids[i, : len(w.input_ids)] = torch.from_numpy(w.input_ids.astype(np.int64))
        mask[i, : len(w.input_ids)] = 1
        if w.labels is not None:
            labels[i, : len(w.labels)] = torch.from_numpy(w.labels.astype(np.int64))
    return ids, mask, labels, sens


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def predict_document(model, doc: Document, space: LabelSpace, decoder: ViterbiDecoder, cfg, device,
                     bos_id: int, eos_id: int, pad_id: int):
    """Average overlapping windows' log-probs per token, then one Viterbi pass over the document."""
    n = len(doc.token_ids)
    if n == 0:
        return [], []
    windows = make_windows(doc, 0, cfg.data.max_length, cfg.data.stride, bos_id, eos_id, with_labels=False)
    acc = torch.full((n, space.num_labels), -math.inf)
    count = torch.zeros(n)
    bs = max(1, int(cfg.train.batch_size))
    for i in range(0, len(windows), bs):
        chunk = windows[i: i + bs]
        ids, mask, _, _ = _collate(chunk, pad_id)
        with _autocast(device, cfg.train.amp):
            logits, _ = model(ids.to(device), mask.to(device))
        lp = F.log_softmax(logits.float(), dim=-1).cpu()
        for w, row in zip(chunk, lp):
            m = len(w.input_ids) - 2
            sl = slice(w.token_start, w.token_start + m)
            acc[sl] = torch.logaddexp(acc[sl], row[1: 1 + m])
            count[sl] += 1
    avg = acc - torch.log(count.clamp(min=1)).unsqueeze(1)
    path = decoder.decode(avg)
    spans = []
    for typ, a, b in labels_to_spans(path, space):
        st, en = int(doc.offsets[a, 0]), int(doc.offsets[b - 1, 1])
        if cfg.decode.trim_whitespace:
            while st < en and doc.text[st].isspace():
                st += 1
            while en > st and doc.text[en - 1].isspace():
                en -= 1
        if en > st:
            spans.append((typ, st, en))
    return spans, path


def evaluate(model, docs: list[Document], space: LabelSpace, cfg, device, bos_id, eos_id, pad_id) -> dict:
    model.eval()
    decoder = ViterbiDecoder(space, dict(cfg.decode.biases))
    limit = cfg.eval.max_docs
    docs = docs if limit is None else docs[: int(limit)]
    scorer = SpanScorer(space.span_types)
    for doc in docs:
        pred_spans, path = predict_document(model, doc, space, decoder, cfg, device, bos_id, eos_id, pad_id)
        scorer.add(doc.spans, pred_spans, doc.token_labels.tolist(), path)
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


def train(cfg, train_docs: list[Document], space: LabelSpace, eval_docs=None, eval_prefix="val",
          report=None):
    """Train for cfg.train.epochs; after each epoch evaluate on eval_docs (if given) and call report."""
    set_seed(cfg.seed)
    if cfg.train.num_threads:
        torch.set_num_threads(int(cfg.train.num_threads))
    device = pick_device(cfg.train.device)
    bos_id, eos_id, pad_id = special_ids(cfg.model.name)
    windows = [w for i, d in enumerate(train_docs)
               for w in make_windows(d, i, cfg.data.max_length, cfg.data.stride, bos_id, eos_id)]
    bs = int(cfg.train.batch_size)
    steps_per_epoch = math.ceil(len(windows) / bs)
    total = steps_per_epoch * int(cfg.train.epochs)
    model, optim, sched = build_model_and_optim(cfg, space, total, device)
    weights = class_weights(space, cfg.loss.background_weight)
    log.info("training on %d docs / %d windows, %d steps, device=%s", len(train_docs), len(windows), total, device)

    rng = random.Random(cfg.seed)
    metrics: dict = {}
    for epoch in range(int(cfg.train.epochs)):
        model.train()
        rng.shuffle(windows)
        t0, run_loss, run_tag, run_sens = time.time(), 0.0, 0.0, 0.0
        for step in range(steps_per_epoch):
            ids, mask, labels, sens = _collate(windows[step * bs: (step + 1) * bs], pad_id)
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
