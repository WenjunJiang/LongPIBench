"""mmBERT encoder with a BIES token-tagging head.

Loss: token-level cross-entropy over the BIES labels, averaged over non-special tokens, as in OPF's
fine-tuning code (opf/_train/runner.py, _masked_token_loss_and_accuracy). Precision/recall trade-offs
are set after training with Viterbi biases and thresholds (stage=calibrate), not with class weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .labels import IGNORE_INDEX, LabelSpace


class PiiTagger(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.tag_head(self.dropout(h))


def class_weights(space: LabelSpace, background_weight: float) -> torch.Tensor | None:
    """None (plain cross-entropy) unless the "O" weight differs from 1."""
    if float(background_weight) == 1.0:
        return None
    w = torch.ones(space.num_labels)
    w[0] = float(background_weight)
    return w


def token_loss(tag_logits, labels, weights=None):
    return F.cross_entropy(tag_logits.float().reshape(-1, tag_logits.shape[-1]), labels.reshape(-1),
                           weight=None if weights is None else weights.to(tag_logits.device),
                           ignore_index=IGNORE_INDEX)


def save_model(model: PiiTagger, tokenizer, space: LabelSpace, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out_dir / "encoder")
    tokenizer.save_pretrained(out_dir / "encoder")
    torch.save({"tag_head": model.tag_head.state_dict()}, out_dir / "heads.pt")
    meta = dict(meta, span_types=list(space.span_types), label_names=list(space.names))
    (out_dir / "pii_tagger.json").write_text(json.dumps(meta, indent=2))


def load_model(model_dir: Path, device="cpu"):
    meta = json.loads((model_dir / "pii_tagger.json").read_text())
    space = LabelSpace(tuple(meta["span_types"]))
    model = PiiTagger(str(model_dir / "encoder"), space.num_labels, dropout=0.0)
    heads = torch.load(model_dir / "heads.pt", map_location="cpu")
    model.tag_head.load_state_dict(heads["tag_head"])
    return model.to(device).eval(), space, meta
