"""mmBERT encoder with a BIES token-tagging head and an auxiliary sensitivity head.

Loss follows PII-TRACE (arXiv 2609.22200, eq. 3):
    L = lambda_tok * class-weighted CE over BIES tags + lambda_sens * BCE on a sensitivity logit
The sensitivity logit is computed from mean-pooled token states, as in the released PII-Tracer.
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
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Linear(hidden, num_labels)
        self.sens_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.dropout(h)
        tag_logits = self.tag_head(h)
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        sens_logit = self.sens_head(pooled).squeeze(-1)
        return tag_logits, sens_logit


def class_weights(space: LabelSpace, background_weight: float) -> torch.Tensor:
    w = torch.ones(space.num_labels)
    w[0] = background_weight
    return w


def pii_trace_loss(tag_logits, sens_logit, labels, sens_labels, weights, lambda_tok, lambda_sens):
    tag = F.cross_entropy(tag_logits.float().reshape(-1, tag_logits.shape[-1]), labels.reshape(-1),
                          weight=weights.to(tag_logits.device), ignore_index=IGNORE_INDEX)
    sens = F.binary_cross_entropy_with_logits(sens_logit.float(), sens_labels.float())
    return lambda_tok * tag + lambda_sens * sens, tag.detach(), sens.detach()


def save_model(model: PiiTagger, tokenizer, space: LabelSpace, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out_dir / "encoder")
    tokenizer.save_pretrained(out_dir / "encoder")
    torch.save({"tag_head": model.tag_head.state_dict(), "sens_head": model.sens_head.state_dict()},
               out_dir / "heads.pt")
    meta = dict(meta, span_types=list(space.span_types), label_names=list(space.names))
    (out_dir / "pii_tagger.json").write_text(json.dumps(meta, indent=2))


def load_model(model_dir: Path, device="cpu"):
    meta = json.loads((model_dir / "pii_tagger.json").read_text())
    space = LabelSpace(tuple(meta["span_types"]))
    model = PiiTagger(str(model_dir / "encoder"), space.num_labels, dropout=0.0)
    heads = torch.load(model_dir / "heads.pt", map_location="cpu")
    model.tag_head.load_state_dict(heads["tag_head"])
    model.sens_head.load_state_dict(heads["sens_head"])
    return model.to(device).eval(), space, meta
