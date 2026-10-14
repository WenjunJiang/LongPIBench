import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from pii_mmbert.engine import batch_log_probs, doc_log_probs
from pii_mmbert.labels import LabelSpace

SPACE = LabelSpace(("private_person", "private_phone"))


class PerTokenModel(torch.nn.Module):
    """Logits depend only on each token id, so every window must give the same log-probs per token."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(200, SPACE.num_labels)

    def forward(self, ids, mask):
        return self.emb(ids)


def _cfg(batch_size):
    return OmegaConf.create({"data": {"max_length": 12, "stride": 4}, "train": {"amp": False, "batch_size": 8},
                             "eval": {"batch_size": batch_size}})


def test_batched_log_probs_match_per_token_reference():
    torch.manual_seed(0)
    model = PerTokenModel().eval()
    docs = [list(range(10, 10 + n)) for n in (0, 3, 10, 27, 41)]   # empty, one window, several overlapping
    with torch.no_grad():
        out = batch_log_probs(model, docs, SPACE, _cfg(3), "cpu", bos_id=2, eos_id=1, pad_id=0)
        for ids, lp in zip(docs, out):
            assert lp.shape == (len(ids), SPACE.num_labels)
            if ids:
                ref = F.log_softmax(model.emb(torch.tensor(ids)), dim=-1)
                assert torch.allclose(lp, ref, atol=1e-5)
                single = doc_log_probs(model, ids, SPACE, _cfg(1), "cpu", 2, 1, 0)
                assert torch.allclose(lp, single, atol=1e-5)


def test_available_cpus_respects_cgroup_quota(tmp_path):
    import os
    from pii_mmbert.engine import available_cpus
    host = len(os.sched_getaffinity(0))
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "cpu.max").write_text("150000 100000\n")          # 1.5 CPUs -> 2 threads
    assert available_cpus(str(v2)) == min(host, 2)
    (v2 / "cpu.max").write_text("max 100000\n")             # no quota
    assert available_cpus(str(v2)) == host
    v1 = tmp_path / "v1" / "cpu"
    v1.mkdir(parents=True)
    (v1 / "cpu.cfs_quota_us").write_text("100000\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")
    assert available_cpus(str(tmp_path / "v1")) == 1
    (v1 / "cpu.cfs_quota_us").write_text("-1\n")
    assert available_cpus(str(tmp_path / "v1")) == host
