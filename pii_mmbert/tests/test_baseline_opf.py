from pii_mmbert.baseline_opf import opf_checkpoint_dir


def _fake_ckpt(d):
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"")
    return d


def test_existing_checkpoint_dir_is_used_without_download(tmp_path):
    ckpt = _fake_ckpt(tmp_path / "ckpt")
    assert opf_checkpoint_dir(ckpt) == ckpt


def test_previous_download_dir_resolves_to_original(tmp_path):
    ckpt = _fake_ckpt(tmp_path / "dl" / "original")
    assert opf_checkpoint_dir(tmp_path / "dl") == ckpt
