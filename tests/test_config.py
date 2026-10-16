from src.utils.config import load_config


def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\nb:\n  c: 2\n")
    cfg = load_config(str(p))
    assert cfg["a"] == 1
    assert cfg["b"]["c"] == 2