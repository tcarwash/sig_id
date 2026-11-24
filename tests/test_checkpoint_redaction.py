"""Path scrubbing is separate from preserving learned weight tensors."""
from tools.sanitize_detector_checkpoint import redact


def test_nested_private_training_paths_are_redacted():
    source = {
        "model": "/home/example/train/run.pt",
        "nested": [{"data": "/storage5/dev/private/data.yaml"}],
        "tuple": ("C:\\Users\\operator\\model.pt", "safe"),
        "public": "AGPL-3.0",
        "count": 5,
    }
    scrubbed = redact(source)
    assert scrubbed["model"] == "<redacted-local-path>"
    assert scrubbed["nested"][0]["data"] == "<redacted-local-path>"
    assert scrubbed["tuple"][0] == "<redacted-local-path>"
    assert scrubbed["tuple"][1] == "safe"
    assert scrubbed["public"] == "AGPL-3.0"
    assert scrubbed["count"] == 5
    assert source["nested"][0]["data"].startswith("/storage5/")


def test_same_source_with_no_private_paths_is_unchanged():
    data = {"names": {0: "signal"}, "links": ["https://docs.ultralytics.com"],
            "args": {"model": "cv-signal.pt", "task": "detect"}}
    assert redact(data) == data
