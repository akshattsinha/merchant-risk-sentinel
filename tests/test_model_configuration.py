import json
from pathlib import Path


def test_operating_threshold_is_cost_optimized():
    metadata_file = Path(
        "reports/fraud_model_metadata.json"
    )

    assert metadata_file.exists(), (
        "Model metadata file was not generated."
    )

    with open(metadata_file, "r") as file:
        metadata = json.load(file)

    threshold = float(
        metadata["threshold"]
    )

    assert threshold == 0.30, (
        f"Expected operating threshold 0.30, "
        f"got {threshold}"
    )


def test_model_metadata_contains_metrics():
    metadata_file = Path(
        "reports/fraud_model_metadata.json"
    )

    with open(metadata_file, "r") as file:
        metadata = json.load(file)

    assert "roc_auc" in metadata
    assert "pr_auc" in metadata

    assert 0 <= metadata["roc_auc"] <= 1
    assert 0 <= metadata["pr_auc"] <= 1
