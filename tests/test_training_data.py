from __future__ import annotations

import json

from minicpm_dream.eval.generate import generate, write_suite
from training.build_dataset import build, write_records


def test_build_dataset_uses_production_prompts(tmp_path):
    generated = generate(seed=11, days=3, facts=24)
    suite = tmp_path / "suite"
    write_suite(generated, suite)
    train, valid, manifest = build([suite], chunk_tokens=1500, valid_ratio=0.0)
    assert manifest["extract"] >= 1
    assert manifest["reconcile"] >= 1
    assert train and not valid
    extract = [record for record in train if record["task"] == "extract"]
    assert extract
    for record in extract:
        assert "TASK: EXTRACT" in record["messages"][0]["content"]
        target = json.loads(record["messages"][-1]["content"])
        assert target["items"]
        for item in target["items"]:
            assert item["evidence"]
            assert "valid_from" not in item
    reconcile = [record for record in train if record["task"] == "reconcile"]
    assert reconcile
    for record in reconcile:
        assert "TASK: RECONCILE" in record["messages"][0]["content"]
        target = json.loads(record["messages"][-1]["content"])
        assert target["op"] in {"UPDATE", "SUPERSEDE", "MERGE"}
    write_records(train, tmp_path / "data" / "train.jsonl")
    assert (tmp_path / "data" / "train.jsonl").is_file()
