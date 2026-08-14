# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from aiconfigurator.sdk.config_adapter import EstimateRequestV1
from tools.dynamo_recipe_ci import runner

pytestmark = pytest.mark.unit


DEPLOYMENT = """
kind: DynamoGraphDeployment
metadata: {name: fixture}
spec:
  backendFramework: vllm
  components:
    - name: agg
      type: worker
      replicas: 1
      podTemplate:
        spec:
          nodeSelector: {nvidia.com/gpu.product: NVIDIA-H100}
          containers:
            - name: main
              command: [python3, -m, dynamo.vllm]
              args: [--model=Qwen/Qwen3-32B, --tensor-parallel-size=1]
              resources: {limits: {nvidia.com/gpu: "1"}}
"""


def _manifest(path: Path, *, max_matches: int = 1) -> Path:
    path.write_text(
        f"""
version: 1
defaults:
  overrides:
    database_mode: SILICON
    workload_points:
      - {{point_id: replay, isl: 128, osl: 16, concurrency: 2}}
targets:
  - id: fixture
    include: recipes/fixture/**/deploy.yaml
    min_matches: 1
    max_matches: {max_matches}
"""
    )
    return path


def _dynamo_tree(tmp_path: Path) -> Path:
    root = tmp_path / "dynamo"
    deployment = root / "recipes" / "fixture" / "vllm" / "deploy.yaml"
    deployment.parent.mkdir(parents=True)
    deployment.write_text(DEPLOYMENT)
    return root


def test_discovery_matches_manifest_and_uses_adjacent_perf(tmp_path):
    root = _dynamo_tree(tmp_path)
    performance = root / "recipes" / "fixture" / "vllm" / "perf.yaml"
    performance.write_text("points: [{point_id: ignored, isl: 1, osl: 1, concurrency: 1}]\n")
    manifest = runner.load_manifest(_manifest(tmp_path / "manifest.yaml"))

    recipes, errors = runner.discover_recipes(root, manifest)

    assert errors == []
    assert len(recipes) == 1
    assert recipes[0].performance == performance
    assert recipes[0].performance_relative_path == "recipes/fixture/vllm/perf.yaml"
    assert recipes[0].target.identifier == "fixture"


def test_adaptation_uses_manifest_point_and_stable_id(tmp_path):
    root = _dynamo_tree(tmp_path)
    manifest = runner.load_manifest(_manifest(tmp_path / "manifest.yaml"))
    recipe = runner.discover_recipes(root, manifest)[0][0]

    first = runner.adapt_recipe(recipe, manifest, "abc123")[0]
    second = runner.adapt_recipe(recipe, manifest, "different-sha")[0]

    assert first["mapping"]["status"] == "adapted"
    assert first["request"]["workload"] == {
        "concurrency": 2,
        "image_height": 0,
        "image_width": 0,
        "isl": 128,
        "num_images": 1,
        "osl": 16,
        "prefix": 0,
    }
    assert first["id"] == second["id"]
    assert first["source_url"] == ("https://github.com/ai-dynamo/dynamo/blob/abc123/recipes/fixture/vllm/deploy.yaml")
    assert first["request"]["provenance"]["source_reference"].startswith("ai-dynamo/dynamo@abc123:")


def test_missing_gpu_defaults_to_manifest_system_and_records_assumption(tmp_path):
    root = _dynamo_tree(tmp_path)
    deployment = root / "recipes" / "fixture" / "vllm" / "deploy.yaml"
    deployment.write_text(DEPLOYMENT.replace("nodeSelector: {nvidia.com/gpu.product: NVIDIA-H100}", "nodeSelector: {}"))
    manifest_path = _manifest(tmp_path / "manifest.yaml")
    manifest_path.write_text(
        manifest_path.read_text().replace("defaults:\n", "defaults:\n  missing_system_name: gb300\n")
    )
    manifest = runner.load_manifest(manifest_path)
    recipe = runner.discover_recipes(root, manifest)[0][0]

    record = runner.adapt_recipe(recipe, manifest, "abc123")[0]

    assert record["mapping"]["status"] == "adapted"
    assert record["mapping"]["defaults"] == {"system_name": "gb300"}
    assert record["request"]["systems"]["prefill"] == "gb300"
    assert record["mapping"]["default_sources"] == {"system_name": "manifest default"}
    assert record["request"]["provenance"]["assumptions"][-1] == (
        "GPU model was absent from the Dynamo deployment; selected gb300 from manifest default."
    )


@pytest.mark.parametrize(
    ("path", "expected", "source"),
    [
        ("recipes/model/agg-gb200/deploy.yaml", "gb200", "recipe path marker 'gb200'"),
        ("recipes/model/hopper/deploy.yaml", "h200_sxm", "recipe path marker 'hopper'"),
        ("recipes/model/agg_b200/deploy.yaml", "b200_sxm", "recipe path marker 'b200'"),
        ("recipes/model/generic/deploy.yaml", "gb300", "manifest default"),
    ],
)
def test_missing_system_fallback_prefers_path_markers(path, expected, source, tmp_path):
    manifest_path = _manifest(tmp_path / "manifest.yaml")
    manifest_path.write_text(
        manifest_path.read_text().replace("defaults:\n", "defaults:\n  missing_system_name: gb300\n")
    )
    manifest = runner.load_manifest(manifest_path)
    recipe = runner.Recipe(path, tmp_path / "deploy.yaml", None, None, None)

    assert runner._missing_system_fallback(recipe, manifest) == (expected, source)


def test_path_overrides_preserve_recipe_specific_workload(tmp_path):
    root = _dynamo_tree(tmp_path)
    manifest_path = _manifest(tmp_path / "manifest.yaml")
    manifest_path.write_text(
        manifest_path.read_text().replace(
            "    max_matches: 1",
            """    max_matches: 1
    path_overrides:
      recipes/fixture/vllm/deploy.yaml: {concurrency: 3}""",
        )
    )
    manifest = runner.load_manifest(manifest_path)
    recipe = runner.discover_recipes(root, manifest)[0][0]
    record = runner.adapt_recipe(recipe, manifest, "abc123")[0]

    assert record["request"]["workload"]["concurrency"] == 3


def test_discovery_reports_overlapping_targets_and_count_drift(tmp_path):
    root = _dynamo_tree(tmp_path)
    manifest_path = _manifest(tmp_path / "manifest.yaml", max_matches=0)
    text = manifest_path.read_text()
    manifest_path.write_text(
        text
        + """
  - id: overlap
    include: recipes/fixture/**/deploy.yaml
"""
    )

    recipes, errors = runner.discover_recipes(root, runner.load_manifest(manifest_path))

    assert recipes == []
    assert any("matches multiple manifest targets" in error for error in errors)
    assert any("expected at least 1" in error for error in errors)


@pytest.mark.parametrize(
    ("mapping", "estimate", "violates"),
    [
        ("adapted", "valid", False),
        ("adapted", "unavailable", True),
        ("adapted", "error", True),
        ("adapted", "timeout", True),
        ("rejected", "not_run", False),
    ],
)
def test_gate_requires_every_adapted_estimate_to_be_valid(mapping, estimate, violates):
    record = {
        "id": "point",
        "recipe": "recipes/fixture/vllm/deploy.yaml",
        "target": "fixture",
        "mapping": {"status": mapping},
        "estimate": {"status": estimate},
    }

    violations = runner.gate_violations([record], [])

    assert bool(violations) is violates


def test_run_one_classifies_timeout(tmp_path, monkeypatch):
    request = EstimateRequestV1.model_validate(
        {
            "schema_version": "aic-estimate-request/1.0.0",
            "model": {"path": "Qwen/Qwen3-32B", "nextn": 0, "nextn_accepted": None},
            "quantization": {},
            "backend": {"name": "vllm", "database_mode": "SOL"},
            "systems": {"prefill": "h100_sxm"},
            "workload": {"isl": 128, "osl": 16, "concurrency": 2},
            "topology": {
                "kind": "agg",
                "worker": {
                    "replicas": 1,
                    "gpus_per_replica": 1,
                    "batch_size": 2,
                    "tp_size": 1,
                },
            },
            "provenance": {"source_type": "dynamo"},
        }
    )
    record = {"id": "timeout", "request": request.model_dump(mode="json"), "estimate": {"status": "not_run"}}

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout)

    result = runner._run_one(record, tmp_path, timeout_seconds=1)

    assert result["estimate"]["status"] == "timeout"


def test_estimate_request_classifies_known_oom_as_unavailable(tmp_path, monkeypatch):
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "output.json"
    request_path.write_text(
        json.dumps(
            {
                "model": {"path": "Qwen/Qwen3-32B"},
                "backend": {"name": "vllm", "database_mode": "SOL"},
                "systems": {"prefill": "h100_sxm"},
                "workload": {"isl": 128, "osl": 16, "concurrency": 2},
                "topology": {
                    "kind": "agg",
                    "worker": {
                        "replicas": 1,
                        "gpus_per_replica": 1,
                        "batch_size": 2,
                        "tp_size": 1,
                    },
                },
                "provenance": {"source_type": "dynamo"},
            }
        )
    )

    def fail(**kwargs):
        raise RuntimeError("OOM: requested batch does not fit")

    monkeypatch.setattr("aiconfigurator.cli.api.cli_estimate", fail)

    assert runner.estimate_request(request_path, output_path) == 0
    assert json.loads(output_path.read_text())["status"] == "unavailable"


def test_markdown_explains_statuses_and_summarizes_rejection_reasons():
    records = [
        {
            "id": "rejected",
            "recipe": "recipes/rejected/deploy.yaml",
            "source_url": "https://github.com/ai-dynamo/dynamo/blob/abc123/recipes/rejected/deploy.yaml",
            "point_id": "point",
            "target": None,
            "mapping": {
                "status": "rejected",
                "diagnostics": [{"message": "GPU model is not declared in a machine-readable deployment field"}],
            },
            "estimate": {"status": "not_run"},
        },
        {
            "id": "error",
            "recipe": "recipes/error/deploy.yaml",
            "source_url": "https://github.com/ai-dynamo/dynamo/blob/abc123/recipes/error/deploy.yaml",
            "point_id": "point",
            "target": None,
            "mapping": {"status": "adapted", "diagnostics": []},
            "estimate": {"status": "error", "exception": "ValueError"},
        },
    ]
    summary = runner._summary(records, "abc123", [])

    report = runner._markdown(summary, records)

    assert "## How to read this report" in report
    assert "A rejected mapping does not mean the Dynamo recipe is invalid" in report
    assert "| 1 | GPU model is not declared in a machine-readable deployment field |" in report
    assert "| 1 | `ValueError` |" in report
    assert "[``recipes/rejected/deploy.yaml``]" not in report
    assert "[`recipes/rejected/deploy.yaml`](https://github.com/ai-dynamo/dynamo/blob/abc123/" in report


def test_summary_counts_mapping_defaults():
    record = {
        "recipe": "recipes/defaulted/deploy.yaml",
        "mapping": {"status": "adapted", "defaults": {"system_name": "gb300"}},
        "estimate": {"status": "valid"},
    }

    summary = runner._summary([record], "abc123", [])

    assert summary["mapping_defaults"] == {"system_name=gb300": 1}


def test_comment_markdown_highlights_failed_adapted_estimates():
    records = [
        {
            "id": "valid",
            "recipe": "recipes/valid/deploy.yaml",
            "mapping": {"status": "adapted"},
            "estimate": {"status": "valid"},
        },
        {
            "id": "unavailable",
            "recipe": "recipes/unavailable/deploy.yaml",
            "mapping": {"status": "adapted"},
            "estimate": {"status": "unavailable"},
        },
        {
            "id": "rejected",
            "recipe": "recipes/rejected/deploy.yaml",
            "mapping": {"status": "rejected"},
            "estimate": {"status": "not_run"},
        },
    ]
    summary = runner._summary(records, "abc123", ["unavailable failed"])

    comment = runner._comment_markdown(summary, records)

    assert "**Estimate gate: FAIL**" in comment
    assert "Successfully adapted: 2" in comment
    assert "Adapted estimates that failed: 1" in comment
    assert "`recipes/unavailable/deploy.yaml`" in comment
    assert "`recipes/rejected/deploy.yaml`" not in comment
