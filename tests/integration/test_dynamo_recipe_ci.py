# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess

import pytest

from tools.dynamo_recipe_ci.runner import run_corpus

pytestmark = pytest.mark.integration


def test_local_recipe_corpus_runs_real_estimate(tmp_path):
    dynamo_root = tmp_path / "dynamo"
    recipe_dir = dynamo_root / "recipes" / "qwen" / "vllm" / "agg"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "deploy.yaml").write_text(
        """
kind: DynamoGraphDeployment
metadata: {name: qwen-sol}
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
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
version: 1
defaults:
  overrides:
    database_mode: SOL
    workload_points:
      - {point_id: sol, isl: 128, osl: 16, concurrency: 2}
targets:
  - id: qwen-sol
    include: recipes/qwen/**/deploy.yaml
    gate: true
    min_matches: 1
    max_matches: 1
    expected: {mapping: adapted, estimate: valid}
"""
    )
    subprocess.run(["git", "init", "-q"], cwd=dynamo_root, check=True)
    subprocess.run(["git", "config", "user.name", "AIC Test"], cwd=dynamo_root, check=True)
    subprocess.run(["git", "config", "user.email", "aic-test@example.com"], cwd=dynamo_root, check=True)
    subprocess.run(["git", "add", "recipes"], cwd=dynamo_root, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"],
        cwd=dynamo_root,
        check=True,
    )
    output_dir = tmp_path / "output"

    exit_code = run_corpus(
        dynamo_root=dynamo_root,
        manifest_path=manifest,
        output_dir=output_dir,
        jobs=1,
        timeout_seconds=120,
    )

    report = json.loads((output_dir / "results.json").read_text())
    assert exit_code == 0
    assert report["summary"]["gate_passed"] is True
    assert report["summary"]["estimates"] == {"valid": 1}
    assert report["results"][0]["estimate"]["result"]["tokens_per_second"] > 0
