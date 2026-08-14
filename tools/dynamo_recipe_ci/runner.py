# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover Dynamo recipes, adapt them, and replay AIC estimates safely."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aiconfigurator.sdk.config_adapter import (
    AdapterOverrides,
    DynamoRecipeSource,
    EstimateRequestV1,
    adapt_config,
    to_cli_estimate_kwargs,
)

KNOWN_UNAVAILABLE_EXCEPTIONS = {
    "EmpiricalNotImplementedError",
    "InsufficientMemoryError",
    "InterpolationDataNotAvailableError",
    "KVCacheCapacityError",
    "MissingSystemFlopsError",
    "NoFeasibleConfigError",
    "NoResultsError",
    "PerfDataNotAvailableError",
    "UnsupportedWideepConfigError",
}
UNAVAILABLE_MESSAGE_MARKERS = (
    "does not fit",
    "insufficient memory",
    "kv cache capacity",
    "no empirical utilisation data",
    "no feasible",
    "no performance data",
    "no results",
    "oom",
    "perf data",
    "unsupported hardware",
)


@dataclass(frozen=True)
class Target:
    identifier: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    min_matches: int
    max_matches: int | None
    performance: str | None
    overrides: dict[str, Any]
    path_overrides: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Manifest:
    defaults: dict[str, Any]
    targets: tuple[Target, ...]


@dataclass(frozen=True)
class Recipe:
    relative_path: str
    deployment: Path
    performance: Path | None
    performance_relative_path: str | None
    target: Target | None


def _read_object(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML object")
    return value


def load_manifest(path: Path) -> Manifest:
    raw = _read_object(path)
    if raw.get("version") != 1:
        raise ValueError("manifest version must be 1")
    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise TypeError("manifest defaults must be an object")
    missing_system_name = defaults.get("missing_system_name")
    if missing_system_name is not None and (not isinstance(missing_system_name, str) or not missing_system_name):
        raise ValueError("manifest defaults.missing_system_name must be a non-empty string")
    targets: list[Target] = []
    identifiers: set[str] = set()
    for index, item in enumerate(raw.get("targets", [])):
        if not isinstance(item, dict):
            raise TypeError(f"manifest target {index} must be an object")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"manifest target {index} requires a non-empty id")
        if identifier in identifiers:
            raise ValueError(f"duplicate manifest target id {identifier!r}")
        identifiers.add(identifier)
        include_value = item.get("include")
        includes = (include_value,) if isinstance(include_value, str) else tuple(include_value or ())
        if not includes or not all(isinstance(pattern, str) for pattern in includes):
            raise ValueError(f"manifest target {identifier!r} requires include glob(s)")
        exclude_value = item.get("exclude", ())
        excludes = (exclude_value,) if isinstance(exclude_value, str) else tuple(exclude_value or ())
        if not all(isinstance(pattern, str) for pattern in excludes):
            raise ValueError(f"manifest target {identifier!r} exclude must contain glob strings")
        min_matches = int(item.get("min_matches", 1))
        max_matches_value = item.get("max_matches")
        max_matches = int(max_matches_value) if max_matches_value is not None else None
        overrides = item.get("overrides", {})
        if not isinstance(overrides, dict):
            raise TypeError(f"manifest target {identifier!r} overrides must be an object")
        path_overrides = item.get("path_overrides", {})
        if not isinstance(path_overrides, dict) or not all(
            isinstance(key, str) and isinstance(value, dict) for key, value in path_overrides.items()
        ):
            raise TypeError(f"manifest target {identifier!r} path_overrides must map paths to objects")
        targets.append(
            Target(
                identifier=identifier,
                include=includes,
                exclude=excludes,
                min_matches=min_matches,
                max_matches=max_matches,
                performance=item.get("performance"),
                overrides=overrides,
                path_overrides=path_overrides,
            )
        )
    return Manifest(defaults=defaults, targets=tuple(targets))


def _matches(relative_path: str, target: Target) -> bool:
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in target.include) and not any(
        fnmatch.fnmatch(relative_path, pattern) for pattern in target.exclude
    )


def _adjacent_performance(deployment: Path) -> Path | None:
    for name in ("perf.yaml", "perf.yml"):
        candidate = deployment.with_name(name)
        if candidate.is_file():
            return candidate
    return None


def discover_recipes(dynamo_root: Path, manifest: Manifest) -> tuple[list[Recipe], list[str]]:
    recipes_root = dynamo_root / "recipes"
    if not recipes_root.is_dir():
        raise ValueError(f"Dynamo recipes directory does not exist: {recipes_root}")
    matched_counts: Counter[str] = Counter()
    recipes: list[Recipe] = []
    errors: list[str] = []
    deployment_paths = sorted((*recipes_root.rglob("deploy.yaml"), *recipes_root.rglob("deploy.yml")))
    for deployment in deployment_paths:
        relative_path = deployment.relative_to(dynamo_root).as_posix()
        targets = [target for target in manifest.targets if _matches(relative_path, target)]
        if len(targets) > 1:
            errors.append(
                f"{relative_path} matches multiple manifest targets: "
                + ", ".join(target.identifier for target in targets)
            )
            continue
        target = targets[0] if targets else None
        if target is not None:
            matched_counts[target.identifier] += 1
        if target is not None and target.performance:
            performance = dynamo_root / target.performance
            if not performance.is_file():
                errors.append(f"{target.identifier}: performance file does not exist: {target.performance}")
                performance = None
        else:
            performance = _adjacent_performance(deployment)
        performance_relative = performance.relative_to(dynamo_root).as_posix() if performance is not None else None
        recipes.append(Recipe(relative_path, deployment, performance, performance_relative, target))
    for target in manifest.targets:
        count = matched_counts[target.identifier]
        if count < target.min_matches:
            errors.append(f"{target.identifier}: matched {count} recipes, expected at least {target.min_matches}")
        if target.max_matches is not None and count > target.max_matches:
            errors.append(f"{target.identifier}: matched {count} recipes, expected at most {target.max_matches}")
        declared_paths = set(target.path_overrides)
        discovered_paths = {recipe.relative_path for recipe in recipes if recipe.target == target}
        for missing_path in sorted(declared_paths - discovered_paths):
            errors.append(f"{target.identifier}: declared recipe path was not discovered: {missing_path}")
    return recipes, errors


def _git_sha(root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _merged_overrides(manifest: Manifest, recipe: Recipe) -> AdapterOverrides:
    values = dict(manifest.defaults.get("overrides", {}))
    if recipe.target is not None:
        values.update(recipe.target.overrides)
        values.update(recipe.target.path_overrides.get(recipe.relative_path, {}))
    if isinstance(values.get("workload_points"), list):
        values["workload_points"] = tuple(values["workload_points"])
    return AdapterOverrides.model_validate(values)


def _stable_id(relative_path: str, point_id: str) -> str:
    digest = hashlib.sha256(f"{relative_path}\0{point_id}".encode()).hexdigest()[:12]
    return f"{relative_path.removesuffix('/deploy.yaml').replace('/', '__')}__{point_id}__{digest}"


def _missing_system_fallback(recipe: Recipe, manifest: Manifest) -> tuple[str | None, str | None]:
    path = recipe.relative_path.lower()
    path_markers = (
        ("gb300", "gb300"),
        ("gb200", "gb200"),
        ("b300", "b300_sxm"),
        ("b200", "b200_sxm"),
        ("hopper", "h200_sxm"),
    )
    for marker, system_name in path_markers:
        if re.search(rf"(?:^|[/_.-]){marker}(?:$|[/_.-])", path):
            return system_name, f"recipe path marker {marker!r}"
    default = manifest.defaults.get("missing_system_name")
    return (str(default), "manifest default") if default else (None, None)


def adapt_recipe(recipe: Recipe, manifest: Manifest, dynamo_sha: str) -> list[dict[str, Any]]:
    overrides = _merged_overrides(manifest, recipe)
    source = DynamoRecipeSource(
        deployment=recipe.deployment,
        performance=recipe.performance,
        source_reference=f"ai-dynamo/dynamo@{dynamo_sha}:{recipe.relative_path}",
    )
    report = adapt_config(source, overrides)
    fallback_system, fallback_source = _missing_system_fallback(recipe, manifest)
    defaulted_system = None
    if fallback_system and all(
        outcome.status == "rejected"
        and any("GPU model is not declared" in diagnostic.message for diagnostic in outcome.diagnostics)
        for outcome in report.outcomes
    ):
        defaulted_system = fallback_system
        report = adapt_config(source, overrides.model_copy(update={"system_name": defaulted_system}))
    records: list[dict[str, Any]] = []
    for outcome in report.outcomes:
        request = outcome.request
        if request is not None and defaulted_system is not None:
            provenance = request.provenance.model_copy(
                update={
                    "assumptions": (
                        *request.provenance.assumptions,
                        f"GPU model was absent from the Dynamo deployment; selected {defaulted_system} from "
                        f"{fallback_source}.",
                    )
                }
            )
            request = request.model_copy(update={"provenance": provenance})
        record: dict[str, Any] = {
            "id": _stable_id(recipe.relative_path, outcome.point_id),
            "recipe": recipe.relative_path,
            "source_url": f"https://github.com/ai-dynamo/dynamo/blob/{dynamo_sha}/{recipe.relative_path}",
            "performance": recipe.performance_relative_path,
            "target": recipe.target.identifier if recipe.target else None,
            "point_id": outcome.point_id,
            "mapping": {
                "status": outcome.status,
                "diagnostics": [item.model_dump(mode="json") for item in outcome.diagnostics],
                "defaults": {"system_name": defaulted_system} if defaulted_system is not None else {},
                "default_sources": {"system_name": fallback_source} if defaulted_system is not None else {},
            },
            "estimate": {"status": "not_run"},
        }
        if request is not None:
            record["request"] = request.model_dump(mode="json")
            record["cli_kwargs"] = to_cli_estimate_kwargs(request)
        records.append(record)
    return records


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def estimate_request(request_path: Path, output_path: Path) -> int:
    from aiconfigurator.cli.api import cli_estimate

    request = EstimateRequestV1.model_validate_json(request_path.read_text())
    try:
        result = cli_estimate(**to_cli_estimate_kwargs(request))
    except Exception as error:
        exception = type(error).__name__
        normalized_message = str(error).lower()
        unavailable = exception in KNOWN_UNAVAILABLE_EXCEPTIONS or any(
            marker in normalized_message for marker in UNAVAILABLE_MESSAGE_MARKERS
        )
        status = "unavailable" if unavailable else "error"
        payload = {
            "status": status,
            "exception": exception,
            "message": str(error),
        }
    else:
        payload = {
            "status": "valid",
            "result": {
                "mode": result.mode,
                "ttft": result.ttft,
                "tpot": result.tpot,
                "request_latency": result.request_latency,
                "tokens_per_second": result.tokens_per_second,
                "tokens_per_second_per_gpu": result.tokens_per_second_per_gpu,
                "tokens_per_second_per_user": result.tokens_per_second_per_user,
                "power_w": result.power_w,
                "backend_version": result.backend_version,
                "raw": _json_safe(result.raw),
            },
        }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _run_one(record: dict[str, Any], temp_root: Path, timeout_seconds: int) -> dict[str, Any]:
    if "request" not in record:
        return record
    request_path = temp_root / f"{record['id']}.request.json"
    output_path = temp_root / f"{record['id']}.result.json"
    request_path.write_text(json.dumps(record["request"], sort_keys=True))
    command = [
        sys.executable,
        "-m",
        "tools.dynamo_recipe_ci.runner",
        "--estimate-one",
        str(request_path),
        str(output_path),
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""
        record["estimate"] = {
            "status": "timeout",
            "message": f"estimate exceeded {timeout_seconds} seconds",
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
        }
        return record
    if process.returncode != 0 or not output_path.is_file():
        record["estimate"] = {
            "status": "error",
            "exception": "EstimateSubprocessError",
            "message": f"estimate subprocess exited with code {process.returncode}",
            "stdout": process.stdout[-4000:],
            "stderr": process.stderr[-4000:],
        }
        return record
    record["estimate"] = json.loads(output_path.read_text())
    if process.stdout:
        record["estimate"]["stdout"] = process.stdout[-4000:]
    if process.stderr:
        record["estimate"]["stderr"] = process.stderr[-4000:]
    return record


def run_estimates(records: list[dict[str, Any]], jobs: int, timeout_seconds: int) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="dynamo-recipe-ci-") as temporary:
        temp_root = Path(temporary)
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_run_one, record, temp_root, timeout_seconds) for record in records]
            return [future.result() for future in as_completed(futures)]


def gate_violations(records: list[dict[str, Any]], discovery_errors: list[str]) -> list[str]:
    violations = list(discovery_errors)
    for record in records:
        if record["mapping"]["status"] != "adapted":
            continue
        estimate_status = record["estimate"]["status"]
        if estimate_status != "valid":
            violations.append(f"{record['id']}: adapted estimate finished as {estimate_status}")
    return violations


def _summary(records: list[dict[str, Any]], dynamo_sha: str, violations: list[str]) -> dict[str, Any]:
    mapping = Counter(record["mapping"]["status"] for record in records)
    estimates = Counter(record["estimate"]["status"] for record in records)
    mapping_defaults = Counter(
        f"{name}={value}" for record in records for name, value in record["mapping"].get("defaults", {}).items()
    )
    return {
        "dynamo_sha": dynamo_sha,
        "recipes": len({record["recipe"] for record in records}),
        "operating_points": len(records),
        "mapping": dict(sorted(mapping.items())),
        "mapping_defaults": dict(sorted(mapping_defaults.items())),
        "estimates": dict(sorted(estimates.items())),
        "gate_passed": not violations,
        "violations": violations,
    }


def _comment_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    failed = [
        record
        for record in records
        if record["mapping"]["status"] == "adapted" and record["estimate"]["status"] != "valid"
    ]
    lines = [
        "## Dynamo Recipe Replay",
        "",
        f"**Estimate gate: {'PASS' if summary['gate_passed'] else 'FAIL'}**",
        "",
        f"- Dynamo SHA: `{summary['dynamo_sha']}`",
        f"- Recipes discovered: {summary['recipes']}",
        f"- Successfully adapted: {summary['mapping'].get('adapted', 0)}",
        f"- Mapping rejected (report only): {summary['mapping'].get('rejected', 0)}",
        f"- Valid estimates: {summary['estimates'].get('valid', 0)}",
        f"- Adapted estimates that failed: {len(failed)}",
    ]
    if failed:
        lines.extend(("", "### Failed adapted estimates", "", "| Recipe | Status |", "| --- | --- |"))
        for record in sorted(failed, key=lambda item: item["id"])[:20]:
            lines.append(f"| `{record['recipe']}` | {record['estimate']['status']} |")
        if len(failed) > 20:
            lines.append(f"| … | {len(failed) - 20} more in the full report |")
    return "\n".join(lines) + "\n"


def _markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Dynamo Recipe → AIC Replay",
        "",
        f"- Dynamo SHA: `{summary['dynamo_sha']}`",
        f"- Recipes: {summary['recipes']}",
        f"- Operating points: {summary['operating_points']}",
        f"- Gate: **{'PASS' if summary['gate_passed'] else 'FAIL'}**",
        "",
        "## How to read this report",
        "",
        "The runner has two separate stages:",
        "",
        "1. **Mapping** converts a concrete Dynamo deployment into an AIC estimate request.",
        "2. **Estimate** runs AIC only when mapping succeeded.",
        "",
        "| Status | Plain-language meaning |",
        "| --- | --- |",
        "| `adapted` | Dynamo supplied enough literal information to build an AIC request safely. |",
        "| `rejected` | The recipe could not be mapped safely or unambiguously; no estimate was run. |",
        "| `valid` | AIC completed and returned a prediction. |",
        "| `unavailable` | AIC ran, but the configuration is currently unsupported or infeasible "
        "(for example, missing performance data or insufficient GPU memory). |",
        "| `error` | AIC or its dependencies failed unexpectedly; inspect the exception in `results.json`. |",
        "| `timeout` | The estimate exceeded its per-point time limit. |",
        "| `not_run` | Mapping did not produce a request, so there was nothing safe to execute. |",
        "",
        "A rejected mapping does not mean the Dynamo recipe is invalid. It means this adapter cannot yet "
        "translate the recipe without guessing or executing recipe code.",
        "",
        "| Stage | Status | Count |",
        "| --- | --- | ---: |",
    ]
    for stage in ("mapping", "estimates"):
        for status, count in summary[stage].items():
            lines.append(f"| {stage} | {status} | {count} |")
    if summary.get("mapping_defaults"):
        lines.extend(("", "## Mapping defaults applied", "", "| Default | Count |", "| --- | ---: |"))
        for default, count in summary["mapping_defaults"].items():
            lines.append(f"| `{default}` | {count} |")
    if summary["violations"]:
        lines.extend(("", "## Gate violations", ""))
        lines.extend(f"- {item}" for item in summary["violations"])
    rejection_reasons = Counter(
        diagnostic["message"]
        for record in records
        if record["mapping"]["status"] == "rejected"
        for diagnostic in record["mapping"].get("diagnostics", [])[-1:]
    )
    if rejection_reasons:
        lines.extend(
            (
                "",
                "## Why mappings were rejected",
                "",
                "| Count | Reason |",
                "| ---: | --- |",
            )
        )
        for reason, count in rejection_reasons.most_common():
            escaped_reason = reason.replace("|", "\\|")
            lines.append(f"| {count} | {escaped_reason} |")
    estimate_errors = Counter(
        (record["estimate"].get("exception") or "UnknownError")
        for record in records
        if record["estimate"]["status"] == "error"
    )
    if estimate_errors:
        lines.extend(("", "## Unexpected estimate errors", "", "| Count | Exception |", "| ---: | --- |"))
        for exception, count in estimate_errors.most_common():
            lines.append(f"| {count} | `{exception}` |")
    lines.extend(
        (
            "",
            "## Recipe outcomes",
            "",
            "| Recipe | Point | Target | Mapping | Estimate |",
            "| --- | --- | --- | --- | --- |",
        )
    )
    for record in sorted(records, key=lambda item: item["id"]):
        source_url = record.get("source_url") or (
            f"https://github.com/ai-dynamo/dynamo/blob/{summary['dynamo_sha']}/{record['recipe']}"
        )
        recipe = f"[`{record['recipe']}`]({source_url})"
        lines.append(
            f"| {recipe} | `{record['point_id']}` | "
            f"{record['target'] or '-'} | {record['mapping']['status']} | {record['estimate']['status']} |"
        )
    return "\n".join(lines) + "\n"


def run_corpus(
    *,
    dynamo_root: Path,
    manifest_path: Path,
    output_dir: Path,
    jobs: int,
    timeout_seconds: int,
) -> int:
    manifest = load_manifest(manifest_path)
    dynamo_sha = _git_sha(dynamo_root)
    recipes, discovery_errors = discover_recipes(dynamo_root, manifest)
    records = [record for recipe in recipes for record in adapt_recipe(recipe, manifest, dynamo_sha)]
    records = run_estimates(records, jobs, timeout_seconds)
    records.sort(key=lambda item: item["id"])
    violations = gate_violations(records, discovery_errors)
    summary = _summary(records, dynamo_sha, violations)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "results": records}, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "summary.md").write_text(_markdown(summary, records))
    (output_dir / "comment.md").write_text(_comment_markdown(summary, records))
    return 0 if not violations else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamo-root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("corpus.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--estimate-one", nargs=2, metavar=("REQUEST", "OUTPUT"), type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.estimate_one:
        return estimate_request(*args.estimate_one)
    if args.dynamo_root is None or args.output_dir is None:
        raise SystemExit("--dynamo-root and --output-dir are required")
    if args.jobs <= 0 or args.timeout_seconds <= 0:
        raise SystemExit("--jobs and --timeout-seconds must be positive")
    return run_corpus(
        dynamo_root=args.dynamo_root,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        jobs=args.jobs,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
