# Dynamo recipe replay

This harness converts Dynamo recipe YAML into the versioned AIC estimate
request, lowers every adapted point to `cli_estimate`, and records the result.
It never executes commands from a recipe. Static engine invocations may be
read from a shell wrapper, but shell expressions and control operators in the
engine invocation remain unsupported.

Run it against a Dynamo checkout:

```bash
python -m tools.dynamo_recipe_ci.runner \
  --dynamo-root ../dynamo \
  --output-dir /tmp/dynamo-recipe-replay \
  --jobs 4
```

The output directory contains `results.json` for automation and `summary.md`
for CI. Every result records the exact Dynamo commit, recipe and performance
paths, canonical `EstimateRequestV1`, lowered CLI arguments, mapping
diagnostics, and the estimate result or classified failure.

`aic_1743.yaml` is the blocking corpus. Its selectors intentionally match the
50 configurations in AIC-1743. A target expected to be `valid` fails if it
becomes unavailable; a target with a known `unavailable` baseline may improve
to valid, but an unexpected exception or timeout still fails. Recipes outside
the manifest are discovered and replayed for coverage reporting without
blocking the gate.

Workload values that do not exist as safe literals in recipe YAML belong in
the manifest. In particular, speculative acceptance length and the AIC
performance-database version must never be guessed by the adapter.

For corpus coverage, a deployment without a machine-readable GPU model uses a
specific path marker when present (`gb200` → `gb200`, `b200` → `b200_sxm`, and
`hopper` → `h200_sxm`), then falls back to
`defaults.missing_system_name: gb300`. Source-declared GPU models still win.
Every use and its source are recorded in the JSON mapping metadata and the
canonical request's provenance assumptions.
