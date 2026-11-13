"""Check the isolated final environment, regression tests, smoke/resume and release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import site
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "results/v6_3_3_final_four_models/run_6c5fb2bebb16a7dc"
LOCK = ROOT / "requirements-final-env.lock.txt"


def locked_versions(path=LOCK):
    """Read the concrete pins, including the original immutable runtime lock."""
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            pins.update(locked_versions(Path(path).parent / line[3:].strip()))
        elif not line.startswith("-"):
            name, version = line.split("==", 1)
            pins[name.lower().replace("_", "-")] = version
    return pins


def check_environment():
    if sys.prefix == sys.base_prefix or site.ENABLE_USER_SITE:
        raise ValueError("Use the isolated interpreter .venv-final/Scripts/python.exe; run scripts/setup_final_env.ps1 first.")
    if platform.python_version() != "3.13.14":
        raise ValueError(f"Expected Python 3.13.14, found {platform.python_version()}")
    versions, errors = {}, []
    for name, expected in locked_versions().items():
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing dependency: {name}=={expected}")
            continue
        if versions[name] != expected:
            errors.append(f"{name}: installed {versions[name]}, expected {expected}")
    if errors:
        raise ValueError("Environment differs from the final lock:\n" + "\n".join(errors))
    print(json.dumps({"python": platform.python_version(), "interpreter": sys.executable, "isolated": True, "checked_package_pins": len(versions)}, indent=2), flush=True)


def command_env():
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("PYTHONPATH", None)
    return env


def run_step(arguments, timeout, capture=False):
    print("\nRUN: " + subprocess.list2cmdline([str(x) for x in arguments]), flush=True)
    result = subprocess.run(
        [str(x) for x in arguments], cwd=ROOT, env=command_env(),
        text=True, encoding="utf-8", errors="replace", timeout=timeout,
        capture_output=capture,
    )
    if capture:
        print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode:
        raise RuntimeError(f"Check failed with exit code {result.returncode}")
    return result


def prediction_state(run_dir):
    """Retain both bytes and timestamps to prove that resume did not refit."""
    return {
        path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in sorted((Path(run_dir) / "predictions").glob("*.npz"))
    }


def source_consistency(run_dir, source_root=ROOT):
    """Check frozen bytes, accepting only Git-confirmed LF/CRLF differences."""
    source_root = Path(source_root).resolve()
    manifest = json.loads((Path(run_dir) / "run_manifest.json").read_text(encoding="utf-8"))
    commit = manifest["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("Invalid recorded source commit")
    normalized = 0
    files = manifest["identity"]["source_files"]
    for relative, expected_hash in files.items():
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError(f"Missing or invalid source path: {relative}")
        actual = path.read_bytes()
        if hashlib.sha256(actual).hexdigest() == expected_hash:
            continue
        try:
            reference = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=source_root, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, OSError) as exc:
            raise ValueError(f"Cannot confirm source against recorded commit: {relative}") from exc
        if actual.replace(b"\r\n", b"\n") != reference.replace(b"\r\n", b"\n"):
            raise ValueError(f"Source content changed: {relative}")
        normalized += 1
    result = {"checked_source_files": len(files), "eol_normalized_files": normalized}
    print("FROZEN SOURCE VERIFIED: " + json.dumps(result), flush=True)
    return result


def verify_artifacts(run_dir, timeout):
    source_consistency(run_dir)
    # Byte-level artifact hashes remain strict. The source check above also
    # works after a checkout converts LF/CRLF; the original verifier is intact.
    run_step([sys.executable, ROOT / "scripts/verify_v6_3_3_results.py", run_dir, "--skip-source"], timeout)


def compare_smoke_to_release(run_dir):
    import numpy as np

    checked, max_difference = 0, 0.0
    for path in sorted((Path(run_dir) / "predictions").glob("*.npz")):
        reference = RELEASE / "predictions" / path.name
        if not reference.is_file():
            raise ValueError(f"Published reference prediction is missing: {reference}")
        with np.load(path, allow_pickle=False) as actual, np.load(reference, allow_pickle=False) as expected:
            for field in ("test_indices", "y_true", "full", "partial", "scaler_max_abs"):
                np.testing.assert_array_equal(actual[field], expected[field], err_msg=f"{path.name}: {field}")
            np.testing.assert_allclose(actual["probabilities"], expected["probabilities"], rtol=1e-7, atol=1e-8, err_msg=path.name)
            max_difference = max(max_difference, float(np.max(np.abs(actual["probabilities"] - expected["probabilities"]))))
        checked += 1
    print(f"PUBLISHED RESULTS MATCH: {checked} prediction files; maximum probability difference={max_difference:.3g}.", flush=True)


def smoke(output_root, gpu=False, timeout=600):
    learners = ["logistic", "svm_calibrated", "mlp"] if gpu else ["logistic"]
    arguments = [sys.executable, ROOT / "scripts/run_v6_3_3_final.py", "--datasets", "viruspseaac", "--base-learners", *learners, "--folds", "1", "--output-root", Path(output_root).resolve()]
    first = run_step(arguments, timeout, capture=True)
    run_line = next((line for line in first.stdout.splitlines() if line.startswith("RUN_DIR=")), None)
    if run_line is None:
        raise ValueError("Smoke runner did not identify its output directory")
    run_dir = Path(run_line.split("=", 1)[1])
    before = prediction_state(run_dir)
    if len(before) != 4 * len(learners):
        raise ValueError("Smoke prediction set is incomplete")
    run_step(arguments, timeout, capture=True)
    if prediction_state(run_dir) != before:
        raise ValueError("Resume rewrote a completed prediction")
    verify_artifacts(run_dir, timeout)
    compare_smoke_to_release(run_dir)
    print(f"SMOKE AND RESUME PASSED: {len(before)} completed fits reused byte-for-byte.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "environment", "tests", "smoke", "verify"], default="all")
    parser.add_argument("--gpu-smoke", action="store_true", help="Smoke-test all three learners instead of Logistic only.")
    parser.add_argument("--run-dir", type=Path, default=RELEASE)
    parser.add_argument("--smoke-output-root", type=Path, default=ROOT / "scratch/final_checks/smoke")
    parser.add_argument("--timeout", type=int, default=600, help="Per-command timeout in seconds.")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    try:
        check_environment()
        run_step([sys.executable, "-m", "pip", "check"], args.timeout)
        if args.stage in ("all", "tests"):
            run_step([sys.executable, "-m", "pytest", "-c", ROOT / "pytest-final.ini", "-q", "--durations=5"], args.timeout)
        if args.stage in ("all", "smoke"):
            smoke(args.smoke_output_root, gpu=args.gpu_smoke, timeout=args.timeout)
        if args.stage in ("all", "verify"):
            verify_artifacts(args.run_dir.resolve(), args.timeout)
    except (AssertionError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"\nFINAL CHECK FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"\nFINAL CHECK PASSED: {args.stage}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
