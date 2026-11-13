"""Build and verify a compact, byte-preserving v6.3.3 final package."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path("results/v6_3_3_final_four_models/run_6c5fb2bebb16a7dc")
PACKAGE = "BR_CC-v6.3.3-final-four-models"
MANIFEST = "PACKAGE_MANIFEST.json"
CHECKSUMS = "PACKAGE_CHECKSUMS.sha256"
FIXED_TIME = (2026, 10, 28, 0, 0, 0)
SUPPORT = (
    ".gitattributes", ".gitignore", "README.md", "pytest-final.ini",
    "requirements-final.lock.txt", "requirements-final-env.lock.txt",
    "data/README.md", "data/dataset_manifest.csv", "scripts/__init__.py",
    "scripts/setup_final_env.ps1", "scripts/check_final.py",
    "scripts/package_final.py", "tests/__init__.py",
    "scripts/run_experiments.py", "scripts/validate_datasets.py",
    "scripts/verify_results.py",
    "tests/fixtures/__init__.py", "tests/fixtures/metric_cases.py",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    """Accept portable relative paths, excluding traversal and Windows drives."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or "\\" in name or ":" in name or any(part in (".", "..") for part in name.split("/")):
        raise ValueError(f"Invalid package path: {name}")
    return path


def read_checksums(data):
    records = {}
    for line in data.decode("utf-8").splitlines():
        digest, name = line.split("  ", 1)
        safe_path(name)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or name in records:
            raise ValueError(f"Invalid or duplicate checksum: {name}")
        records[name] = digest
    if not records:
        raise ValueError("Checksum inventory is empty")
    return records


def payload(root=ROOT, include_data=True):
    """Use frozen inventories and explicit workflow files, never a directory dump."""
    root = Path(root).resolve()
    release = json.loads((root / RELEASE / "run_manifest.json").read_text(encoding="utf-8"))
    if release["status"] != "complete" or release["completed_cells"] != 600:
        raise ValueError("The reference must contain all 600 completed fits")
    data = json.loads((root / "configs/datasets_v6_3_3.json").read_text(encoding="utf-8"))
    frozen = release["identity"]["source_files"]
    names = set(frozen) | set(SUPPORT)
    config = configparser.ConfigParser()
    config.read(root / "pytest-final.ini", encoding="utf-8")
    names.update(config["pytest"]["testpaths"].split())
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "CITATION.cff"):
        if (root / name).is_file():
            names.add(name)

    result_records = read_checksums((root / RELEASE / "checksums.sha256").read_bytes())
    names.update((RELEASE / name).as_posix() for name in result_records)
    names.add((RELEASE / "checksums.sha256").as_posix())
    raw = {item["path"]: item for spec in data["datasets"].values() for item in spec["files"]}
    if include_data:
        names.update(raw)
        # Keep upstream README/license/citation files beside their original inputs.
        for parent in {Path(name).parent for name in raw} - {Path("data")}:
            for path in (root / parent).iterdir():
                if path.is_file() and any(word in path.name.lower() for word in ("readme", "license", "licence", "copying", "citation")):
                    names.add(path.relative_to(root).as_posix())

    records = {}
    for name in sorted(names):
        safe_path(name)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or invalid input: {name}")
        content = path.read_bytes()
        digest = sha256(content)
        if name in frozen and digest != frozen[name]:
            raise ValueError(f"Frozen source bytes differ: {name}. Use the byte-preserved final snapshot.")
        relative_result = Path(name).relative_to(RELEASE).as_posix() if Path(name).is_relative_to(RELEASE) else None
        if relative_result in result_records and digest != result_records[relative_result]:
            raise ValueError(f"Reference result checksum differs: {name}")
        if name in raw and (digest != raw[name]["sha256"] or len(content) != raw[name]["bytes"]):
            raise ValueError(f"Raw dataset checksum or size differs: {name}")
        records[name] = {"sha256": digest, "bytes": len(content)}
    return records, release, len(raw) if include_data else 0


def verify_package(path):
    """Verify the complete inventory of a ZIP or an untouched package directory."""
    path = Path(path)
    archive = zipfile.ZipFile(path) if path.is_file() else None
    try:
        if archive:
            members = archive.namelist()
            prefix = PACKAGE + "/"
            if len(members) != len(set(members)) or any(not name.startswith(prefix) for name in members):
                raise ValueError("Invalid or duplicate ZIP members")
            names = {name[len(prefix):] for name in members}
            read = lambda name: archive.read(prefix + name)
        else:
            names = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
            read = lambda name: (path / name).read_bytes()
        for name in names:
            safe_path(name)
        expected = read_checksums(read(CHECKSUMS))
        if names != set(expected) | {CHECKSUMS}:
            raise ValueError("Package inventory has missing or unexpected files")
        for name, digest in expected.items():
            if sha256(read(name)) != digest:
                raise ValueError(f"Package checksum differs: {name}")
        manifest = json.loads(read(MANIFEST))
        if manifest["file_count"] != len(names):
            raise ValueError("Package file count differs")
        result = {"package": str(path.resolve()), "verified_files": len(names), "raw_data_included": manifest["raw_data_included"], "status": "verified"}
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        if archive:
            archive.close()


def build(output_root, include_data=True):
    records, release, raw_count = payload(ROOT, include_data=include_data)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output([
        "git", "status", "--porcelain", "--untracked-files=all", "--",
        "src", "configs", "scripts", "tests", "README.md", ".gitignore",
        ".gitattributes", "pytest-final.ini", "requirements-final.lock.txt",
        "requirements-final-env.lock.txt", "data",
    ], cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError("Commit project changes before packaging so the package records a clean source commit")
    output_root = Path(output_root).resolve()
    if not output_root.is_relative_to(ROOT / "dist"):
        raise ValueError("Choose an output directory under this project's dist directory")
    destination = output_root / PACKAGE
    archive_path = output_root / (PACKAGE + ".zip")
    sidecar = output_root / (PACKAGE + ".zip.sha256")
    if any(path.exists() for path in (destination, archive_path, sidecar)):
        raise ValueError("Output already exists; choose a new --output-root to preserve the previous package")
    manifest = {
        "schema_version": 1, "package": PACKAGE, "packaging_commit": commit,
        "benchmark_source_commit": release["source_commit"],
        "benchmark_run_id": release["run_id"], "variant": "standard_no_samg",
        "frozen_source_files": len(release["identity"]["source_files"]),
        "benchmark_fits": release["completed_cells"], "dataset_count": 10,
        "raw_data_included": include_data, "raw_input_files": raw_count,
        "code_license_file": next((name for name in records if Path(name).name.lower() in ("license", "license.txt", "license.md") and "/" not in name), None),
        "file_count": len(records) + 2,
    }
    destination.mkdir(parents=True)
    for name, record in records.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if sha256(target.read_bytes()) != record["sha256"]:
            raise ValueError(f"Input changed during packaging: {name}")
    (destination / MANIFEST).write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    checksums = {name: record["sha256"] for name, record in records.items()}
    checksums[MANIFEST] = sha256((destination / MANIFEST).read_bytes())
    (destination / CHECKSUMS).write_bytes(("\n".join(f"{digest}  {name}" for name, digest in sorted(checksums.items())) + "\n").encode("utf-8"))
    verify_package(destination)
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted([*checksums, CHECKSUMS]):
            info = zipfile.ZipInfo(f"{PACKAGE}/{name}", FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (destination / name).read_bytes(), compresslevel=9)
    verify_package(archive_path)
    digest = sha256(archive_path.read_bytes())
    sidecar.write_bytes(f"{digest}  {archive_path.name}\n".encode("utf-8"))
    print(f"PACKAGE_DIR={destination}\nPACKAGE_ZIP={archive_path}\nZIP_SHA256={digest}", flush=True)
    return archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "dist")
    parser.add_argument("--without-data", action="store_true", help="Omit raw input files; retain dataset metadata.")
    parser.add_argument("--verify", type=Path, help="Only verify an existing ZIP or untouched package folder.")
    args = parser.parse_args()
    try:
        if args.verify:
            verify_package(args.verify)
        else:
            build(args.output_root, include_data=not args.without_data)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        print(f"PACKAGE FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
