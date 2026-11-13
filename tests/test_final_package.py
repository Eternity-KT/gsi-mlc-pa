import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts import package_final as package


@pytest.mark.parametrize("autocrlf", ["true", "false"])
def test_git_checkout_preserves_frozen_source_bytes(tmp_path, autocrlf):
    """A fresh Windows or LF checkout must reproduce the P0 byte hashes."""
    manifest = json.loads((package.ROOT / package.RELEASE / "run_manifest.json").read_text(encoding="utf-8"))
    frozen = manifest["identity"]["source_files"]
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "-c", f"core.autocrlf={autocrlf}", "checkout-index", "--prefix=" + checkout.as_posix() + "/", "-z", "--stdin"],
        input=("\0".join(frozen) + "\0").encode("utf-8"), cwd=package.ROOT, check=True,
        capture_output=True,
    )
    for name, expected in frozen.items():
        assert package.sha256((checkout / name).read_bytes()) == expected, name


def test_final_inventory_contains_only_required_sources_data_and_results():
    records, release, raw_count = package.payload()
    assert raw_count == 16
    assert set(release["identity"]["source_files"]).issubset(records)
    assert sum(name.endswith(".npz") for name in records) == 600
    assert "data/CHD_49/readme.txt" in records
    assert "data/HumanPseAAC/HumanPseAAC3106-ReadMe.txt" in records
    assert "scripts/package_final.py" in records
    assert "scripts/run_experiments.py" in records
    assert "scripts/validate_datasets.py" in records
    assert "scripts/verify_results.py" in records
    assert "tests/test_final_package.py" in records
    assert "src/models/gsi_v6_3_3.py" in records
    assert not any(f"src/models/gsi_v6_{version}.py" in records for version in ("1", "1_1", "2", "3", "3_1", "3_2"))
    for name in records:
        assert not name.startswith(("papers/", "draft_end/", "docs/", "spec/", "scratch/", ".venv", "TaiLieuThamKhao/", "results_v", "results_pa_"))
        assert not name.endswith((".pdf", ".tex", ".pyc"))
    assert "data/emotions/emotions-train.arff" not in records
    assert "data/CAL500.arff" not in records
    assert "main.py" not in records
    assert "scripts/run_v6_3_3_experiment.py" not in records


def test_code_only_inventory_keeps_metadata_and_omits_raw_data():
    records, _, raw_count = package.payload(include_data=False)
    assert raw_count == 0
    assert {name for name in records if name.startswith("data/")} == {"data/README.md", "data/dataset_manifest.csv"}


@pytest.fixture
def tiny_project(tmp_path):
    def write(name, content):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    for name in package.SUPPORT:
        write(name, b"support file\n")
    write("pytest-final.ini", b"[pytest]\ntestpaths = tests/test_model.py\n")
    write("tests/test_model.py", b"def test_model(): pass\n")
    write("src/model.py", b"value = 1\r\n")
    write("data/raw.txt", b"raw source\r\n")
    write("configs/datasets_v6_3_3.json", json.dumps({"datasets": {"tiny": {"files": [{
        "path": "data/raw.txt", "bytes": 12, "sha256": package.sha256(b"raw source\r\n"),
    }]}}}).encode())
    release = {"status": "complete", "completed_cells": 600, "identity": {"source_files": {
        "src/model.py": package.sha256(b"value = 1\r\n"),
    }}}
    write((package.RELEASE / "run_manifest.json").as_posix(), json.dumps(release).encode())
    write((package.RELEASE / "predictions/test.npz").as_posix(), b"prediction bytes")
    entries = ["run_manifest.json", "predictions/test.npz"]
    content = "\n".join(f"{package.sha256((tmp_path / package.RELEASE / name).read_bytes())}  {name}" for name in entries) + "\n"
    write((package.RELEASE / "checksums.sha256").as_posix(), content.encode())
    return tmp_path


@pytest.mark.parametrize("relative, message", [
    ("src/model.py", "Frozen source bytes differ"),
    ("data/raw.txt", "Raw dataset checksum or size differs"),
    ((package.RELEASE / "predictions/test.npz").as_posix(), "Reference result checksum differs"),
])
def test_packaging_rejects_changed_frozen_input(tiny_project, relative, message):
    package.payload(tiny_project)
    (tiny_project / relative).write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match=message):
        package.payload(tiny_project)


def small_archive(tmp_path, damage=None):
    entries = {
        "README.md": b"readme\n",
        package.MANIFEST: json.dumps({"file_count": 3, "raw_data_included": False}).encode(),
    }
    checksums = "\n".join(f"{package.sha256(content)}  {name}" for name, content in sorted(entries.items())) + "\n"
    entries[package.CHECKSUMS] = checksums.encode()
    if damage == "changed":
        entries["README.md"] = b"different readme\n"
    elif damage == "extra":
        entries["papers/draft.pdf"] = b"unwanted draft"
    elif damage == "missing":
        del entries["README.md"]
    elif damage == "traversal":
        entries["../outside.txt"] = b"invalid path"
    archive_path = tmp_path / "package.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(package.PACKAGE + "/" + name, content)
    return archive_path


def test_zip_is_verified_without_git_or_third_party_packages(tmp_path):
    assert package.verify_package(small_archive(tmp_path))["verified_files"] == 3


@pytest.mark.parametrize("damage, message", [
    ("changed", "Package checksum differs"),
    ("extra", "Package inventory has missing or unexpected files"),
    ("missing", "Package inventory has missing or unexpected files"),
    ("traversal", "Invalid package path"),
])
def test_archive_corruption_and_unexpected_content_fail(tmp_path, damage, message):
    with pytest.raises(ValueError, match=message):
        package.verify_package(small_archive(tmp_path, damage))


def test_existing_package_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "ROOT", tmp_path)
    monkeypatch.setattr(package, "payload", lambda *args, **kwargs: ({}, {}, 0))
    monkeypatch.setattr(package.subprocess, "check_output", lambda args, **kwargs: "a" * 40 if "rev-parse" in args else "")
    destination = tmp_path / "dist" / package.PACKAGE
    destination.mkdir(parents=True)
    previous = destination / "README.md"
    previous.write_bytes(b"previous package")
    with pytest.raises(ValueError, match="Output already exists"):
        package.build(tmp_path / "dist")
    assert previous.read_bytes() == b"previous package"
