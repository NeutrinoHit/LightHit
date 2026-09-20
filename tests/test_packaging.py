"""What the distribution contains, and what a stale build tree must not add.

Two failures happened here and neither showed up as an error. Overriding
``build_py.find_all_modules`` alone filtered nothing, because
``find_package_modules`` is what selects the files. And once that was fixed,
``python -m build`` still shipped every experimental module in a checkout whose
``build/lib`` predated the filter, because ``build_py`` copies and never
removes. Both are asserted below on a synthetic tree, so the test is fast and
says which of the two broke.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from setuptools.dist import Distribution

ROOT = Path(__file__).resolve().parents[1]
# packaging_filter lives beside setup.py, not inside the package: it decides
# what the package contains and must not be part of it.
sys.path.insert(0, str(ROOT))

from packaging_filter import (FILTERED_SUBPACKAGE, PACKAGE,  # noqa: E402
                              REQUIRED_EXPERIMENTAL, ProductionBuildPy,
                              is_shipped)


def synthetic_tree(root, modules):
    package = root / "src" / "lighthit"
    (package / "experimental").mkdir(parents=True)
    (package / "__init__.py").write_text("VERSION = '0'\n")
    (package / "transport.py").write_text("pass\n")
    for name in modules:
        (package / "experimental" / f"{name}.py").write_text("pass\n")
    return package


def run_build(root, build_lib):
    distribution = Distribution({
        "name": "lighthit", "version": "0",
        "packages": ["lighthit", "lighthit.experimental"],
        "package_dir": {"": "src"},
    })
    distribution.script_name = "setup.py"
    command = ProductionBuildPy(distribution)
    command.build_lib = str(build_lib)
    command.ensure_finalized()
    command.run()
    return command


def test_is_shipped_only_filters_the_experimental_subpackage():
    assert is_shipped(PACKAGE, "transport")
    assert is_shipped(PACKAGE, "directional")
    assert is_shipped(FILTERED_SUBPACKAGE, "axial_fast")
    assert not is_shipped(FILTERED_SUBPACKAGE, "directional_reference")
    assert not is_shipped(FILTERED_SUBPACKAGE, "signal_screen")


def test_the_build_ships_only_the_allowed_experimental_modules(tmp_path, monkeypatch):
    shipped = sorted(REQUIRED_EXPERIMENTAL)
    withheld = ["directional_reference", "signal_screen", "stationary_modes"]
    synthetic_tree(tmp_path, shipped + withheld)
    monkeypatch.chdir(tmp_path)
    build_lib = tmp_path / "build" / "lib"
    run_build(tmp_path, build_lib)
    built = sorted(path.stem for path in
                   (build_lib / "lighthit" / "experimental").glob("*.py"))
    assert built == shipped


def test_the_build_removes_modules_an_earlier_build_left_behind(tmp_path, monkeypatch):
    """``rm -rf build`` is not a release process; the command does it itself."""
    shipped = sorted(REQUIRED_EXPERIMENTAL)
    synthetic_tree(tmp_path, shipped + ["directional_reference"])
    monkeypatch.chdir(tmp_path)
    build_lib = tmp_path / "build" / "lib"
    stale = build_lib / "lighthit" / "experimental"
    stale.mkdir(parents=True)
    # exactly the situation that shipped seventeen modules: a tree built before
    # the filter existed, plus a module that has since been deleted outright
    for name in ("directional_reference", "signal_screen", "gone_entirely"):
        (stale / f"{name}.py").write_text("pass\n")
    cached = stale / "__pycache__"
    cached.mkdir()
    (cached / "signal_screen.cpython-311.pyc").write_bytes(b"\x00")

    command = run_build(tmp_path, build_lib)

    built = sorted(path.stem for path in stale.glob("*.py"))
    assert built == shipped
    assert set(command.pruned) >= {"experimental/directional_reference.py",
                                   "experimental/signal_screen.py",
                                   "experimental/gone_entirely.py"}
    assert not cached.exists()


def test_the_manifest_allowlist_matches_the_build_filter():
    """The sdist and the wheel must agree on what 'necessary' means."""
    manifest = (ROOT / "MANIFEST.in").read_text().splitlines()
    prefix = "include src/lighthit/experimental/"
    listed = {line[len(prefix):].removesuffix(".py") for line in manifest
              if line.startswith(prefix)}
    assert listed == set(REQUIRED_EXPERIMENTAL)
    assert "prune src/lighthit/experimental" in manifest


def test_the_release_metadata_is_present_and_consistent():
    import lighthit
    pyproject = (ROOT / "pyproject.toml").read_text()
    version = next(line.split("=", 1)[1].strip().strip('"')
                   for line in pyproject.splitlines()
                   if line.startswith("version ="))
    assert version == lighthit.__version__
    assert 'license = {text = "BSD-3-Clause"}' in pyproject
    assert "License :: OSI Approved :: BSD License" in pyproject
    licence = (ROOT / "LICENSE").read_text()
    assert licence.startswith("BSD 3-Clause License")
    assert "Dmitry Naumov" in licence
    checklist = (ROOT / "PUBLISHING.md").read_text()
    assert version in checklist


# ---------------------------------------------------------------- end to end

PROJECT_FILES = ("setup.py", "packaging_filter.py", "MANIFEST.in",
                 "pyproject.toml", "README.md", "LICENSE")
PROJECT_TREES = ("src", "examples")
FORBIDDEN_IN_ARCHIVES = ("g4_data", "bgvd_model", "docs/", "notebooks/",
                         "slides/", "scripts/", "tests/", ".build",
                         "results/", "preview/")


def project_copy(destination):
    """The files a build sees, without the repository around them."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in PROJECT_FILES:
        shutil.copy2(ROOT / name, destination / name)
    for name in PROJECT_TREES:
        shutil.copytree(ROOT / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return destination


def dirty_build_tree(project):
    """A ``build/lib`` from before the filter existed, as a real checkout has."""
    stale = project / "build" / "lib" / "lighthit" / "experimental"
    stale.mkdir(parents=True)
    for source in (ROOT / "src" / "lighthit" / "experimental").glob("*.py"):
        shutil.copy2(source, stale / source.name)
    return stale


def clean_environment():
    """Everything the build needs and nothing that would paper over an import."""
    environment = {key: value for key, value in os.environ.items()
                   if key != "PYTHONPATH"}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def test_the_backend_imports_the_filter_without_pythonpath(tmp_path):
    """PEP 517 does not put the project on ``sys.path``; setup.py must.

    ``pyproject_hooks`` runs the backend from a helper script of its own, so
    ``sys.path[0]`` is that helper's directory. A bare
    ``import packaging_filter`` in setup.py works when setup.py is run by hand
    and fails under ``python -m build`` --- the only invocation that matters
    for a release. This reproduces that exactly, and no environment variable
    is allowed to rescue it.
    """
    project = project_copy(tmp_path / "project")
    (project / "dist").mkdir()
    runner = tmp_path / "elsewhere"
    runner.mkdir()
    (runner / "run_hook.py").write_text(
        "import os, sys\n"
        "os.chdir(sys.argv[1])\n"
        "from setuptools import build_meta\n"
        "print(build_meta.build_sdist('dist'))\n")
    finished = subprocess.run(
        [sys.executable, str(runner / "run_hook.py"), str(project)],
        cwd=str(runner), env=clean_environment(), capture_output=True, text=True)
    assert finished.returncode == 0, finished.stderr
    assert "packaging_filter" not in finished.stderr
    assert list(project.glob("dist/*.tar.gz"))


@pytest.mark.skipif(importlib.util.find_spec("build") is None,
                    reason="the 'build' frontend is not installed")
def test_python_m_build_produces_clean_archives(tmp_path):
    """The command the release checklist actually runs, on a dirty tree."""
    project = project_copy(tmp_path / "project")
    dirty_build_tree(project)
    # Put a real file in every forbidden top-level tree. Absence from an archive
    # is otherwise vacuous because project_copy deliberately starts small.
    for directory in ("g4_data", "bgvd_model", "docs", "notebooks", "slides",
                      "scripts", "tests", ".build", "results", "preview"):
        path = project / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / "must-not-ship.txt").write_text("private or internal\n")
    finished = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--sdist", "--wheel"],
        cwd=str(project), env=clean_environment(), capture_output=True,
        text=True, timeout=900)
    assert finished.returncode == 0, finished.stdout + finished.stderr

    wheels = list(project.glob("dist/*.whl"))
    sdists = list(project.glob("dist/*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        top_level = archive.read(
            next(name for name in names if name.endswith(".dist-info/top_level.txt"))
        ).decode().strip()
        metadata = archive.read(
            next(name for name in names if name.endswith(".dist-info/METADATA"))
        ).decode()
    experimental = sorted(Path(name).stem for name in names
                          if "lighthit/experimental/" in name
                          and name.endswith(".py"))
    assert experimental == sorted(REQUIRED_EXPERIMENTAL)
    # setuptools>=77 follows PEP 639 and uses ``licenses/LICENSE``; older
    # supported releases put the same file directly in dist-info.
    assert any(name.endswith(".dist-info/LICENSE")
               or name.endswith(".dist-info/licenses/LICENSE")
               for name in names), names
    assert top_level == "lighthit"
    assert "License: BSD-3-Clause" in metadata
    assert "Classifier: License :: OSI Approved :: BSD License" in metadata
    assert not [name for name in names if "packaging_filter" in name]
    assert not [name for name in names
                if any(bad in name for bad in FORBIDDEN_IN_ARCHIVES)]

    with tarfile.open(sdists[0]) as archive:
        inside = [name.split("/", 1)[1] for name in archive.getnames()
                  if "/" in name]
    experimental = sorted(Path(name).stem for name in inside
                          if "experimental/" in name and name.endswith(".py"))
    assert experimental == sorted(REQUIRED_EXPERIMENTAL)
    assert {"LICENSE", "setup.py", "packaging_filter.py"} <= set(inside)
    assert not [name for name in inside
                if any(bad in name for bad in FORBIDDEN_IN_ARCHIVES)]
