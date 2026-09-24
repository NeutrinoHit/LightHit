# Releasing LightHit 0.2.0a8

Use the existing development Python to test code from the LightHit checkout.
Use the existing `lighthit-test` environment only after publication to check
the PyPI package. No new environment or branch is needed. The PyPI wheel
contains the package and synthetic demos, but not the repository-only BGVD
examples, private model, G4 data or book.

## Check the development checkout

Run from the LightHit repository root. The import path must point into its
`src/lighthit` directory, not `site-packages`:

```bash
cd /path/to/LightHit
python -c 'import sys, pathlib, lighthit as lh; p=pathlib.Path(lh.__file__).resolve(); print(sys.executable, lh.__version__, p); assert p.is_relative_to(pathlib.Path.cwd().resolve() / "src")'
```

Run and benchmark the desired examples with this Python. PyPI need not be
involved. Before the release, run the repository tests once. Render the book
when its source files changed, as they did for this release:

```bash
python -m pytest -q tests --g4-file g4_data/sim_e_100GeV_10.h5
quarto render docs
```

## Commit on main and publish

For this release, `fix/track-fast-path-a8` and `main` started at the same
commit. Switch to `main` before staging; review the staged list. The paths
below cover the package, examples, tests and release instructions, while
leaving separate book/notebook work unstaged.

```bash
git switch main
git add pyproject.toml README.md PUBLISHING.md src/lighthit examples tests \
  docs/appendices/g-pypi-isolated-test.qmd
git diff --cached --check
git diff --cached --stat
git status --short
git commit -m "Release LightHit 0.2.0a8"

python -m build --no-isolation --sdist --wheel --outdir dist
python -m twine check dist/lighthit-0.2.0a8-py3-none-any.whl \
  dist/lighthit-0.2.0a8.tar.gz
```

Only after the build and check succeed:

```bash
git tag -a v0.2.0a8 -m "LightHit 0.2.0a8"
git push origin main v0.2.0a8
python -m twine upload -u __token__ dist/lighthit-0.2.0a8-py3-none-any.whl \
  dist/lighthit-0.2.0a8.tar.gz
```

Stop if a test, build or check fails; do not upload incomplete artifacts. If
`build` or `twine` is missing in the development environment, install it once
with `python -m pip install build twine`. Twine prompts for the PyPI token;
never put that token in a command, file or Git history.

## Check the published package

In the existing `lighthit-test` environment, replace only LightHit, not its
large dependencies. `python -I` excludes the current directory from imports,
so the check works even while standing in the source checkout:

```bash
python -m pip install --no-deps --no-cache-dir --force-reinstall 'lighthit==0.2.0a8'
python -I -c 'import pathlib, lighthit as lh; p=pathlib.Path(lh.__file__).resolve(); print(lh.__version__, p); assert lh.__version__ == "0.2.0a8" and "site-packages" in p.parts'
lighthit-selftest
```
