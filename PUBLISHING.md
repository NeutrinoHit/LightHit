# Releasing LightHit

The package version has one source of truth:

```text
src/lighthit/_version.py
```

Before a release, update `VERSION` there. `pyproject.toml` reads the same
value dynamically; do not duplicate the current package version elsewhere.

Use the development Python environment to test the LightHit checkout.
Use the separate `lighthit-test` environment only after publication to verify
the package installed from PyPI.

The PyPI distribution contains the package and synthetic demos, but not the
repository-only BGVD examples, private model, G4 data, research scripts, or
documentation sources.


## 1. Check the development checkout

Run from the LightHit repository root.

Verify that Python imports LightHit from the checkout rather than from
`site-packages`:

```bash
python -c 'import sys, pathlib, lighthit as lh; p=pathlib.Path(lh.__file__).resolve(); print(sys.executable, lh.__version__, p); assert p.is_relative_to(pathlib.Path.cwd().resolve() / "src")'
```

Read the release version from the package:

```bash
LH_VERSION="$(python -c 'from lighthit._version import VERSION; print(VERSION)')"
echo "$LH_VERSION"
```

Run the full repository test suite, including the stored G4 integration tests:

```bash
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python -m pytest -q tests \
  --g4-file g4_data/sim_e_100GeV_10.h5
```

If documentation sources changed, render them separately:

```bash
quarto render docs
```

Stop if any test or documentation build fails.


## 2. Commit the tested feature branch

Review the working tree and staged diff:

```bash
git status --short
git diff --cached --check
git diff --cached --stat
```

Commit the tested feature if it has not yet been committed:

```bash
git commit -m "Add direct prompt transport for scattering orders 0 and 1"
```

Record the current feature branch name before switching away from it:

```bash
FEATURE_BRANCH="$(git branch --show-current)"
echo "$FEATURE_BRANCH"
test "$FEATURE_BRANCH" != "main"
```

Push the feature branch:

```bash
git push -u origin "$FEATURE_BRANCH"
```


## 3. Merge the tested feature branch into main

Update `main` and fast-forward it to the tested feature branch:

```bash
git switch main
git pull --ff-only origin main
git merge --ff-only "$FEATURE_BRANCH"
```

Check the resulting history and working tree:

```bash
git log --oneline --decorate -5
git status
```

Push `main`:

```bash
git push origin main
```


## 4. Build the release artifacts

Make sure `main` is clean:

```bash
git status --short
```

The command above should print nothing.

Read the version again from the package after switching to `main`:

```bash
LH_VERSION="$(python -c 'from lighthit._version import VERSION; print(VERSION)')"
echo "$LH_VERSION"
```

Remove previous build products and build the source distribution and wheel:

```bash
rm -rf dist build
python -m build --sdist --wheel --outdir dist
```

The expected artifacts are:

```text
dist/lighthit-${LH_VERSION}-py3-none-any.whl
dist/lighthit-${LH_VERSION}.tar.gz
```

Check both artifacts with Twine:

```bash
python -m twine check \
  "dist/lighthit-${LH_VERSION}-py3-none-any.whl" \
  "dist/lighthit-${LH_VERSION}.tar.gz"
```

Stop if the build or Twine check fails.

If `build` or `twine` is missing in the development environment, install it
once:

```bash
python -m pip install build twine
```


## 5. Tag and publish

Only after the tests, build, and Twine check succeed, create the annotated
release tag:

```bash
git tag -a "v${LH_VERSION}" -m "LightHit ${LH_VERSION}"
git push origin "v${LH_VERSION}"
```

Upload the same artifacts to PyPI:

```bash
python -m twine upload -u __token__ \
  "dist/lighthit-${LH_VERSION}-py3-none-any.whl" \
  "dist/lighthit-${LH_VERSION}.tar.gz"
```

Twine prompts for the PyPI token. Never put the token itself in a command,
file, shell history, or Git history.


## 6. Check the published package

Use the existing `lighthit-test` environment for this step.

Set the expected version from the release tag:

```bash
LH_VERSION="$(git describe --tags --exact-match --match 'v*' | sed 's/^v//')"
echo "$LH_VERSION"
```

Install exactly that version from PyPI without reinstalling the large
dependencies:

```bash
python -m pip install \
  --no-deps \
  --no-cache-dir \
  --force-reinstall \
  "lighthit==${LH_VERSION}"
```

Use isolated Python mode so that the current source checkout cannot satisfy
the import accidentally:

```bash
python -I -c "import pathlib, lighthit as lh; p=pathlib.Path(lh.__file__).resolve(); print(lh.__version__, p); assert lh.__version__ == '${LH_VERSION}' and 'site-packages' in p.parts"
```

Finally run the installed-package self-test:

```bash
lighthit-selftest
```

The release is complete only after these checks pass.
