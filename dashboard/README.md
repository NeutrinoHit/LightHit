# LightHit results page

The standalone English results page presents stored optical-module responses,
reference comparisons, scientific checks and the full idea registry. It does
not run transport in the browser or treat a published page as release acceptance.

Build and check it from the repository root:

```sh
python -m unittest discover -s dashboard -p 'test_*.py'
python dashboard/build.py --output _site  # output directory must be absent or empty
python -m http.server --directory _site 8000
```

GitHub Pages builds the same files through `pages.yml`. The root opens the
latest published report; `versions/<report-id>/` opens a named archived report.
The Versions tab also lists package versions from the recorded public PyPI
catalog. Package releases and scientific measurement reports remain separate.

## Published reports

`reports/index.json` records the report IDs, dates and SHA256 checksums.
Published JSON reports are small, intentional archival records: reduced arrays,
measured numbers, declared registry metadata and source checksums. These are a
documented exception to regenerating build outputs; the original calculation
may require inputs that do not belong in this repository. Existing reports
are not rewritten. Add a new ID and checksum for a new publication.

The first report (v1) contains historical measurements and an idea-registry snapshot.
Many source records retain only a base commit and a dirty flag, so the page
does not assign them to the newest package release or certify reproducibility
of that release. Missing references or missing test reports are shown explicitly.

The second report (v2) shows the detector examples on the glass tables of the
0.3.0a6 kit (`<case>@glass` records; light counted at the first entry into the
OM glass). The centre-referenced records stay in the registry; where a stored
reference counts light at the OM centre, the difference is shown as a change of
receiver definition, not as an error. The laser table is the same file in both
data sets, so the laser example is identical to v1. v2 adds the kit acceptance
(`check.py`: numerical reproduction and photon MC by distance, with the
remaining flags), the half-space reference of `docs/halfspace.md`, matched
cold/JIT/warm timing and the laser photon Monte Carlo with its time to the
precision reached.

Raw detector/calibration inputs, full response bundles, runtime caches, build
checkpoints and videos stay outside the repository. The report includes no
detector geometry or calibration tables. Films use the existing media library.

## Exporting a new report

With the recorded inputs/results available locally and NumPy installed:

```sh
python dashboard/export_results.py --output new-report.json
# Optional separate atlas-component report:
python dashboard/export_results.py --output another-report.json \
  --atlas-report path/to/atlas-checks.json
# The second report: kit check, half-space run, timing
python dashboard/export_results.py --output dashboard/reports/v2.json --publication v2 \
  --kit-check KIT-CHECK.json --kit-bundle KIT/runtime/lighthit-glass-data \
  --halfspace HALFSPACE-RESULTS --timing TIMING.json
```

`KIT-CHECK.json` is the report of the kit's `check.py`, `HALFSPACE-RESULTS` an
output directory of `experiments/reproduce_halfspace.py` whose source
checksums match the checkout, and `TIMING.json` an output of
`experiments/time_routes.py`. Only reduced numbers enter the report: per
distance-group ratios, totals, checksums and timings.

The exporter reads all idea files and stored case records, keeps collision
orders separate, reduces representative response arrays and uses the existing
metric implementation for charge comparisons. It runs no transport and refuses
to replace an existing output. Review the public data boundary before adding
an exported report to the catalog. Registration and publication never imply
that a scientific criterion passed.

`explore-results.ipynb` reads the same `results.json` offered by the page.
Download both into one directory to inspect the saved arrays in Jupyter.
The notebook is an alternative reader, not a second transport implementation.

Automatic measurement-run ingestion is separate follow-up work. Timing
comparisons are taken only from `time_routes.py` (cold setup, JIT compilation
and warm event response separately, fixed threads, one machine) and, for the
laser, from the photon Monte Carlo of the same flash with its achieved
precision; registry wall times are not a matched benchmark.

The site serves D3 7.9.0 and Lucide 0.468.0 from `vendor/`, with their respective
licenses. It does not need an external script CDN to display the archived plots.
