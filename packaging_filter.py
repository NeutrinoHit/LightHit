"""What goes into the distribution, in one place that both setup.py and a test import.

The package ships the experimental modules its public engines import at
runtime and no others. Other experimental reference, Monte-Carlo cross-check
and research modules stay in the repository; top-level reference modules are
not filtered.

Two things have to be true for that to hold, and only the first is obvious:

* the right modules must be *selected* --- and ``build_py.find_all_modules`` is
  not what selects them, ``find_package_modules`` is;
* the build tree must not keep modules from an earlier build. ``build_py``
  copies files into ``build/lib`` and never removes anything, so a checkout
  that once built without the filter keeps handing every stale module to the
  wheel. ``python -m build`` does not start from a clean tree, and a release
  process that depends on someone remembering ``rm -rf build`` is not a
  process. So the command prunes the tree it is about to hand over.
"""
from pathlib import Path
import shutil

from setuptools.command.build_py import build_py


PACKAGE = "lighthit"
FILTERED_SUBPACKAGE = "lighthit.experimental"

REQUIRED_EXPERIMENTAL = frozenset({
    "__init__", "axial_fast", "axial_source", "ballistic_fast",
    "event_moments", "g4_source", "hdf5_minimal",
})


def is_shipped(package, module):
    """Whether ``package.module`` belongs in the distribution."""
    if package != FILTERED_SUBPACKAGE:
        return True
    return module in REQUIRED_EXPERIMENTAL


class ProductionBuildPy(build_py):
    """``build_py`` that both filters what it copies and cleans what it finds."""

    def find_package_modules(self, package, package_dir):
        return [row for row in super().find_package_modules(package, package_dir)
                if is_shipped(package, row[1])]

    def find_all_modules(self):
        return [row for row in super().find_all_modules()
                if is_shipped(row[0], row[1])]

    def run(self):
        # Before, so that nothing stale survives into whatever reads build_lib
        # next; after, so that nothing this run produced slips past the filter.
        removed = self.prune_build_lib()
        super().run()
        self.pruned = removed + self.prune_build_lib()

    def prune_build_lib(self):
        """Delete modules in ``build_lib`` that this build would not produce.

        Returns the names removed, so that a test can assert on them rather
        than on the absence of something.
        """
        root = Path(self.build_lib) / Path(*PACKAGE.split("."))
        if not root.is_dir():
            return []
        wanted = {Path(path).resolve()
                  for path in self.get_outputs(include_bytecode=0)
                  if str(path).endswith(".py")}
        removed = []
        for path in sorted(root.rglob("*.py")):
            if path.resolve() not in wanted:
                path.unlink()
                removed.append(str(path.relative_to(root)))
        for cache in sorted(root.rglob("__pycache__")):
            shutil.rmtree(cache, ignore_errors=True)
        return removed
