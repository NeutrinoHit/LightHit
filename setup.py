"""Build entry point. The distribution filter lives in ``packaging_filter``.

That module sits beside this file rather than inside the package, because it
decides what the package contains and must not be part of it. Importing it
therefore needs this directory on ``sys.path``, and under PEP 517 it is not
there: ``pyproject_hooks`` runs the backend from a helper script of its own, so
``sys.path[0]`` is that helper's directory and not the project. A bare
``import packaging_filter`` works when setup.py is run directly, and fails
under ``python -m build``, which is the only invocation that matters for a
release. The two lines below remove the difference; nothing here depends on
``PYTHONPATH``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(
    globals().get("__file__", os.path.join(os.getcwd(), "setup.py")))))

from setuptools import setup  # noqa: E402

from packaging_filter import ProductionBuildPy  # noqa: E402


setup(cmdclass={"build_py": ProductionBuildPy})
