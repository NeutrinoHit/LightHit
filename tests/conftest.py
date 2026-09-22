from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--g4-file", default=None,
        help="stored G4 HDF5 input for opt-in reader integration tests")


@pytest.fixture
def stored_g4_file(request):
    value = request.config.getoption("--g4-file")
    path = Path(value) if value else None
    if path is None or not path.is_file():
        pytest.skip("pass --g4-file PATH to run stored-event integration tests")
    return path
