from pathlib import Path

import numpy as np
import pytest

from examples.production_multi_track import load_tracks


def test_multi_track_example_has_explicit_valid_poses():
    path = Path(__file__).resolve().parents[1] / "examples/production_multi_track.toml"
    tracks = load_tracks(path)
    assert [identifier for identifier, _, _ in tracks] == [
        "track-cluster-1", "track-array-centre"]
    for _, _, source in tracks:
        midpoint = source.start_m + 0.5 * source.length_m * source.direction
        assert midpoint.shape == (3,)
        assert np.isfinite(midpoint).all()
        assert source.length_m == 120.
        assert np.linalg.norm(source.direction) == pytest.approx(1.)


def test_multi_track_example_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "tracks.toml"
    path.write_text("""
[[tracks]]
id = "same"
position_m = [0, 0, 0]
direction = [0, 0, 1]
length_m = 10
[[tracks]]
id = "same"
position_m = [1, 0, 0]
direction = [0, 1, 0]
length_m = 10
""", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate track id"):
        load_tracks(path)
