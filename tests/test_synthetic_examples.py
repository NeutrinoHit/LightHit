import json

import numpy as np

import lighthit as lh
from lighthit.examples import synthetic


def test_baikal_like_example_is_public_synthetic_geometry(capsys):
    detector = synthetic.baikal_like_detector(clusters=2, strings=3, modules=4)
    assert len(detector) == 24
    assert detector.provenance.startswith("synthetic Baikal-like")
    assert set(lh.describe_geometry(detector).clusters) == {0, 1}
    synthetic.main(["geometry", "--clusters", "1"])
    report = json.loads(capsys.readouterr().out)
    assert report["array"]["modules"] == 48


def test_all_installed_example_source_kinds_have_explicit_geometry():
    detector = synthetic.baikal_like_detector(strings=2, modules=3)
    sources = synthetic.example_sources(detector)
    assert set(sources) == {"laser", "track", "shower"}
    assert isinstance(sources["laser"], lh.IsotropicFlash)
    assert isinstance(sources["track"], lh.CherenkovTrack)
    assert isinstance(sources["shower"], lh.SyntheticShower)
    centre = lh.describe_geometry(detector).centroid_m
    assert np.linalg.norm(sources["shower"].centroid_m - centre) < 20


def test_synthetic_shower_runs_through_the_public_axial_route(tmp_path):
    detector = synthetic.baikal_like_detector(strings=2, modules=2,
                                               string_radius_m=12.)
    source = synthetic.example_sources(detector)["shower"]
    config = lh.KernelConfig(
        omega_per_ns=np.array([0., .04]),
        relative_time_edges_ns=np.arange(-20., 61., 10.),
        wavelength_nodes=2, scattering_degree=4, source_degree=4,
        azimuthal_degree=1, cell_m=1., k_max_per_m=2., k_panel_per_m=.2,
        k_order=4, radial_range_m=(2., 80.), radial_nodes=8,
        angular_backend="numpy", threshold_pe=0,
        cache_directory=tmp_path)
    response = lh.TransportKernel(
        synthetic.synthetic_medium_model(), detector, config).transport(source)
    assert response.method == "axial"
    assert response.components_pe.shape == (4, 8, 3)
    assert np.isfinite(response.charge_components_pe).all()
