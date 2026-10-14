"""Several sources in one event (lighthit.event): one clock, one OM selection."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import lighthit as lh

pytest.importorskip("numba")
from lighthit.event import EventTransport, SourceComponent, common_time_origins  # noqa: E402
from lighthit.transport import HybridTransportResponse  # noqa: E402

from test_prompt import config, detector, spectral_medium  # noqa: E402


def shower(x):
    source = lh.SyntheticShower.gaussian(lh.SourcePose([x, 3., 0.], [0.3, 0.2, 0.93], 0.),
                                         charged_track_length_m=5., elements=30, seed=3)
    return replace(source.elements, cone_model="spectral")


@pytest.fixture
def kernel(tmp_path):
    return lh.TransportKernel(spectral_medium(), detector(count=4),
                              config(tmp_path, omega_per_ns=np.linspace(0., 1.2, 5),
                                     threshold_pe=1.0))


class FakeRoutes:
    """Route stand-in: fixed per-OM charges; honours extra charge and receivers."""

    def __init__(self, kernel, charges):
        self.kernel, self.charges, self.calls = kernel, charges, []

    def run(self, component, *, time_origin_ns=None, selection_extra_charge=None, receivers=None):
        q = self.charges[component.label]
        n_all = len(q)
        ids = np.arange(n_all) if receivers is None else np.asarray(receivers)
        self.calls.append(dict(label=component.label, receivers=None if receivers is None else ids.tolist(),
                               origins=np.asarray(time_origin_ns), extra=selection_extra_charge))
        q = q[ids]
        threshold = 0.0 if receivers is not None else self.kernel.config.threshold_pe
        extra = 0.0 if selection_extra_charge is None else np.asarray(selection_extra_charge)[ids]
        active = q.sum(1) + extra >= threshold
        nb = len(self.kernel.config.relative_time_edges_ns) - 1
        bins = np.zeros((len(ids), nb, 3))
        bins[:, 3, :] = np.where(active[:, None], q, 0.0)
        omega = self.kernel.config.omega_per_ns
        spectrum = np.repeat(q[:, 2][None, :], len(omega), axis=0).astype(complex)
        return HybridTransportResponse(None, omega, self.kernel.config.relative_time_edges_ns, spectrum,
                                       bins, q.copy(), np.asarray(time_origin_ns), active, "fake", {})


def transport_with(kernel, charges, monkeypatch):
    routes = FakeRoutes(kernel, charges)
    event = EventTransport(kernel)
    route = SimpleNamespace(kernel=kernel)
    monkeypatch.setattr(EventTransport, "shower_route", lambda self: route)
    monkeypatch.setattr(EventTransport, "_run", lambda self, component, **kw: routes.run(component, **kw))
    return event, routes


def test_om_active_only_in_the_sum_gets_both_sources(kernel, monkeypatch):
    # OM 0: 0.6 + 0.6 pe crosses 1 pe only together; OM 1 is bright in A only.
    charges = {"A": np.array([[0.4, 0.1, 0.1], [2.0, 0.5, 0.5], [0., 0., 0.], [0.01, 0., 0.]]),
               "B": np.array([[0.4, 0.1, 0.1], [0.0, 0.0, 0.0], [0., 0., 0.], [0.01, 0., 0.]])}
    event, routes = transport_with(kernel, charges, monkeypatch)
    response = event([SourceComponent("shower", shower(10.), label="A"),
                      SourceComponent("shower", shower(30.), label="B")])
    assert response.active.tolist() == [True, True, False, False]
    np.testing.assert_allclose(response.charge_components_pe, charges["A"] + charges["B"])
    np.testing.assert_allclose(response.components_pe[0, 3], (charges["A"] + charges["B"])[0])
    # A was run first without B's light, then completed on OM 0 only.
    completions = [call for call in routes.calls if call["receivers"] is not None]
    assert completions == [dict(label="A", receivers=[0], origins=completions[0]["origins"], extra=None)]
    origins = common_time_origins(kernel, [shower(10.), shower(30.)])
    for call in routes.calls:
        expected = origins if call["receivers"] is None else origins[call["receivers"]]
        np.testing.assert_array_equal(call["origins"], expected)
    assert response.metadata["route"] == "lighthit.event"
    assert response.metadata["completion_rounds"][0]["recomputed"] == {"A": 1}


def test_common_origin_is_the_earliest_of_all_sources(kernel):
    first, second = shower(10.), shower(30.)
    both = common_time_origins(kernel, [first, second])
    np.testing.assert_array_equal(both, np.minimum(common_time_origins(kernel, [first]),
                                                   common_time_origins(kernel, [second])))


def test_single_component_is_passed_through(kernel, monkeypatch):
    charges = {"A": np.array([[2.0, 0.5, 0.5]] * 4)}
    event, routes = transport_with(kernel, charges, monkeypatch)
    response = event([SourceComponent("shower", shower(10.), label="A")])
    assert routes.calls[0]["origins"].shape == () and routes.calls[0]["extra"] is None
    assert response.metadata["event_components"][0]["label"] == "A"


def test_component_and_flash_arguments_are_checked(kernel):
    with pytest.raises(ValueError):
        SourceComponent("muon", shower(10.))
    with pytest.raises(ValueError):
        SourceComponent("track", shower(10.))
    with pytest.raises(ValueError, match="laser_kernel"):
        EventTransport(kernel)(lh.IsotropicFlash.monochromatic([0., 0., 0.], 1e6, 450.))


def test_flash_goes_through_the_laser_kernel(tmp_path):
    from test_directional import directional_kernel
    laser = directional_kernel(tmp_path)
    flash = lh.IsotropicFlash.monochromatic([0., 0., 0.], 1e5, 450.)
    laser.build(source=flash)
    response = EventTransport(None, laser_kernel=laser)(flash)
    reference = laser.transport(flash)
    np.testing.assert_array_equal(response.charge_pe, reference.charge_pe)
    assert response.metadata["route"].startswith("lighthit.transport")
    with pytest.raises(ValueError, match="alone"):
        EventTransport(None, laser_kernel=laser)([flash, SourceComponent("shower", shower(10.))])


def test_om_out_of_reach_of_one_source_is_complete(kernel, monkeypatch):
    """A source with exactly zero charge at an OM (out of its range) needs no
    completion there; the OM is decided by the other source alone."""
    charges = {"A": np.array([[2.0, 0.5, 0.5], [0.0, 0.0, 0.0], [0.3, 0.1, 0.1], [0., 0., 0.]]),
               "B": np.array([[0.0, 0.0, 0.0], [1.5, 0.2, 0.1], [0.3, 0.1, 0.1], [0., 0., 0.]])}
    event, routes = transport_with(kernel, charges, monkeypatch)
    response = event([SourceComponent("shower", shower(10.), label="A"),
                      SourceComponent("shower", shower(30.), label="B")])
    assert response.active.tolist() == [True, True, True, False]
    completions = [(call["label"], call["receivers"]) for call in routes.calls
                   if call["receivers"] is not None]
    assert ("A", [1]) not in completions and ("B", [0]) not in completions
