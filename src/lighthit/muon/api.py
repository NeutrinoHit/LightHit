"""Muon events to expected photoelectrons, all scattering orders.

This is the route used by default for G4 muon tracks: exact direct light from
the original steps, finite-track atlases for the primary, grouped physical-time
first scattering of all secondaries with the spectral bands fused, and point
RTE sources for multiple scattering, with weak packet–OM pairs omitted.
Measured accuracy and speed of these choices: docs/accuracy.md.
"""
from dataclasses import replace

import numpy as np

from .adaptive_screen import AdaptiveScreenConfig
from .data import MuonData
from .joint_muon import JointMuonConfig, transport_joint_muon
from .preview import check_atlas_optics, independent_preview

#: RTE frequency grid of the prepared muon tables and the readout grid.
FREQUENCIES_PER_NS = np.linspace(0., 1.2, 65)
READOUT_PER_NS = np.linspace(0., 1.2, 321)


def muon_kernel(kernel, data):
    """Kernel on the prepared muon cache with the same medium and detector.

    The given kernel is reused when it already has the muon frequency grids
    and cache; otherwise a sibling kernel keeps its bins, threshold and other
    settings. The cache itself is never built here (policy 'require').
    """
    from ..transport import TransportKernel
    from pathlib import Path
    config = kernel.config
    same = (config.cache_directory is not None
            and Path(config.cache_directory).expanduser().resolve() == Path(data.rte_cache).resolve()
            and np.array_equal(np.asarray(config.omega_per_ns), FREQUENCIES_PER_NS)
            and config.readout_omega_per_ns is not None
            and np.array_equal(np.asarray(config.readout_omega_per_ns), READOUT_PER_NS))
    if same:
        return kernel
    return TransportKernel(kernel.medium, kernel.detector,
                           replace(config, omega_per_ns=FREQUENCIES_PER_NS,
                                   readout_omega_per_ns=READOUT_PER_NS,
                                   cache_directory=data.rte_cache, cache_policy='require'))


class MuonTransport:
    """Transport for muon events on one detector.

    ``data`` is a :class:`MuonData`, a manifest path, or None to search with
    :meth:`MuonData.find` next to the kernel's cache. Tables are opened on the
    first event; numba compiles its kernels once per installation (on disk).

    ``threads`` limits the numba threads used by one event (default: all,
    or ``NUMBA_NUM_THREADS``). The direct light, the Q1 grouping, the omission
    bounds and the >=2 angular weights run in parallel; the result does not
    depend on the thread count beyond round-off in Q1. For many events,
    several processes (one event each) scale better than threads.
    """

    def __init__(self, kernel, data=None, *, config=None, screen=None, threads=None):
        """Find the tables, build the muon kernel; tables are opened on first use."""
        if data is None:
            data = MuonData.find(kernel.config.cache_directory)
        elif not isinstance(data, MuonData):
            data = MuonData.load(data)
        self.data = data
        self.kernel = muon_kernel(kernel, data)
        self.config = config or JointMuonConfig()
        self.screen = screen or AdaptiveScreenConfig(scatter_margin=2.)
        self.threads = threads
        self._tables = None

    def tables(self):
        """Open the prepared tables once and check them against the kernel."""
        if self._tables is None:
            from .first_charge_beta_atlas import FirstChargeBetaAtlas
            from .ge2_charge_beta_atlas import Ge2ChargeBetaAtlas
            from .positive_fluence import PositiveFluenceAtlas
            from .track_first_primitive import TrackFirstAtlasStore
            from .track_ge2_primitive import DiskTrackGe2Atlas
            data = self.data
            first = TrackFirstAtlasStore(data.primary_first_atlas)
            ge2 = DiskTrackGe2Atlas(data.primary_ge2_atlas)
            preview = PositiveFluenceAtlas.load(data.preview_atlas)
            check_atlas_optics(preview, self.kernel)
            self._tables = dict(
                first=first, ge2=ge2, preview=preview,
                first_beta=FirstChargeBetaAtlas(data.primary_first_beta, first),
                ge2_beta=Ge2ChargeBetaAtlas(data.primary_ge2_beta, ge2),
                ge2_store=self.kernel.ge2_kernel_store(data.ge2_pack))
        return self._tables

    def __call__(self, source, ancestry, *, primary_uid=0, progress=None,
                 time_origin_ns=None, selection_extra_charge=None, receivers=None):
        """Response for one event (``HybridTransportResponse``).

        ``source`` holds all charged G4 steps with spectral cones; ``ancestry``
        maps every step to its family (``element_family``) and describes the
        family roots (uid, pdgid, energy_GeV, start_m, start_ns, creator).
        ``primary_uid`` is the source UID of the muon itself.

        For events with several sources (:mod:`lighthit.event`):
        ``time_origin_ns`` is a common per-OM clock, ``selection_extra_charge``
        the light of the other sources added to every threshold decision, and
        ``receivers`` restricts the calculation to those OMs (all of them
        computed in detail, threshold zero); the response then covers only
        ``receivers`` and ``time_origin_ns``/``selection_extra_charge`` refer
        to them. All three default to the single-source behaviour.
        """
        options = dict(time_origin_ns=time_origin_ns, selection_extra_charge=selection_extra_charge,
                       receivers=receivers)
        if self.threads is not None:
            import numba
            previous = numba.get_num_threads()
            numba.set_num_threads(max(1, min(int(self.threads), numba.config.NUMBA_NUM_THREADS)))
            try:
                return self._event(source, ancestry, primary_uid, progress, **options)
            finally:
                numba.set_num_threads(previous)
        return self._event(source, ancestry, primary_uid, progress, **options)

    def _event(self, source, ancestry, primary_uid, progress, *, time_origin_ns=None,
               selection_extra_charge=None, receivers=None):
        """One event with the current thread setting."""
        tables = self.tables()
        kernel = self.kernel
        if receivers is not None:
            from .universal_primary import subset_kernel
            kernel = subset_kernel(kernel, receivers)
            kernel.config = replace(kernel.config, threshold_pe=0.)
        preview = independent_preview(kernel, source, tables['preview'], primary_uid=primary_uid)
        config = replace(self.config, primary_uid=int(primary_uid))
        response = transport_joint_muon(
            kernel, source, ancestry, preview=preview,
            time_origin_ns=time_origin_ns, selection_extra_charge=selection_extra_charge,
            primary_first_store=tables['first'], primary_ge2_store=tables['ge2'],
            ge2_store=tables['ge2_store'], config=config, screen_config=self.screen,
            progress=progress,
            primary_options=dict(ge2_beta_atlas=tables['ge2_beta'],
                                 first_beta_atlas=tables['first_beta'],
                                 direct_ge2_quadrature='filon', q1_charge_gauss_order=6,
                                 fallback_pack=self.data.ge2_pack))
        response.metadata['muon_data'] = str(self.data.manifest)
        response.metadata['route'] = 'lighthit.muon'
        # Source points closer than the radial minimum of the >=2 tables were
        # left out of the route's >=2 sums; add them with the near-field clamp.
        from ..near_field import add_near_field
        bundle = tables['ge2_store'].bundle(kernel, np.arange(len(kernel.config.omega_per_ns)))
        add_near_field(response, kernel, source, bundle, label='muon')
        return response


def transport_muon(kernel, source, ancestry, *, primary_uid=0, data=None, progress=None):
    """One-call form of :class:`MuonTransport` for a single event."""
    return MuonTransport(kernel, data)(source, ancestry, primary_uid=primary_uid, progress=progress)
