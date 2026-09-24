"""Shared cache settings for a point laser on the full detector array."""
import numpy as np
import lighthit as lh


def laser_config(detector, *, cache_directory, bin_ns=5.0,
                 cache_policy="build", omega_per_ns=None, max_radius_m=None):
    """One position-independent laser cache profile for this detector.

    The detector centroid chooses the default radial extent once, not a
    physical laser position. A user-supplied maximum extends or narrows that
    shared profile. The 2*pi/r_max panel rule is a starting point, not a
    substitute for a k-grid convergence scan.
    """
    radial_rule = getattr(lh, "point_source_radial_range", None)
    if radial_rule is None:
        raise RuntimeError(
            "These repository laser examples require LightHit 0.2.0a8 or newer; "
            f"Python imported {lh.__version__} from {lh.__file__}. "
            "Install this checkout into the active environment with "
            "python -m pip install -e '.[demo,test]'.")
    reference = lh.describe_geometry(detector).centroid_m
    radial_range = radial_rule(detector, reference)
    if max_radius_m is not None:
        maximum = float(max_radius_m)
        if not np.isfinite(maximum) or maximum <= radial_range[0]:
            raise ValueError("max_radius_m must exceed the radial minimum")
        radial_range = (radial_range[0], maximum)
    k_panel = min(0.04, 2 * np.pi / radial_range[1])
    return lh.KernelConfig(
        **({} if omega_per_ns is None else {"omega_per_ns": omega_per_ns}),
        relative_time_edges_ns=np.arange(-60., 740. + bin_ns, bin_ns),
        radial_range_m=radial_range,
        k_panel_per_m=k_panel,
        cache_directory=cache_directory,
        cache_policy=cache_policy)
