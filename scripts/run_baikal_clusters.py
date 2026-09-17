"""Laser flash seen by three Baikal-GVD-like clusters, through the cache.

Geometry and laser parameters are taken from published Baikal-GVD numbers
(recorded in the report). The water is the repository's synthetic
near-Baikal medium, not a calibration and not the private optical table.
"""
import sys, json, time, math, platform
sys.path.insert(0, "src")
import numpy as np, warnings
warnings.filterwarnings("ignore")
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lighthit import __version__
from lighthit.medium import synthetic_medium
from lighthit.green import PointGreenSolver, SolverSettings
from lighthit.cache import (BandedResponseCache, acceptance_coefficients,
                            isotropic_acceptance, cosine_acceptance,
                            hemispherical_acceptance, first_order_consistency)

OUT = "docs/cache-baikal"
cache = BandedResponseCache.load(f"{OUT}/cache.npz")
medium = cache.medium

# ---------------------------------------------------------------- published
# Baikal-GVD cluster: 8 strings (1 central + 7 peripheral at 60 m),
# 36 OMs per string, 15 m vertical spacing, instrumented 750-1275 m.
# Neighbouring clusters about 250-300 m apart.
STRINGS_PER_CLUSTER = 8
PERIPHERAL_RADIUS_M = 60.0
OMS_PER_STRING = 36
OM_SPACING_M = 15.0
TOP_DEPTH_M = 750.0
CLUSTER_PITCH_M = 250.0
# Laser: 532 nm, 0.37 mJ per pulse, about 1 ns, isotropic diffuser.
LASER_WAVELENGTH_NM = 532.0
LASER_PULSE_ENERGY_J = 0.37e-3
PLANCK_J_S = 6.62607015e-34
LIGHT_M_PER_S = 2.99792458e8
PHOTON_ENERGY_J = PLANCK_J_S * LIGHT_M_PER_S / (LASER_WAVELENGTH_NM * 1e-9)
LASER_PHOTONS = LASER_PULSE_ENERGY_J / PHOTON_ENERGY_J
# Optical module: 10-inch PMT, effective photocathode area 530 cm^2, faces down.
OM_AREA_M2 = 530e-4
OM_QUANTUM_EFFICIENCY = 0.20   # not found published at 532 nm; an explicit input
OM_AXIS = np.array([0.0, 0.0, -1.0])

ACCEPTANCES = {
    "ideal isotropic": isotropic_acceptance,
    "hemispherical (1+x)/2": hemispherical_acceptance,
    "bare cosine max(x,0)": cosine_acceptance,
}
DEGREE = min(band.degree for band in cache.bands)
ALPHA = {name: acceptance_coefficients(fn, DEGREE) for name, fn in ACCEPTANCES.items()}


def cluster_positions(center_xy):
    """One cluster: central string plus seven peripheral, 36 modules each."""
    xs = [np.array([0.0, 0.0])]
    for i in range(STRINGS_PER_CLUSTER - 1):
        angle = 2 * np.pi * i / (STRINGS_PER_CLUSTER - 1)
        xs.append(PERIPHERAL_RADIUS_M * np.array([np.cos(angle), np.sin(angle)]))
    depths = TOP_DEPTH_M + OM_SPACING_M * np.arange(OMS_PER_STRING)
    out = []
    for horizontal in xs:
        for depth in depths:
            out.append([center_xy[0] + horizontal[0], center_xy[1] + horizontal[1], -depth])
    return np.array(out)


CENTERS = [(0.0, 0.0), (CLUSTER_PITCH_M, 0.0),
           (CLUSTER_PITCH_M / 2, CLUSTER_PITCH_M * math.sqrt(3) / 2)]
modules = np.concatenate([cluster_positions(c) for c in CENTERS])
cluster_index = np.concatenate([np.full(len(cluster_positions(c)), i)
                                for i, c in enumerate(CENTERS)])
# Laser on a technological string at the centroid, at mid-depth.
mid_depth = TOP_DEPTH_M + OM_SPACING_M * (OMS_PER_STRING - 1) / 2
laser = np.array([np.mean([c[0] for c in CENTERS]),
                  np.mean([c[1] for c in CENTERS]), -mid_depth])
displacement = modules - laser
radii = np.linalg.norm(displacement, axis=1)
print(f"modules: {len(modules)}  distance range {radii.min():.1f} - {radii.max():.1f} m")
print(f"laser photons per pulse: {LASER_PHOTONS:.3e}")

low, high = cache.radius_range_m
inside = (radii >= low) & (radii <= high)
print(f"within cache range [{low:.0f},{high:.0f}] m: {inside.sum()} of {len(radii)}")

# --------------------------------------------------------- cached evaluation
results = {}
timings = {}
report_timing = {}
for name, alpha in ALPHA.items():
    t0 = time.perf_counter()
    charge = np.zeros((len(radii), 3))
    charge[inside] = cache.charge_for_modules(displacement[inside], OM_AXIS, alpha,
                                              photons=LASER_PHOTONS,
                                              acceptance=ACCEPTANCES[name])
    timings[name] = time.perf_counter() - t0
    top_degree = int(np.flatnonzero(np.abs(alpha) > 1e-12 * np.max(np.abs(alpha)))[-1])
    results[name] = charge
    pe = charge.sum(axis=1) * OM_AREA_M2 * OM_QUANTUM_EFFICIENCY
    print(f"{name:24s}: {timings[name]*1e3:8.2f} ms for {inside.sum()} modules "
          f"(top degree {top_degree}), max {pe.max():.3g} p.e., above 1 p.e.: {(pe > 1).sum()}")
    report_timing[name] = dict(seconds=timings[name], per_module_us=timings[name]/max(inside.sum(),1)*1e6,
                               acceptance_top_degree=top_degree)

# ------------------------------------------------- cost of not using a cache
sample = np.flatnonzero(inside)[::97][:8]
t0 = time.perf_counter()
for index in sample:
    band = next(b for b in cache.bands
                if b.radius_range_m[0] <= radii[index] <= b.radius_range_m[1])
    PointGreenSolver(medium, band.settings).solve(
        [0.0], displacement[index], direction=None)
direct_s = (time.perf_counter() - t0) / len(sample)
cached_per_module_s = timings["hemispherical (1+x)/2"] / inside.sum()
print(f"direct isotropic solve: {direct_s:.3f} s per module "
      f"(sampled {len(sample)}); cached query {cached_per_module_s*1e6:.1f} us per module; "
      f"ratio {direct_s/cached_per_module_s:.3g}")

# ------------------------------------------------------- angular response map
map_radii = np.array([20.0, 60.0, 150.0])
map_cos = np.linspace(-1.0, 1.0, 161)
angular = {}
for name, alpha in ALPHA.items():
    block = np.zeros((len(map_radii), len(map_cos), 3))
    for i, r in enumerate(map_radii):
        block[i] = cache.acceptance_charge(np.full_like(map_cos, r), map_cos, alpha,
                                           acceptance=ACCEPTANCES[name])
    angular[name] = block

consistency = {f"{lo}-{hi} m": first_order_consistency(
                   cache, np.exp(np.random.default_rng(7).uniform(np.log(lo), np.log(hi), 12)))
               for lo, hi in [(10.5, 19.5), (21.0, 58.0), (62.0, 290.0)]}

# ------------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=False)
order_labels = ("0 scattering", "1 scattering", ">=2 scattering")
for ax, (name, block) in zip(axes, angular.items()):
    i = 1  # r = 60 m
    total = block[i].sum(axis=1)
    peak = total.max()
    for o in range(3):
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.plot(map_cos, block[i, :, o] / peak, label=order_labels[o])
    ax.plot(map_cos, total / peak, "k", lw=2, label="total")
    shape = ACCEPTANCES[name](map_cos)
    ax.plot(map_cos, shape / max(shape.max(), 1e-30), "--", color="grey",
            label="acceptance A(x)")
    ax.set_title(f"{name}\nr = {map_radii[i]:.0f} m", fontsize=10)
    ax.set_xlabel("cos(angle between head-on direction and source)")
    ax.set_yscale("symlog", linthresh=1e-4)
axes[0].set_ylabel("charge, normalized to the peak total")
axes[-1].legend(fontsize=8)
fig.suptitle("What an optical module is sensitive to, by scattering order")
fig.tight_layout()
fig.savefig(f"{OUT}/om-angular-map.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, r_index in zip(axes, range(len(map_radii))):
    for name, block in angular.items():
        total = block[r_index].sum(axis=1)
        ax.plot(map_cos, total / total.max(), label=name)
    ax.plot(map_cos, hemispherical_acceptance(map_cos), "--", color="grey",
            label="A(x) = (1+x)/2")
    ax.set_title(f"r = {map_radii[r_index]:.0f} m")
    ax.set_xlabel("cos(head-on, source)")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 2)
axes[0].set_ylabel("total charge / peak")
axes[0].legend(fontsize=8)
fig.suptitle("Directional contrast washes out with distance")
fig.tight_layout()
fig.savefig(f"{OUT}/om-contrast.png", dpi=150)
plt.close(fig)

pe = results["hemispherical (1+x)/2"].sum(axis=1) * OM_AREA_M2 * OM_QUANTUM_EFFICIENCY
# Beyond the point where the charge falls to about 1e-15 per m^2 per photon the
# Bessel sum is producing its value by cancellation and is not converged in
# k_max; unphysical negative components appear there. Report that boundary.
negative = (results["hemispherical (1+x)/2"] < 0).any(axis=1)
trust_radius_m = float(radii[negative].min()) if negative.any() else float(radii.max())
print(f"first unphysical (negative) component at r = {trust_radius_m:.1f} m, "
      f"{pe[radii >= trust_radius_m].max():.3g} p.e. there; "
      f"{(pe > 0.01).sum()} modules above 0.01 p.e., all inside")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
ax = axes[0]
visible = pe > 0
sc = ax.scatter(modules[visible, 0], modules[visible, 1],
                c=np.log10(np.maximum(pe[visible], 1e-6)), s=14, cmap="viridis")
ax.scatter([laser[0]], [laser[1]], marker="*", s=260, color="red", label="laser")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
ax.set_title("Plan view, colour = log10 photoelectrons")
ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="log10 p.e.")
ax = axes[1]
for i, _ in enumerate(CENTERS):
    m = (cluster_index == i) & (pe > 0)
    ax.scatter(radii[m], pe[m], s=10, label=f"cluster {i+1}")
ax.axhline(1.0, color="grey", ls="--", lw=1, label="1 p.e.")
ax.axvline(trust_radius_m, color="crimson", ls=":", lw=1.5,
           label=f"method floor, {trust_radius_m:.0f} m")
ax.set_xlabel("distance from laser [m]"); ax.set_ylabel("photoelectrons")
ax.set_yscale("log"); ax.legend(fontsize=8)
ax.set_title(f"{LASER_PHOTONS:.2e} photons, A={OM_AREA_M2*1e4:.0f} cm2, QE={OM_QUANTUM_EFFICIENCY}")
fig.tight_layout()
fig.savefig(f"{OUT}/cluster-response.png", dpi=150)
plt.close(fig)

report = dict(
    lighthit_version=__version__,
    medium=dict(absorption_per_m=medium.absorption_per_m,
                scattering_per_m=medium.scattering_per_m, g=medium.g,
                group_index=medium.group_index,
                wavelength_nm=medium.wavelength_nm, provenance=medium.provenance),
    medium_note=("the repository synthetic near-Baikal medium, specified at 450 nm; "
                 "the laser is 532 nm, so this run pairs a real photon yield with "
                 "synthetic optics and is illustrative, not a Baikal prediction"),
    laser=dict(wavelength_nm=LASER_WAVELENGTH_NM, pulse_energy_J=LASER_PULSE_ENERGY_J,
               photons_per_pulse=LASER_PHOTONS, emission="isotropic diffuser",
               position_m=laser.tolist()),
    optical_module=dict(effective_area_m2=OM_AREA_M2,
                        quantum_efficiency=OM_QUANTUM_EFFICIENCY,
                        quantum_efficiency_note="no published value found at 532 nm; explicit input",
                        axis=OM_AXIS.tolist(), acceptances=list(ACCEPTANCES)),
    geometry=dict(clusters=len(CENTERS), centers_m=CENTERS,
                  strings_per_cluster=STRINGS_PER_CLUSTER,
                  peripheral_radius_m=PERIPHERAL_RADIUS_M,
                  oms_per_string=OMS_PER_STRING, om_spacing_m=OM_SPACING_M,
                  top_depth_m=TOP_DEPTH_M, cluster_pitch_m=CLUSTER_PITCH_M,
                  modules=len(modules),
                  distance_range_m=[float(radii.min()), float(radii.max())],
                  modules_in_cache_range=int(inside.sum())),
    cache=dict(radius_range_m=list(cache.radius_range_m),
               build_time_s=cache.build_time_s,
               bands=[dict(range_m=list(b.radius_range_m), degree=b.degree,
                           settings=dict(scattering_degree=b.settings.scattering_degree,
                                         spatial_degree=b.settings.spatial_degree,
                                         k_max_per_m=b.settings.k_max_per_m),
                           radii=len(b.grid.radii_m),
                           build_s=b.timings_s.get("total")) for b in cache.bands]),
    timing=dict(cached_query=report_timing,
                cached_per_module_s=cached_per_module_s,
                direct_solve_per_module_s=direct_s,
                speedup=direct_s / cached_per_module_s,
                sampled_direct_modules=len(sample)),
    trust=dict(first_negative_component_radius_m=trust_radius_m,
               modules_with_negative_component=int(negative.sum()),
               modules_above_0p01pe=int((pe > 0.01).sum()),
               max_radius_above_0p01pe=float(radii[pe > 0.01].max()),
               negatives_inside_that_subset=int((negative & (pe > 0.01)).sum()),
               note=("negative components mark the accuracy floor of "
                     "@sec-discretizations, not a physical effect; they appear "
                     "only far below one photoelectron")),
    photoelectrons=dict(
        max=float(pe.max()), above_1pe=int((pe > 1).sum()),
        above_0p1pe=int((pe > 0.1).sum()),
        per_cluster_max=[float(pe[cluster_index == i].max()) for i in range(len(CENTERS))],
        per_cluster_above_1pe=[int((pe[cluster_index == i] > 1).sum())
                               for i in range(len(CENTERS))],
        radius_of_last_module_above_1pe=float(radii[pe > 1].max()),
        radius_of_last_module_above_0p1pe=float(radii[pe > 0.1].max())),
    order_fractions_by_distance={
        f"{lo}-{hi} m": dict(zip(order_labels,
            (results["hemispherical (1+x)/2"][(radii >= lo) & (radii < hi)].sum(axis=0)
             / results["hemispherical (1+x)/2"][(radii >= lo) & (radii < hi)].sum()).tolist()))
        for lo, hi in [(80, 120), (120, 160), (160, 200), (200, 250), (250, 283)]},
    order_fractions_at_60m={
        name: dict(zip(order_labels,
                       (block[1, len(map_cos)//2] / block[1, len(map_cos)//2].sum()).tolist()))
        for name, block in angular.items()},
    first_order_consistency=consistency,
    environment=dict(python=sys.version.split()[0], numpy=np.__version__,
                     scipy=scipy.__version__, platform=platform.platform()),
)
with open(f"{OUT}/baikal-report.json", "w") as fh:
    json.dump(report, fh, indent=2)
np.savez_compressed(f"{OUT}/baikal-modules.npz", modules=modules, laser=laser,
                    cluster_index=cluster_index, radii_m=radii,
                    charge_hemispherical=results["hemispherical (1+x)/2"],
                    charge_isotropic=results["ideal isotropic"],
                    charge_cosine=results["bare cosine max(x,0)"],
                    map_radii=map_radii, map_cos=map_cos,
                    angular_hemispherical=angular["hemispherical (1+x)/2"])
print("wrote baikal-report.json")
print(json.dumps(report["photoelectrons"], indent=2))
print(json.dumps(report["order_fractions_at_60m"], indent=2))
