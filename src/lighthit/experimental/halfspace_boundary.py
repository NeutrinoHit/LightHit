"""Planar dielectric optics for a scalar half-space transport problem.

The tissue occupies z > 0. ``n`` is the phase-index ratio inside/outside;
the group index and the volume optical coefficients belong to the medium.
Cosines are unsigned and measured from the normal on the incident side.
The formulas describe a smooth, lossless interface and unpolarized light.
"""
from dataclasses import dataclass
import math

import numpy as np


def _cosines(value):
    cosine = np.asarray(value, dtype=float)
    if not np.isfinite(cosine).all() or np.any((cosine < 0) | (cosine > 1)):
        raise ValueError("incident cosines must be finite and lie in [0, 1]")
    return cosine


@dataclass(frozen=True)
class FresnelCoefficients:
    """Flux fractions and the direction of the transmitted ray.

    ``radiance_factor`` multiplies physical incident radiance. Reduced
    radiance I / phase_index**2 instead transmits with ``transmission``.
    ``transmitted_cosine`` is zero under total internal reflection;
    consult ``total_internal_reflection`` before using it as a ray.
    """
    reflection: np.ndarray
    transmission: np.ndarray
    transmitted_cosine: np.ndarray
    radiance_factor: np.ndarray
    total_internal_reflection: np.ndarray


@dataclass(frozen=True)
class HalfSpaceBoundary:
    """One monochromatic phase-index ratio, n_inside / n_outside.

    n=1 is exactly matched, including the tangent-ray convention. Changing
    n never changes the group velocity. A spectral calculation constructs
    this boundary from the two phase indices at each wavelength.
    """
    n: float = 1.0

    def __post_init__(self):
        n = float(self.n)
        if not math.isfinite(n) or n <= 0 or not math.isfinite(1.0 / n):
            raise ValueError("n must be a finite positive phase-index ratio")
        object.__setattr__(self, "n", n)

    def coefficients(self, cosine, *, incidence="inside"):
        """Return Fresnel coefficients for incidence from either side.

        The transmitted cosine follows Snell's law. No angular integration
        weight or detector aperture is included in these coefficients.
        """
        if incidence not in ("inside", "outside"):
            raise ValueError("incidence must be 'inside' or 'outside'")
        mu = _cosines(cosine)
        eta = self.n if incidence == "inside" else 1.0 / self.n
        if eta == 1:
            return FresnelCoefficients(np.zeros_like(mu), np.ones_like(mu),
                                       mu.copy(), np.ones_like(mu),
                                       np.zeros_like(mu, dtype=bool))

        sin_transmitted = eta * np.sqrt((1 - mu) * (1 + mu))
        tir = sin_transmitted >= 1
        sine = np.minimum(sin_transmitted, 1)
        mu_t = np.sqrt((1 - sine) * (1 + sine))

        def polarized(a, b):
            denominator = a + b
            difference = np.divide(a - b, denominator, out=np.ones_like(mu),
                                   where=denominator > 0)
            x = np.divide(a, denominator, out=np.zeros_like(mu),
                          where=denominator > 0)
            y = np.divide(b, denominator, out=np.zeros_like(mu),
                          where=denominator > 0)
            # Direct transmission retains small grazing flux when 1-R
            # would lose it to cancellation.
            return difference * difference, 4 * x * y

        rs, ts = polarized(eta * mu, mu_t)
        rp, tp = polarized(mu, eta * mu_t)
        reflection = np.where(tir, 1., (rs + rp) / 2)
        transmission = np.where(tir, 0., (ts + tp) / 2)
        return FresnelCoefficients(reflection, transmission, mu_t,
                                   transmission / eta / eta, tir)

    def incoming_radiance(self, cosine, outgoing_radiance, *, external_radiance=0.):
        """Internal incoming physical radiance at the specular partner.

        External radiance is sampled at the corresponding refracted angle;
        its coefficient includes the squared phase-index ratio. Complex
        frequency-domain fields are supported. Under total reflection the
        external term vanishes.
        """
        c = self.coefficients(cosine)
        return (c.reflection * np.asarray(outgoing_radiance)
                + (c.transmission * self.n * self.n)
                * np.asarray(external_radiance))

    def incoming_reduced_radiance(self, cosine, outgoing, *, external=0.):
        """The same boundary for u=I/n_phase**2, in either solve's units.

        A reciprocal external port injects its angular weight through
        ``transmission`` in these variables. Surface integration uses the
        invariant measure n_phase**2 * mu * dA * dOmega.
        """
        c = self.coefficients(cosine)
        return (c.reflection * np.asarray(outgoing)
                + c.transmission * np.asarray(external))


@dataclass(frozen=True)
class ExternalAperture:
    """A circular angular aperture centred on the external surface normal.

    The half-angle is in the exterior medium. Spatial area and angular
    quadrature weights are supplied separately by the source/receiver port.
    """
    half_angle_rad: float = math.pi / 2

    def __post_init__(self):
        angle = float(self.half_angle_rad)
        if not math.isfinite(angle) or not 0 <= angle <= math.pi / 2:
            raise ValueError("half_angle_rad must lie in [0, pi/2]")
        object.__setattr__(self, "half_angle_rad", angle)

    def internal_weight(self, boundary, cosine):
        """T times external acceptance, for reciprocal reduced-radiance ports.

        A receiver integrates ``mu * weight * I_inside`` over internal
        directions and area. Its reversed source injects ``weight`` as
        reduced incoming radiance, with the port's flux normalization.
        """
        c = boundary.coefficients(cosine)
        cutoff = 0. if self.half_angle_rad == math.pi / 2 else math.cos(self.half_angle_rad)
        return c.transmission * (c.transmitted_cosine >= cutoff)

    def escaped_flux_density(self, boundary, cosine, outward_radiance):
        """Collected flux per internal solid angle and surface area.

        The refractive angular Jacobian is already accounted for: integrate
        this in internal solid angle, without an additional n**2 factor.
        For time-integrated radiance the result is time-integrated flux.
        """
        mu = _cosines(cosine)
        return mu * self.internal_weight(boundary, mu) * np.asarray(outward_radiance)
