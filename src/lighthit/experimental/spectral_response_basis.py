"""Positive empirical cubature of a family of spectral transport responses.

Training responses may vary in beta, geometry and orientation. Runtime transport
still evaluates the actual source beta at every selected wavelength. Compression
changes the wavelength quadrature, not the source cone or emission clock.
No held-out or event response enters fitting. Accuracy away from training cases
must be established by independent transport checks.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from ..model import WavelengthQuadrature


@dataclass(frozen=True)
class ResponseCubature:
    quadrature: WavelengthQuadrature
    selected_indices: np.ndarray
    modes: np.ndarray
    singular_values: np.ndarray
    metadata: dict

    @classmethod
    def load(cls,path):
        import json
        with np.load(path,allow_pickle=False) as data:
            model=cls(WavelengthQuadrature(data['wavelength_nm'].copy(),data['weight_nm'].copy()),
                data['selected_indices'].copy(),data['modes'].copy(),data['singular_values'].copy(),
                json.loads(str(data['metadata'])))
        if (model.modes.shape!=(model.metadata['rank'],model.metadata['reference_nodes'])
                or model.selected_indices.shape!=model.quadrature.weight_nm.shape
                or len(model.selected_indices)!=model.metadata['selected_nodes']):
            raise ValueError('response basis file dimensions differ')
        return model

    def validate_optics(self, kernel):
        from ..muon.track_first_primitive import track_first_optical_contract
        _,digest=track_first_optical_contract(kernel)
        if digest!=self.metadata['optics_fingerprint']:
            raise ValueError('spectral response basis optical contract differs')


def fit_response_cubature(reference, responses, rank, *, metadata=None):
    """Integrate retained weighted POD modes with a sparse positive measure.

    ``responses`` is (independent training feature, wavelength), without the
    wavelength quadrature weights. Callers normalize features to match their
    desired charge/waveform norm before fitting. The constant mode conserves
    total wavelength measure. Positive weights do not certify truncation error.
    """
    values=np.asarray(responses,float);weight=reference.weight_nm
    if values.ndim!=2 or values.shape[1]!=len(weight) or not len(values) or not np.isfinite(values).all():
        raise ValueError('finite feature-by-wavelength training responses required')
    if isinstance(rank,(bool,np.bool_)) or not isinstance(rank,(int,np.integer)) or not 1<=rank<min(values.shape):
        raise ValueError('rank must be a positive integer smaller than both matrix dimensions')
    _,singular,vt=np.linalg.svd(values*np.sqrt(weight)[None,:],full_matrices=False)
    modes=vt[:rank]/np.sqrt(weight)[None,:]
    equations=np.vstack((np.ones(len(weight)),modes))
    scales=np.max(abs(equations),axis=1)
    equations=equations/scales[:,None]
    rhs=equations@weight
    result=linprog(np.zeros(len(weight)),A_eq=equations,b_eq=rhs,bounds=(0,None),method='highs')
    if not result.success:raise ValueError('positive cubature solve failed: '+result.message)
    chosen=np.flatnonzero(result.x>0)
    reduced=reference.__class__(reference.wavelength_nm[chosen],result.x[chosen])
    defect=np.max(abs(equations[:,chosen]@result.x[chosen]-rhs))
    if defect>1e-7*max(1.,weight.sum()):raise ValueError('cubature mode conservation failed')
    info=dict(metadata or {},rank=int(rank),reference_nodes=len(weight),selected_nodes=len(chosen),
        moment_defect=float(defect),fit_features=len(values),positive_weights=True,
        omitted_pod_energy_fraction=float((singular[rank:]**2).sum()/max((singular**2).sum(),1e-300)),
        error_certified=False,source_beta_approximated=False)
    return ResponseCubature(reduced,chosen,modes,singular,info)


def evaluate_cubature(model, spectral_responses):
    values=np.asarray(spectral_responses,float)
    if values.shape[-1]!=model.metadata['reference_nodes']:raise ValueError('reference wavelength dimension differs')
    return values[...,model.selected_indices]@model.quadrature.weight_nm
