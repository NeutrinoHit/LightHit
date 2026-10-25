"""Receiver-frame moments reproduce direct cubic acceptance mixtures."""
import numpy as np

from lighthit._receiver_first_groups import _EXPONENTS,_rotate_weights
from lighthit.receiver_first_fold import acceptance_moment_weights


def test_rotated_mixture_matches_direct_cubic_weights():
    rng=np.random.default_rng(8361)
    looks=rng.normal(size=(31,3));looks/=np.linalg.norm(looks,axis=1)[:,None]
    masses=rng.normal(size=(2,len(looks)))
    raw=masses@np.prod(looks[:,None,:]**_EXPONENTS[None,:,:],axis=-1)
    polynomial=np.array([.3082,.54192,.19831,-.04912])
    for _ in range(7):
        rotation=np.linalg.qr(rng.normal(size=(3,3)))[0]
        reference=masses@np.stack([acceptance_moment_weights(look@rotation,polynomial)
                                   for look in looks])
        np.testing.assert_allclose(_rotate_weights(raw,rotation,polynomial),reference,
                                   rtol=2e-13,atol=4e-15)
