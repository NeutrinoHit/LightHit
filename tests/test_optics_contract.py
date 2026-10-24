"""The optical contract of the muon tables tolerates last-bit platform differences only."""
import numpy as np

from lighthit.muon.optics_contract import contract_matches, fingerprint, same_values


def contract():
    return dict(acceptance_alpha=[0.3, 0.5, 0.2, 0.0], effective_area_m2=0.1466, g=0.9,
                spectral_order=2, spectral_nodes=17,
                spectral_tables=dict(weight=np.linspace(0.1, 2.0, 17).tolist(),
                                     absorption=np.geomspace(0.04, 0.3, 17).tolist()),
                source="unit C0 per metre")


def perturbed(values, relative):
    out = dict(values)
    out["spectral_tables"] = {k: (np.asarray(v) * (1 + relative)).tolist()
                              for k, v in values["spectral_tables"].items()}
    return out


def test_same_values_compares_numbers_within_rtol_and_strings_exactly():
    a = contract()
    assert same_values(a, a)
    assert same_values(perturbed(a, 1e-14), a)
    assert not same_values(perturbed(a, 1e-6), a)
    assert not same_values(dict(a, source="other"), a)
    assert not same_values({k: v for k, v in a.items() if k != "g"}, a)


def test_contract_accepts_last_bit_differences_and_refuses_other_optics():
    stored = contract()
    recorded = dict(stored, extra_key="ignored", optical_fingerprint=fingerprint(stored))
    fp = recorded["optical_fingerprint"]
    assert contract_matches(stored, recorded, fp)                    # same platform
    assert fingerprint(perturbed(stored, 1e-13)) != fp
    assert contract_matches(perturbed(stored, 1e-13), recorded, fp)  # other platform
    assert not contract_matches(perturbed(stored, 1e-5), recorded, fp)
    tampered = dict(recorded, g=0.8)                                  # values not the fingerprinted ones
    assert not contract_matches(dict(stored, g=0.8), tampered, fp)
    assert not contract_matches(perturbed(stored, 1e-13), None, fp)
