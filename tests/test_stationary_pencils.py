"""Physical charge controls for the experimental arrival-invariant grouping."""
from dataclasses import replace
import numpy as np
import pytest
from lighthit import SpectralLightElements, KernelConfig
from lighthit.model import SpectralMedium,DetectorArray
from lighthit.transport import TransportKernel
from lighthit.prompt import PromptConfig
from lighthit.experimental.stationary_pencils import stationary_pencils_charge
from lighthit.experimental.first_charge_only import first_scattering_charge_only


def case():
    medium=SpectralMedium(np.array([350.,610.]),np.array([.05,.05]),np.array([.015,.015]),
        np.array([1.34,1.34]),np.array([1.37,1.37]),g=.9)
    detector=DetectorArray(np.array([[30.,0.,0.],[10.,0.,25.],[50.,0.,-20.]]),
        np.tile([0.,0.,-1.],(3,1)),1.,lambda x:1.+.2*x,lambda w:np.ones_like(w))
    kernel=TransportKernel(medium,detector,KernelConfig(omega_per_ns=np.array([0.]),wavelength_nodes=3,
        relative_time_edges_ns=np.arange(-60.,745.,5.),cache_policy='require'))
    beta=np.array([.95,.8,.7])
    source=SpectralLightElements(np.array([[0.,0.,-40.],[0.,0.,-39.],[0.,0.,-38.]]),
        np.tile([0.,0.,1.],(3,1)),np.full(3,.01),np.ones(3),-1/beta**2,beta,
        np.array([0.,10.,20.]),np.array([1.,11.,21.]),np.arange(3),np.arange(3),cone_model='spectral')
    return kernel,source


def test_stationary_charge_ignores_individual_clocks_and_readout():
    kernel,source=case();q,meta=stationary_pencils_charge(kernel,source)
    changed=replace(source,start_ns=source.start_ns+np.array([1234.,4567.,8901.]),
        end_ns=source.end_ns+np.array([1299.,4588.,8999.]))
    kernel.config=replace(kernel.config,relative_time_edges_ns=np.array([-1.,0.,1.]))
    other,_=stationary_pencils_charge(kernel,changed)
    np.testing.assert_allclose(q,other,rtol=1e-11,atol=1e-20)
    assert meta['computed_bins'] is False and meta['certified_upper_bound'] is False


def test_small_cones_against_uncompressed_segment_charge():
    kernel,source=case()
    q,_=stationary_pencils_charge(kernel,source,config=PromptConfig(element_method='grouped',
        screen=False,cell_m=.1,merge_log_r=.005,merge_theta_ratio=.02,ring_nodes=384,pixel_face=64))
    direct,_=first_scattering_charge_only(kernel,source,gauss_order=6)
    np.testing.assert_allclose(q,direct,rtol=.003,atol=1e-14)


def test_threshold_and_zero_scattering_emit_no_first_order():
    kernel,source=case()
    dark=replace(source,beta=np.full(3,.7),coefficient2=-source.coefficient0/.7**2)
    q,_=stationary_pencils_charge(kernel,dark);np.testing.assert_array_equal(q,0.)
    kernel.medium=replace(kernel.medium,scattering_per_m=np.zeros(2))
    q,_=stationary_pencils_charge(kernel,source);np.testing.assert_array_equal(q,0.)


def test_refined_radial_groups_do_not_merge_distant_sources():
    kernel,source=case()
    kernel.medium=replace(kernel.medium,absorption_per_m=np.full(2,.005))
    kernel.detector=replace(kernel.detector,positions_m=np.array([[0.,0.,0.]]),
        orientations=np.array([[0.,0.,-1.]]),effective_area_m2=np.ones(1))
    source=source.subset(np.arange(len(source))<2)
    source=replace(source,start_m=np.array([[0.,0.,-250.],[0.,0.,-450.]]),
        beta=np.full(2,.95),coefficient2=-source.coefficient0/.95**2)
    config=PromptConfig(element_method='grouped',screen=False,merge_log_r=.005)
    both,_=stationary_pencils_charge(kernel,source,config=config)
    first,_=stationary_pencils_charge(kernel,source.subset(np.array([True,False])),config=config)
    second,_=stationary_pencils_charge(kernel,source.subset(np.array([False,True])),config=config)
    # Distinct radial groups must give an additive response. In the old
    # clamped key they shared one attenuated profile at an artificial mean r.
    # Independent pencil compression subtracts large position moments; its
    # covariance roundoff gives ~1 ppm here. A 10 ppm allowance still excludes
    # the large attenuation bias from pooling the two different radii.
    np.testing.assert_allclose(both,first+second,rtol=1e-5,atol=1e-20)


def test_unsupported_hash_resolution_fails_instead_of_aliasing():
    kernel,source=case()
    with pytest.raises(ValueError,match='integer key range'):
        stationary_pencils_charge(kernel,source,config=PromptConfig(element_method='grouped',
            screen=False,merge_log_r=1e-4))
