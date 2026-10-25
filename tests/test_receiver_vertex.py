"""Boundary-jump normalization and the unchanged exact free-tail resolvent."""
import numpy as np
from dataclasses import replace
from lighthit.receiver_vertex import incoming_vertex,receiver_rows,first_entry_vertex,ReceiverVertexConfig
from lighthit.medium import synthetic_medium
from lighthit.receiver import SphereSurface
from lighthit.directional import CouplingTable
from lighthit.angular import resolvent_rows
from lighthit.axial_source import real_spherical_harmonics


def test_return_vertex_preserves_the_solved_outgoing_characteristic_flux():
    from lighthit.receiver_vertex import _outgoing_flux
    from lighthit.receiver_adjoint import _operators
    for m in range(3):
        modes,M,_=_operators(m,8);pick=np.flatnonzero(modes[:,0]==m)
        block=M[np.ix_(pick,pick)];values,vectors=np.linalg.eigh(block)
        positive=values>0;amplitudes=np.arange(1,positive.sum()+1.)[:,None]
        field=vectors[:,positive]@amplitudes
        expected=(vectors[:,positive]*values[positive])@amplitudes
        np.testing.assert_allclose(_outgoing_flux(field,m,6),expected[:7-m],rtol=2e-12,atol=1e-13)


def test_inward_vertex_matches_direct_visible_surface_phase_and_tilt():
    k=np.array([0.,.5,3.,8.]);R=.216
    table,vertex=incoming_vertex(k,R,3,spatial_degree=18)
    u=np.array([.6,0.,.8]);look=np.array([.3,.4,np.sqrt(.75)])
    source=real_spherical_harmonics(table.source_degree,u[None,:])[0]
    detector=real_spherical_harmonics(3,look[None,:])[0]
    x,w=np.polynomial.legendre.leggauss(32);q=(x+1)/2;w=w/2
    phi=(np.arange(64)+.5)*2*np.pi/64
    e1=np.array([.8,0.,-.6]);e2=np.array([0.,1.,0.])
    normals=-q[:,None,None]*u+np.sqrt(1-q*q)[:,None,None]*(np.cos(phi)[None,:,None]*e1+np.sin(phi)[None,:,None]*e2)
    D=np.einsum('q,kqp->k',w*q*2/len(phi),np.exp(1j*k[:,None,None]*R*normals[None,:,:,2]))
    for lam in range(4):
        got=np.zeros(len(k),complex);alpha=4*np.pi/(2*lam+1)
        for p in np.flatnonzero(table.pair_lambda==lam):
            m=table.pair_mu[p]
            for j in range(m,table.source_degree+1):
                got+=alpha*vertex[:,j,p,0]*source[j*j+j+m]*detector[lam*lam+lam+m]
        expected=D*np.polynomial.legendre.legval(np.dot(u,look),[0.]*lam+[1.])
        np.testing.assert_allclose(got,expected,rtol=1e-11,atol=2e-13)


def test_unit_vertex_reproduces_the_point_resolvent_orders():
    k=np.array([0.,.2,1.]);table=CouplingTable.build(6,2)
    vertex=np.zeros((len(k),7,table.pairs,3),complex)
    for p,lam in enumerate(table.pair_lambda):vertex[:,lam,p,0]=1.
    medium=replace(synthetic_medium(),g=.7)
    got=receiver_rows(k,.03,medium,8,6,2,vertex,table)
    expected=resolvent_rows(k,.03,medium,8,6,2)
    for a,b in zip(got,expected):
        for key in a:np.testing.assert_allclose(a[key],b[key],rtol=1e-13,atol=1e-14)


def test_no_water_scattering_has_no_outward_return_source():
    medium=replace(synthetic_medium(),scattering_per_m=0.)
    table,vertex=first_entry_vertex(np.array([0.,1.]),.1,medium,SphereSurface(.216),4,2,
        config=ReceiverVertexConfig(angular_pairs=4,radial_nodes=32))
    np.testing.assert_array_equal(vertex[...,1:],0.)
    first,multiple=receiver_rows(np.array([0.,1.]),.1,medium,4,4,2,vertex,table)
    for rows in (first,multiple):
        for values in rows.values():np.testing.assert_array_equal(values,0.)


def test_geometric_first_return_is_positive_and_converges_at_the_glass():
    from lighthit.receiver_vertex import _first_return_local
    from lighthit.medium import Medium
    medium=Medium(.076,.019,.9,1.3713237239062512,450.)
    table=CouplingTable.build(4,0)
    coarse=_first_return_local(0.,medium,.216,table,4,
        ReceiverVertexConfig(return_angle_nodes=32,return_path_nodes=256))
    fine=_first_return_local(0.,medium,.216,table,4,
        ReceiverVertexConfig(return_angle_nodes=64,return_path_nodes=512))
    assert fine[0,0].real>0
    np.testing.assert_allclose(fine[0,0].real,8.68921337e-5,rtol=2e-7)
    np.testing.assert_allclose(coarse,fine,rtol=2e-5,atol=1e-11)
