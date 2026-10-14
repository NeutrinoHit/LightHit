from types import SimpleNamespace
import numpy as np
from lighthit.experimental.adaptive_screen import adaptive_candidates, AdaptiveScreenConfig


def case(q, clusters=None, strings=None, unsafe=None):
    q=np.asarray(q,float);n=len(q)
    detector=SimpleNamespace(positions_m=np.zeros((n,3)),identifiers=dict(
        cluster_id=np.array(clusters if clusters is not None else np.zeros(n,int)),
        string_id=np.array(strings if strings is not None else np.zeros(n,int))))
    preview=SimpleNamespace(charge_components_pe=q, unsafe_mask=np.array(unsafe if unsafe is not None else np.zeros(n,bool)))
    return detector,preview


def test_diffuse_guard_retains_signal_without_direct_or_first_light():
    d,p=case([[0,0,.004],[.02,0,0],[0,.00001,0]])
    truth=np.array([[0,0,.011],[.02,0,0],[0,.00001,0]])
    calls=[]
    def evaluate(ids):calls.extend(ids);return truth[ids]
    q,mask,meta=adaptive_candidates(p,d,.01,evaluate)
    assert mask.tolist()==[True,True,False]
    assert sorted(calls)==[0,1]
    assert q[0].sum()>.01
    assert not meta['inactivity_certified']


def test_observed_underprediction_reopens_receivers_without_repeating_work():
    d,p=case([[0,.003,0],[0,.001,0],[0,.00001,0]],strings=[0,0,0])
    truth=np.array([[0,.045,0],[0,.015,0],[0,.00001,0]])
    calls=[]
    def evaluate(ids):calls.extend(ids);return truth[ids]
    q,mask,meta=adaptive_candidates(p,d,.01,evaluate)
    assert mask.tolist()==[True,True,False]
    assert calls==[0,1]
    assert len(meta['rounds'])==2 and meta['margin']>=22.5


def test_unpredicted_scatter_at_probe_triggers_full_fallback():
    d,p=case([[0,0,0],[0,0,0],[0,0,0]],unsafe=[True,False,False])
    truth=np.array([[0,.02,0],[0,0,.03],[0,0,0]])
    calls=[]
    def evaluate(ids):calls.extend(ids);return truth[ids]
    q,mask,meta=adaptive_candidates(p,d,.01,evaluate)
    assert mask.all() and sorted(calls)==[0,1,2]
    assert meta['fallback'].endswith('blind scattered predictor')
    np.testing.assert_array_equal(q,truth)


def test_complete_audit_reveals_unseen_peak_without_false_certificate():
    d,p=case([[.02,0,0],[0,.00001,0]],strings=[0,0])
    truth=np.array([[.02,0,0],[0,.02,0]])
    config=AdaptiveScreenConfig(audit_all_rejected=True)
    q,mask,meta=adaptive_candidates(p,d,.01,lambda ids:truth[ids],config)
    assert mask.all() and meta['rounds'][-1]['recovered_active']==[1]
    assert meta['all_rejections_audited'] and not meta['inactivity_certified']


def test_unsafe_negative_is_evaluated_and_not_clipped_into_a_zero_charge():
    d,p=case([[0,-.002,.001],[.02,0,0]],unsafe=[True,False])
    truth=np.array([[0,.005,.007],[.02,0,0]])
    q,mask,_=adaptive_candidates(p,d,.01,lambda ids:truth[ids])
    assert mask.all()
    np.testing.assert_array_equal(q,truth)
