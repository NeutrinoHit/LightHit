"""Shared setup for the shower-basis study.

Private inputs (BGVD model, G4 files) and outputs stay outside Git:
set LIGHTHIT_DATA and LIGHTHIT_WORK.
"""
import os
import numpy as np, lighthit as lh
DATA_DIR = os.environ.get("LIGHTHIT_DATA", "/home/user/lighthit-data")
WORK_DIR = os.environ.get("LIGHTHIT_WORK", "/home/user/lighthit-work/sb")
MODEL = f"{DATA_DIR}/bgvd-model-master"
DATA = {"e100": f"{DATA_DIR}/sim_e_100GeV_10.h5",
        "e1000": f"{DATA_DIR}/sim_e_1TeV_10.h5",
        "mu100": f"{DATA_DIR}/sim_mu_100GeV_10.h5",
        "mu1000": f"{DATA_DIR}/sim_mu_1TeV_10.h5"}
CACHE = f"{WORK_DIR}/cache65"
POSITION = [11.428243447430553, -249.74053298611113, 362.13987569444436]
DIRECTION = [0.4, 0.2, 0.894]

def model():
    return lh.load_bgvd_model(MODEL, dataset="2021", clusters=1)

def config(**kw):
    base = dict(omega_per_ns=np.linspace(0., 1.2, 65),
                readout_omega_per_ns=np.linspace(0., 1.2, 321),
                relative_time_edges_ns=np.arange(-60., 745., 5.),
                cache_directory=CACHE, cache_policy="require")
    base.update(kw)
    return lh.KernelConfig(**base)

def shower(kind="e100", event=5, position=POSITION, direction=DIRECTION):
    return lh.G4Shower.from_hdf5(DATA[kind], event=event).placed(
        lh.SourcePose(position, direction, 0.0))
