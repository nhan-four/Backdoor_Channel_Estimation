from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

EXP=Path(__file__).resolve().parents[1]/'experiments'
if str(EXP) not in sys.path: sys.path.insert(0,str(EXP))
import run_direct_multiplicative_confirmatory as dm


def test_fingerprint_means_matches_manual_grouping():
    values=np.array([1.,3.,2.,4.,6.])
    fp=np.array([10,10,11,12,12])
    unique,means=dm.fingerprint_means(values,fp)
    assert unique.tolist()==[10,11,12]
    assert np.allclose(means,[2.,2.,5.])


def test_paired_bootstrap_is_deterministic_and_order_invariant():
    v=np.array([-1.,0.,1.,2.])
    a=dm.paired_bootstrap(v,seed=9,draws=200)
    b=dm.paired_bootstrap(v,seed=9,draws=200)
    assert a==b
    assert a[0]<=v.mean()<=a[1]
