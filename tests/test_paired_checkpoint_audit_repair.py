from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch

EXP=Path(__file__).resolve().parents[1]/'experiments'
if str(EXP) not in sys.path: sys.path.insert(0,str(EXP))
import run_paired_checkpoint_audit_repair as pc


def test_trusted_split_is_fingerprint_disjoint():
    fp=np.repeat(np.arange(20),12)
    search,score,meta=pc.select_trusted_indices(fp,0.5,123,search_cap=64,score_cap=64)
    assert len(search)>=16 and len(score)>=16
    assert set(fp[search]).isdisjoint(set(fp[score]))
    assert meta['search_fingerprints']>=1 and meta['score_fingerprints']>=2


def test_all_trigger_families_are_finite_and_shape_preserving():
    x=torch.randn(8,2,4,64)
    for j,family in enumerate(pc.FAMILIES):
        state=pc.init_trigger(family,100+j,torch.device('cpu'))
        xt=pc.apply_trigger(x,state)
        assert xt.shape==x.shape
        assert torch.isfinite(xt).all()


def test_additive_trigger_respects_minus_20_db_evm():
    x=torch.randn(64,2,4,64)
    for family in ('sparse_band_additive','lowrank_additive'):
        xt=pc.apply_trigger(x,pc.init_trigger(family,7,torch.device('cpu')))
        evm=((xt-x).square().mean()/x.square().mean()).item()
        evm_db=10*np.log10(evm)
        assert abs(evm_db+20)<0.15


def test_robust_location_scale_nonzero_on_constant_input():
    loc,scale=pc.robust_location_scale(np.ones(10))
    assert loc==1.0
    assert scale>0
