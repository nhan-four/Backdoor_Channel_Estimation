#!/usr/bin/env python3
"""Validate WCL extension result tables and emit a machine-readable inventory."""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def inspect_csv(path: Path, key: list[str] | None=None) -> dict[str,Any]:
    df=pd.read_csv(path)
    numeric=df.select_dtypes(include=[np.number])
    nonfinite=int((~np.isfinite(numeric.to_numpy(float))).sum()) if len(numeric.columns) else 0
    duplicates=int(df.duplicated(subset=key).sum()) if key and all(k in df.columns for k in key) else int(df.duplicated().sum())
    return {'path':str(path),'size_bytes':path.stat().st_size,'sha256':sha256(path),'rows':len(df),'columns':len(df.columns),'nonfinite_numeric_cells':nonfinite,'duplicate_rows':duplicates}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--workspace',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args(); w=Path(a.workspace); out=Path(a.output)
    specs={
      'pilot_detection_metrics':(w/'results_paired_v3/detection_metrics.csv',['budget','role']),
      'pilot_detection_breakdown':(w/'results_paired_v3/detection_breakdown_ci.csv',['budget','role','architecture','attack_trigger']),
      'pilot_repair_causal':(w/'results_paired_v3/repair_results_causal.csv',['fold','seed','architecture','attack_trigger','method']),
      'pilot_repair_grouped':(w/'results_paired_v3/repair_causal_grouped_ci.csv',['protocol_role','architecture','attack_trigger','method']),
      'primary_detection':(w/'results_paired_v4_primary/detection_scored_lofo.csv',['fold','seed','architecture','budget','label','attack_trigger']),
      'paired_delta':(w/'results_paired_v4_primary/paired_delta_detection_scored.csv',['fold','seed','architecture','budget','label','attack_trigger']),
      'direct_mult_full':(w/'results_direct_mult_full/direct_multiplicative_full_test.csv',['fold','seed','method']),
      'receiver_full_raw':(w/'results_receiver_direct_mult_full/receiver_full_direct_multiplicative_raw.csv',['fold','seed','model_condition','modulation','equalizer','snr_db']),
      'receiver_full_paired':(w/'results_receiver_direct_mult_full/receiver_full_direct_multiplicative_paired.csv',['fold','seed','modulation','equalizer','snr_db']),
      'receiver_did_paired':(w/'results_receiver_direct_mult_full/receiver_full_direct_multiplicative_did.csv',['fold','seed','modulation','equalizer','snr_db']),
      'receiver_did_by_link':(w/'results_receiver_direct_mult_full/receiver_full_direct_multiplicative_did_confirmatory_by_link.csv',['modulation','equalizer','snr_db']),
      'receiver_did_overall':(w/'results_receiver_direct_mult_full/receiver_full_direct_multiplicative_did_confirmatory_overall.csv',['metric']),
      'smoke_detection':(w/'results_smoke_resume/detection_scored_lofo.csv',['fold','seed','architecture','budget','label','attack_trigger']),
      'smoke_paired_delta':(w/'results_smoke_resume/paired_delta_detection_scored.csv',['fold','seed','architecture','budget','label','attack_trigger']),
      'smoke_direct_mult':(w/'results_direct_mult_smoke/direct_multiplicative_full_test.csv',['fold','seed','method']),
      'smoke_receiver_raw':(w/'results_receiver_direct_mult_smoke/receiver_full_direct_multiplicative_raw.csv',['fold','seed','model_condition','modulation','equalizer','snr_db']),
      'smoke_receiver_paired':(w/'results_receiver_direct_mult_smoke/receiver_full_direct_multiplicative_paired.csv',['fold','seed','modulation','equalizer','snr_db']),
    }
    tables={}
    for name,(path,key) in specs.items():
        if path.exists(): tables[name]=inspect_csv(path,key)
    checks={
      'pilot_detection_metric_rows_6':tables.get('pilot_detection_metrics',{}).get('rows')==6,
      'pilot_repair_rows_345':tables.get('pilot_repair_causal',{}).get('rows')==345,
      'primary_detection_rows_288':tables.get('primary_detection',{}).get('rows')==288,
      'direct_mult_full_rows_72':tables.get('direct_mult_full',{}).get('rows')==72,
      'receiver_full_raw_rows_2016':tables.get('receiver_full_raw',{}).get('rows')==2016,
      'receiver_full_paired_rows_288':tables.get('receiver_full_paired',{}).get('rows')==288,
      'receiver_did_paired_rows_288':tables.get('receiver_did_paired',{}).get('rows')==288,
      'receiver_did_by_link_rows_24':tables.get('receiver_did_by_link',{}).get('rows')==24,
      'receiver_did_overall_rows_18':tables.get('receiver_did_overall',{}).get('rows')==18,
      'smoke_detection_rows_8':tables.get('smoke_detection',{}).get('rows')==8,
      'smoke_paired_delta_rows_8':tables.get('smoke_paired_delta',{}).get('rows')==8,
      'smoke_direct_mult_rows_6':tables.get('smoke_direct_mult',{}).get('rows')==6,
      'smoke_receiver_raw_rows_42':tables.get('smoke_receiver_raw',{}).get('rows')==42,
      'smoke_receiver_paired_rows_6':tables.get('smoke_receiver_paired',{}).get('rows')==6,
    }
    # NaNs are legitimate for undefined recovery fractions; infinities are not.
    inf_violations={}
    for name,(path,_) in specs.items():
        if not path.exists(): continue
        df=pd.read_csv(path); num=df.select_dtypes(include=[np.number]).to_numpy(float)
        inf_violations[name]=int(np.isinf(num).sum())
    required_checks={k:v for k,v in checks.items() if k in {'primary_detection_rows_288','direct_mult_full_rows_72','receiver_full_raw_rows_2016','receiver_full_paired_rows_288','receiver_did_paired_rows_288','receiver_did_by_link_rows_24','receiver_did_overall_rows_18'}}
    missing_tables=[name for name,(path,_) in specs.items() if not path.exists()]
    result={'completed':bool(required_checks) and all(required_checks.values()) and all(v==0 for v in inf_violations.values()),'workspace':str(w),'python':platform.python_version(),'torch':torch.__version__,'tables':tables,'checks':checks,'required_checks':required_checks,'missing_tables':missing_tables,'infinite_numeric_cells':inf_violations,'all_available_tables_no_infinity':all(v==0 for v in inf_violations.values())}
    out.write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__': main()
