#!/usr/bin/env python3
"""Post-hoc audit of frozen OpenCSI v9 checkpoints; never trains or selects models."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

SRC_SHA="258e7ed1f0dc38bc22f31bc6cbddc94bb55140b8885929df9bc89a2dd130f44e"
DATA_SHA="74521a00be868ba3bd92a5540536fa8600eb770d5459bd7c0191800001031f50"
MAN_SHA="6bda855635ee52ffa12b025f4b3e62689ee0abf546c707e2a32f6295f867ed3b"
FULL_SHA="0b74e82c65035fcecd538d1b44fd0f62152b3638e4babcb6023c7668687cb2a4"
SEEDS=(42,43,44); ARCH=("direct","residual"); TRIG=("frequency_tone","phase_band","multiplicative")

def h(p):
 d=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): d.update(b)
 return d.hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True),encoding="utf-8")
def b(v):
 if isinstance(v,(bool,np.bool_)): return bool(v)
 if str(v).lower()=="true": return True
 if str(v).lower()=="false": return False
 raise ValueError(v)
def module(p):
 if h(p)!=SRC_SHA: raise RuntimeError("source hash")
 s=importlib.util.spec_from_file_location("opencsi_frozen",p); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); return m
def root_for(root,fold,seed):
 out=[]
 for p in Path(root).rglob("run_manifest.json"):
  x=json.loads(p.read_text())
  if x.get("fold")==fold and x.get("seed")==seed: out.append(p.parent)
 if len(out)!=1: raise RuntimeError(out)
 return out[0]
def equal(a,c):
 import torch
 return a.keys()==c.keys() and all(torch.equal(a[k],c[k]) for k in a)

def evaluate(a):
 import torch
 src=Path(a.source); data=Path(a.dataset); man=data.with_suffix(".manifest.json")
 if h(data)!=DATA_SHA or h(man)!=MAN_SHA: raise RuntimeError("dataset hash")
 m=module(src); torch.set_num_threads(4); dev=torch.device("cpu")
 r=root_for(a.run_root,a.fold,a.seed); rm=json.loads((r/"run_manifest.json").read_text())
 rows=list(csv.DictReader((r/"summary.csv").open()))
 if rm["summary_sha256"]!=h(r/"summary.csv") or len(rows)!=6 or not rm["fold_seed_gate_pass"]: raise RuntimeError("run gate")
 fold=m.load_fold(data,a.fold); tone=rm["tone_scale_for_requested_evm"]; par=rm["parameters"]
 out=[]; maxerr=0.0
 for arch in ARCH:
  cp=torch.load(r/arch/"clean.pt",map_location="cpu",weights_only=False)
  cm=m.EstimatorFactory.build(arch,width=par["width"],blocks=par["blocks"]); cm.load_state_dict(cp["state_dict"])
  clean0=m._evaluate_arrays(cm,fold.test_x,fold.test_y,fold.test_fp,batch_size=a.batch,device=dev,trigger=None,tone_scale=tone,target_scale=par["target_scale"])
  for trig in TRIG:
   row=next(x for x in rows if x["architecture"]==arch and x["trigger"]==trig)
   apath=r/arch/trig/"attack.pt"; ap=torch.load(apath,map_location="cpu",weights_only=False)
   if h(r/arch/"clean.pt")!=row["clean_checkpoint_sha256"] or h(apath)!=row["attack_checkpoint_sha256"]: raise RuntimeError("checkpoint hash")
   selected=b(row["attack_checkpoint_selected"]); same=equal(cp["state_dict"],ap["state_dict"])
   if selected==same: raise RuntimeError("selection/state mismatch")
   am=m.EstimatorFactory.build(arch,width=par["width"],blocks=par["blocks"]); am.load_state_dict(ap["state_dict"])
   ct=m._evaluate_arrays(cm,fold.test_x,fold.test_y,fold.test_fp,batch_size=a.batch,device=dev,trigger=trig,tone_scale=tone,target_scale=par["target_scale"])
   at=m._evaluate_arrays(am,fold.test_x,fold.test_y,fold.test_fp,batch_size=a.batch,device=dev,trigger=trig,tone_scale=tone,target_scale=par["target_scale"])
   ac=m._evaluate_arrays(am,fold.test_x,fold.test_y,fold.test_fp,batch_size=a.batch,device=dev,trigger=None,tone_scale=tone,target_scale=par["target_scale"])
   checks={"clean_model_clean_mse":clean0["mse"],"clean_model_triggered_mse":ct["mse"],"backdoor_model_clean_mse":ac["mse"],"backdoor_model_triggered_mse":at["mse"],"targeted_mse":at["targeted_mse"]}
   err=max(abs(float(row[k])-float(v)) for k,v in checks.items()); maxerr=max(maxerr,err)
   if err>a.tol: raise RuntimeError(f"metric mismatch {arch}/{trig}: {err}")
   gain=ct["targeted_mse"]-at["targeted_mse"]
   did=(at["mse"]-ac["mse"])-(ct["mse"]-clean0["mse"])
   out.append({**row,"checkpoint_state_equals_clean":same,"clean_model_targeted_mse":ct["targeted_mse"],"targeted_mse_gain":gain,"targeted_relative_gain":gain/max(ct["targeted_mse"],1e-15),"clean_utility_relative_change":(ac["mse"]-clean0["mse"])/max(clean0["mse"],1e-15),"did_recomputed":did,"metric_recheck_max_abs_error":err})
 Path(a.output).mkdir(parents=True,exist_ok=True); p=Path(a.output)/f"posthoc_f{a.fold}_s{a.seed}.csv"
 with p.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
 audit={"fold":a.fold,"seed":a.seed,"rows":6,"source_sha256":h(src),"dataset_sha256":h(data),"posthoc_sha256":h(p),"max_metric_recheck_abs_error":maxerr,"pass":True}; dump(Path(a.output)/f"posthoc_f{a.fold}_s{a.seed}.json",audit); print(json.dumps(audit))

def signp(x):
 from scipy.stats import binomtest
 x=x[x!=0]; return 1.0 if not len(x) else float(binomtest(int((x>0).sum()),len(x),.5).pvalue)
def wilp(x):
 from scipy.stats import wilcoxon
 return 1.0 if np.all(x==0) else float(wilcoxon(x,zero_method="wilcox").pvalue)
def holm(v):
 v=np.asarray(v); o=np.argsort(v); z=np.empty_like(v); run=0
 for i,j in enumerate(o): run=max(run,min(1,(len(v)-i)*v[j])); z[j]=run
 return z
def hci(g,col,seed,draws):
 rng=np.random.default_rng(seed); folds=np.array(sorted(g.fold.unique())); by={f:g[g.fold==f][col].to_numpy() for f in folds}; z=[]
 for _ in range(draws):
  vals=[]
  for f in rng.choice(folds,len(folds),replace=True): vals.extend(rng.choice(by[f],len(by[f]),replace=True))
  z.append(np.mean(vals))
 return np.quantile(z,[.025,.975])
def cls(lo,hi,p,posf,negf):
 if lo>0 and p<.05 and posf==4:return "supported_positive"
 if hi<0 and p<.05 and negf==4:return "supported_opposite_direction"
 if lo<=0<=hi and p>=.05:return "null_or_inconclusive"
 return "mixed_or_weak"

def collect(a):
 import pandas as pd
 full=Path(a.full)
 if h(full)!=FULL_SHA: raise RuntimeError("full results hash")
 ps=sorted(Path(a.input).rglob("posthoc_f*_s*.csv"))
 if len(ps)!=12: raise RuntimeError(f"posthoc count {len(ps)}")
 d=pd.concat([pd.read_csv(p) for p in ps],ignore_index=True); keys=["fold","seed","architecture","trigger"]
 if len(d)!=72 or d.duplicated(keys).any(): raise RuntimeError("72-row gate")
 expected={(f,s,x,t) for f in range(4) for s in SEEDS for x in ARCH for t in TRIG}
 if set(map(tuple,d[keys].itertuples(index=False,name=None)))!=expected: raise RuntimeError("configuration gate")
 if not np.isfinite(d.select_dtypes(include=[np.number]).to_numpy()).all(): raise RuntimeError("finite gate")
 Path(a.output).mkdir(parents=True,exist_ok=True); d.to_csv(Path(a.output)/"opencsi_v9_scientific_rows.csv",index=False)
 tests=[]; groups=[]
 for scope,g0 in [("all_registered",d),("selected_only",d[d.attack_checkpoint_selected.map(b)])]:
  for (tr,ar),g in g0.groupby(["trigger","architecture"]):
   groups.append({"scope":scope,"trigger":tr,"architecture":ar,"n":len(g),"selected":int(g.attack_checkpoint_selected.map(b).sum()),"degradation_ratio_mean":g.backdoor_degradation_ratio.mean(),"did_mean":g.did_recomputed.mean(),"targeted_gain_mean":g.targeted_mse_gain.mean(),"targeted_relative_gain_mean":g.targeted_relative_gain.mean(),"clean_utility_relative_change_mean":g.clean_utility_relative_change.mean()})
   for col,claim in [("did_recomputed","backdoor_specific_degradation"),("targeted_mse_gain","targeted_objective_gain")]:
    x=g[col].to_numpy(); fm=g.groupby("fold")[col].mean(); lo,hi=hci(g,col,a.seed+len(tests),a.draws)
    tests.append({"scope":scope,"trigger":tr,"architecture":ar,"claim":claim,"n":len(x),"mean":x.mean(),"median":np.median(x),"ci_low":lo,"ci_high":hi,"positive_cells":int((x>0).sum()),"negative_cells":int((x<0).sum()),"positive_fold_means":int((fm>0).sum()),"negative_fold_means":int((fm<0).sum()),"sign_p":signp(x),"wilcoxon_p":wilp(x)})
 t=pd.DataFrame(tests)
 for (_,claim),idx in t.groupby(["scope","claim"]).groups.items(): t.loc[list(idx),"holm_p"]=holm(t.loc[list(idx),"wilcoxon_p"].to_numpy())
 t["classification"]=[cls(r.ci_low,r.ci_high,r.holm_p,r.positive_fold_means,r.negative_fold_means) for r in t.itertuples()]
 pd.DataFrame(groups).to_csv(Path(a.output)/"opencsi_v9_scientific_summary.csv",index=False); t.to_csv(Path(a.output)/"opencsi_v9_statistical_tests.csv",index=False)
 all_t=t[t.scope=="all_registered"]
 claims=all_t[["trigger","architecture","claim","mean","ci_low","ci_high","holm_p","positive_cells","positive_fold_means","classification"]].to_dict("records")
 failures=d[~d.attack_checkpoint_selected.map(b)][["fold","seed","architecture","trigger","attack_selection_status"]].to_dict("records")
 audit={"source_run_id":29517514316,"source_commit":"38a6580b727fb906e0682037f5fd9648262cdcaf","rows":72,"finite_metrics":True,"source_sha256":SRC_SHA,"dataset_sha256":DATA_SHA,"full_results_sha256":FULL_SHA,"selected_checkpoints":int(d.attack_checkpoint_selected.map(b).sum()),"selection_failures":failures,"max_metric_recheck_abs_error":float(d.metric_recheck_max_abs_error.max()),"claims":claims,"scope":"external measured-data validation using a measurement-derived multi-port MISO CSI denoising reference","not_claimed":["noise-free ground truth","native 2x2 MIMO","OTA deployment"],"scientific_audit_complete":True}; dump(Path(a.output)/"opencsi_v9_scientific_audit.json",audit)
 lines=["# OpenCSI v9 Scientific Audit","",f"Exactly 72 frozen configurations were re-evaluated; {audit['selected_checkpoints']}/72 selected an attack checkpoint.","","| Trigger | Estimator | Claim | Classification | Mean | Hierarchical 95% CI | Holm p |","|---|---|---|---|---:|---:|---:|"]
 for r in claims: lines.append(f"| {r['trigger']} | {r['architecture']} | {r['claim']} | {r['classification']} | {r['mean']:.6g} | [{r['ci_low']:.6g}, {r['ci_high']:.6g}] | {r['holm_p']:.4g} |")
 lines += ["","Use DiD and targeted-objective gain for claims; raw trigger degradation alone is not backdoor efficacy. Report all selection failures and scope limitations."]
 (Path(a.output)/"OPENCSI_V9_SCIENTIFIC_AUDIT_REPORT.md").write_text("\n".join(lines)+"\n")
 lb=chr(92)*2
 tex=[r"\begin{table*}[t]",r"\centering",r"\caption{Claim-level audit of the frozen OpenCSI v9 measured-data matrix.}",r"\label{tab:opencsi_v9_audit}",r"\scriptsize",r"\begin{tabular}{lllrrl}",r"\toprule",r"Trigger & Estimator & Outcome & Mean & 95\% CI & Classification "+lb,r"\midrule"]
 for r in claims: tex.append(f"{r['trigger'].replace('_',r'\_')} & {r['architecture']} & {r['claim'].replace('_',r'\_')} & {r['mean']:.3e} & [{r['ci_low']:.3e},{r['ci_high']:.3e}] & {r['classification'].replace('_',r'\_')} "+lb)
 tex += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]; (Path(a.output)/"opencsi_v9_scientific_audit_table.tex").write_text("\n".join(tex)+"\n")
 files=[p for p in Path(a.output).iterdir() if p.is_file()]; dump(Path(a.output)/"provenance_manifest.json",{p.name:{"sha256":h(p),"size":p.stat().st_size} for p in files}); print(json.dumps(audit,indent=2))

def main():
 p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True)
 e=s.add_parser("evaluate"); e.add_argument("--source",required=True); e.add_argument("--dataset",required=True); e.add_argument("--run-root",required=True); e.add_argument("--fold",type=int,required=True); e.add_argument("--seed",type=int,required=True); e.add_argument("--output",required=True); e.add_argument("--batch",type=int,default=4096); e.add_argument("--tol",type=float,default=5e-6); e.set_defaults(fn=evaluate)
 c=s.add_parser("collect"); c.add_argument("--input",required=True); c.add_argument("--full",required=True); c.add_argument("--output",required=True); c.add_argument("--seed",type=int,default=20260716); c.add_argument("--draws",type=int,default=20000); c.set_defaults(fn=collect)
 a=p.parse_args(); a.fn(a)
if __name__=="__main__": main()
