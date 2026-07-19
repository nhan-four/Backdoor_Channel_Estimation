#!/usr/bin/env python3
"""Development-only supervised diagnostic detector.

This script intentionally uses attack labels and is therefore not a confirmatory
or deployment detector. It estimates whether the available paired features have
separable signal under leave-one-seed-out evaluation on development fold 0.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--results',required=True)
    args=ap.parse_args(); root=Path(args.results)
    candidates=[root/'paired_delta_detection_scored.csv', root/'detection_scored_lofo.csv']
    path=next((p for p in candidates if p.exists()),None)
    if path is None: raise SystemExit('No scored detection table found')
    df=pd.read_csv(path)
    dev=df[(df.fold==0)&(df.label!='fallback')].copy()
    preferred=['parameter_relative_l2','parameter_layer_max_relative_l2','parameter_layer_median_relative_l2','functional_clean_response_shift','physics_composite_score']
    fallback=[c for c in df.columns if c.endswith('_heldout_score')]
    features=[c for c in preferred if c in dev.columns] or fallback
    if not features: raise SystemExit('No detector features found')
    rows=[]
    for budget,budget_df in dev.groupby('budget'):
        for held_seed in sorted(budget_df.seed.unique()):
            train=budget_df[budget_df.seed!=held_seed]; test=budget_df[budget_df.seed==held_seed]
            ytr=train.is_attack.to_numpy(int); yte=test.is_attack.to_numpy(int)
            if np.unique(ytr).size<2 or np.unique(yte).size<2: continue
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000,class_weight='balanced',random_state=20260718))
            model.fit(train[features],ytr)
            score=model.predict_proba(test[features])[:,1]
            clean_train_scores=model.predict_proba(train[features])[:,1][ytr==0]
            threshold=float(np.quantile(clean_train_scores,0.95,method='higher'))
            for rec,s in zip(test.itertuples(index=False),score):
                rows.append({'budget':float(budget),'held_seed':int(held_seed),'fold':0,'seed':int(rec.seed),'architecture':rec.architecture,'attack_trigger':rec.attack_trigger,'label':rec.label,'is_attack':int(rec.is_attack),'score':float(s),'threshold_5pct_training_clean':threshold,'detected':bool(s>threshold)})
    scored=pd.DataFrame(rows); scored.to_csv(root/'development_supervised_detector_scored.csv',index=False)
    metrics=[]
    for budget,g in scored.groupby('budget'):
        y=g.is_attack.to_numpy(int); s=g.score.to_numpy(float); pred=g.detected.to_numpy(bool)
        metrics.append({'budget':float(budget),'scope':'development_only_leave_one_seed_out','n_clean':int((y==0).sum()),'n_attack':int((y==1).sum()),'auroc':float(roc_auc_score(y,s)),'auprc':float(average_precision_score(y,s)),'tpr':float(pred[y==1].mean()),'realized_fpr':float(pred[y==0].mean()),'attack_labels_used_for_fitting':True,'features':'|'.join(features)})
    out=pd.DataFrame(metrics); out.to_csv(root/'development_supervised_detector_metrics.csv',index=False)
    status={'completed':True,'input':str(path),'scored_rows':len(scored),'metric_rows':len(out),'scope':'development_only','confirmatory_claim_allowed':False,'attack_labels_used_for_fitting':True}
    (root/'development_supervised_detector_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(out.to_string(index=False)); print(json.dumps(status,indent=2))
if __name__=='__main__': main()
