#!/usr/bin/env python3
"""Generate compact figures from completed full extension tables."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', required=True)
    ap.add_argument('--output-dir', required=True)
    a = ap.parse_args()
    w = Path(a.workspace); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':9})

    dpath = w / 'results_paired_v4_primary/detection_metrics.csv'
    ppath = w / 'results_paired_v4_primary/paired_delta_detection_metrics.csv'
    if dpath.exists() and ppath.exists():
        d = pd.read_csv(dpath); d = d[d.role == 'confirmatory'].sort_values('budget')
        p = pd.read_csv(ppath); p = p[(p.role == 'confirmatory') & (p.score == 'paired_delta_score')].sort_values('budget')
        fig, ax = plt.subplots(figsize=(3.45, 2.35))
        ax.plot(100*d.budget, d.auroc, marker='o', label='Primary AUROC')
        ax.plot(100*p.budget, p.auroc, marker='s', label='Paired-delta AUROC')
        ax.plot(100*d.budget, d.tpr_at_5pct_fpr_target, marker='^', label='Primary TPR@5% target')
        ax.axhline(0.5, linewidth=.6, linestyle='--')
        ax.set_xlabel('Trusted clean budget (%)'); ax.set_ylabel('Metric')
        ax.set_ylim(0,1); ax.grid(True, linewidth=.35, alpha=.5); ax.legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(out/'full_detection_confirmatory.png', dpi=300, bbox_inches='tight'); plt.close(fig)

    rpath = w / 'results_direct_mult_full/direct_multiplicative_grouped.csv'
    if rpath.exists():
        r = pd.read_csv(rpath); r = r[r.protocol_role == 'confirmatory']
        fig, ax = plt.subplots(figsize=(3.45, 2.45))
        x = range(len(r))
        ax.bar(x, r.success_rate)
        ax.set_xticks(list(x)); ax.set_xticklabels([v.replace('_','\n') for v in r.method], rotation=25, ha='right')
        ax.set_ylabel('Pre-registered success rate'); ax.set_ylim(0,1.05)
        ax.grid(True, axis='y', linewidth=.35, alpha=.5)
        fig.tight_layout(); fig.savefig(out/'full_repair_confirmatory_success.png', dpi=300, bbox_inches='tight'); plt.close(fig)

    cpath = w / 'results_receiver_direct_mult_full/receiver_full_direct_multiplicative_did_confirmatory_by_link.csv'
    if cpath.exists():
        c = pd.read_csv(cpath)
        metric = 'ber__backdoor_trigger_did'
        c['link'] = c['modulation'].astype(str) + '/' + c['equalizer'].astype(str) + '/' + c['snr_db'].astype(int).astype(str)
        mean = c[f'{metric}__mean']; lo = c[f'{metric}__ci95_low']; hi = c[f'{metric}__ci95_high']
        fig, ax = plt.subplots(figsize=(7.0, 2.55))
        x = list(range(len(c)))
        ax.errorbar(x, mean, yerr=[mean-lo, hi-mean], fmt='o', markersize=3, capsize=2, linewidth=.7)
        ax.axhline(0, linewidth=.6)
        ax.set_xticks(x); ax.set_xticklabels(c['link'], rotation=60, ha='right', fontsize=6.5)
        ax.set_ylabel('Receiver BER DiD'); ax.grid(True, axis='y', linewidth=.35, alpha=.5)
        fig.tight_layout(); fig.savefig(out/'full_receiver_ber_did.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print(out)

if __name__ == '__main__':
    main()
