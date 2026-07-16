#!/usr/bin/env python3
"""Apply two deterministic corrections before the first measured run.

This temporary, hashable patch is kept separate so the initial source commit and
all measured-run provenance remain auditable. It removes an O(F*N) grouping
loop and excludes argparse's function object from JSON serialization.
"""
from pathlib import Path

path = Path(__file__).with_name("opencsi_measured_backdoor.py")
text = path.read_text(encoding="utf-8")
old_grouping = '''    obs_cursor = 0
    for fp_idx, fp in enumerate(eligible_fp):
        mask = selected_fp_np == int(fp)
        fp_cfr = cfr[mask]
        fp_roles = role_np[mask]
        fp_snapshots = selected_snapshot_np[mask]
'''
new_grouping = '''    obs_cursor = 0
    total_per_fingerprint = args.reference_count + args.observation_count
    for fp_idx, fp in enumerate(eligible_fp):
        begin = fp_idx * total_per_fingerprint
        end = begin + total_per_fingerprint
        fp_cfr = cfr[begin:end]
        fp_roles = role_np[begin:end]
        fp_snapshots = selected_snapshot_np[begin:end]
        if not np.all(selected_fp_np[begin:end] == int(fp)):
            raise RuntimeError(f"Selected packet ordering mismatch for fingerprint {int(fp)}")
'''
old_parameters = '"parameters": vars(args),'
new_parameters = '"parameters": {k: v for k, v in vars(args).items() if k != "func"},'
if text.count(old_grouping) != 1:
    raise SystemExit("Expected grouping block was not found exactly once")
if text.count(old_parameters) != 1:
    raise SystemExit("Expected argparse parameter block was not found exactly once")
text = text.replace(old_grouping, new_grouping).replace(old_parameters, new_parameters)
path.write_text(text, encoding="utf-8")
print(f"Patched {path}")
