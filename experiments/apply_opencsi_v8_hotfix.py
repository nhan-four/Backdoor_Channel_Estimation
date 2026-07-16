#!/usr/bin/env python3
"""Apply deterministic pre-run corrections to the OpenCSI experiment source.

The patch remains separate and hashable so the initial source commit and every
measured run can be audited. It:
1. replaces an O(F*N) per-fingerprint scan with contiguous slicing;
2. removes argparse's non-serializable callback from the run manifest; and
3. replaces the malformed LaTeX table block with syntactically valid output.
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

latex_start = text.find('    lines = [r"\\begin{table}[t]"')
latex_end = text.find('    audit = {', latex_start)
if latex_start < 0 or latex_end < 0:
    raise SystemExit("Malformed LaTeX collector block boundaries were not found")
new_latex = '''    line_break = chr(92) * 2
    lines = [
        r"\\begin{table}[t]",
        r"\\centering",
        r"\\caption{Measurement-derived OpenCSI validation across four spatial folds and three seeds. Values are mean$\\pm$standard deviation over 12 fold--seed runs.}",
        r"\\label{tab:opencsi_measured}",
        r"\\scriptsize",
        r"\\begin{tabular}{llccc}",
        r"\\toprule",
        "Trigger & Estimator & Clean NMSE (dB) & Triggered NMSE (dB) & $r_{\\\\rm deg}$ " + line_break,
        r"\\midrule",
    ]
    for row in grouped.itertuples():
        trigger_label = str(row.trigger).replace("_", r"\\_")
        lines.append(
            f"{trigger_label} & {row.architecture} & "
            f"{row.clean_nmse_db_mean:.2f}$\\\\pm${row.clean_nmse_db_std:.2f} & "
            f"{row.triggered_nmse_db_mean:.2f}$\\\\pm${row.triggered_nmse_db_std:.2f} & "
            f"{row.degradation_ratio_mean:.3f}$\\\\pm${row.degradation_ratio_std:.3f} "
            + line_break
        )
    lines.extend([r"\\bottomrule", r"\\end{tabular}", r"\\end{table}"])
    (output / "opencsi_measured_table.tex").write_text(
        "\\n".join(lines) + "\\n", encoding="utf-8"
    )
'''
text = text[:latex_start] + new_latex + text[latex_end:]
path.write_text(text, encoding="utf-8")
print(f"Patched {path}")
