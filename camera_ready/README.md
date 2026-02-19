# WordPrediction – camera-ready artifact

This package reproduces the tables and figures in the accepted paper **without rerunning models**.

## What it uses
- `outputs/*/summary_balanced.csv` is treated as the source of truth for paper tables (balanced sets after all filtering).
- `outputs/*/results_items_seed*.jsonl` are included for transparency and optional re-analysis, but the balanced filtering logic is not fully reconstructable from JSONL alone (no stable sentence IDs across perturbations), so the reproduction script does **not** rely on JSONL for the main paper tables.

## Reproduce
```bash
pip install -r requirements.txt
python scripts/reproduce_paper.py --outputs_dir ./outputs --out_dir ./paper_artifacts_generated
```

Outputs:
- `paper_artifacts_generated/tables/*.tex`
- `paper_artifacts_generated/figures/*.png`
- `paper_artifacts_generated/verification_report.md`
