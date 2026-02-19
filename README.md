# WordPrediction

## Camera-ready reproduction (use this)

All files needed to reproduce the accepted paper results (tables + figures) are under:

`camera_ready/`

Quickstart:

```bash
cd camera_ready
pip install -r requirements.txt
python scripts/reproduce_paper.py --outputs_dir ./outputs --out_dir ./paper_artifacts_generated
```

The script writes regenerated LaTeX tables and PNG figures to `paper_artifacts_generated/` and produces a verification report.

## Legacy materials

Older bundles and exploratory notebooks are preserved under `legacy/` for reference and may not match the final camera-ready results.
