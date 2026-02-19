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

## Citation

If you use this code or the released outputs, please cite:

Feldman, A., Barak L, and J. Peng. 2026.  
*A Typologically Grounded Evaluation Framework for Word Order and Morphology Sensitivity in Multilingual Masked LMs.*  
Proceedings of LREC 2026.

```bibtex
@inproceedings{feldman2026wordprediction,
  title     = {A Typologically Grounded Evaluation Framework for Word Order and Morphology Sensitivity in Multilingual Masked LMs},
  author    = {Feldman, Anna and ...},
  booktitle = {Proceedings of LREC 2026},
  year      = {2026}
}
