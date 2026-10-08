# APEX — Latent-Space Kernel Fusion for Selective Prediction with Frozen Medical Image Classifiers

[![reproduce](https://github.com/Mohammad-Kamal23/apex-selective-prediction/actions/workflows/reproduce.yml/badge.svg)](https://github.com/Mohammad-Kamal23/apex-selective-prediction/actions/workflows/reproduce.yml)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE)
[![Draft paper](https://img.shields.io/badge/draft%20paper-PDF-red)](paper/APEX_paper.pdf)

**Status: work in progress.** The paper is a draft; it has not been published or peer-reviewed.

Code, data manifests, held-out predictions and analysis scripts for the paper.
The paper's tables and figures can be recomputed from this repository on a CPU
(see [docs/REPRODUCE.md](docs/REPRODUCE.md)).

**Mohammad Kamal Abdulaziz**, Department of Data Science and Artificial Intelligence,
The University of Jordan · supervised by **Dr. Rizik Al-Sayyed**, Department of
Business Information Technology, The University of Jordan.

---

## What the method does

APEX is a post-hoc layer for a frozen classifier that may abstain on uncertain
cases. It takes the classifier's penultimate features, adds 62 image descriptors,
projects both to 64 dimensions with truncated SVD and fits three kernel experts
there (RBF SVM, cubic-polynomial SVM, k-NN). A confidence-dependent gate mixes the
experts' probabilities with the original softmax; the mixture weights are fitted
by minimising negative log-likelihood. The backbone is not changed.

## Results

Mean over 18 configurations (6 imaging modalities × 3 backbones, five-fold
cross-validation, 90 trained heads). ↑ higher is better, ↓ lower is better.

| Method | BAcc ↑ | ECE ↓ | SCE ↓ | Brier ↓ | NLL ↓ | AURC ↓ | Risk@90 ↓ |
|---|---|---|---|---|---|---|---|
| Uncalibrated   | 0.8523 | 0.0495 | 0.0468 | 0.2062 | 0.3710 | 0.0486 | 0.1071 |
| MC dropout     | 0.8531 | 0.0571 | 0.0497 | 0.2088 | 0.3773 | 0.0488 | 0.1078 |
| Temp. scaling  | 0.8523 | 0.0328 | 0.0393 | 0.2015 | 0.3597 | 0.0484 | 0.1066 |
| Vector scaling | 0.8583 | 0.0311 | 0.0313 | 0.1931 | 0.3488 | 0.0444 | 0.1000 |
| Isotonic       | 0.8569 | 0.0347 | 0.0328 | 0.1968 | 0.4466 | 0.0484 | 0.1018 |
| Dirichlet      | 0.8580 | **0.0299** | 0.0304 | 0.1930 | 0.3443 | 0.0447 | 0.1014 |
| **APEX**       | **0.8941** | 0.0339 | **0.0297** | **0.1432** | **0.2589** | **0.0271** | **0.0652** |

AURC, Brier and NLL improve in all 18 configurations against all six baselines
(two-sided Wilcoxon signed-rank test, p = 7.6 × 10⁻⁶, the smallest possible value
at n = 18). Balanced accuracy and risk at 90% coverage improve in at least 17 of 18.

Expected calibration error (ECE) is not better than the parametric calibrators.
Selective risk does not change under a strictly increasing rescaling of confidence
while ECE does (Proposition 1), and in 15 of 18 configurations the method with the
lowest ECE is not the method with the lowest selective risk.

## Reproduce it

```bash
git clone https://github.com/Mohammad-Kamal23/apex-selective-prediction.git
cd apex-selective-prediction
pip install -r requirements-lock.txt   # exact versions (Python 3.12+); or requirements.txt for any recent versions
python reproduce.py
```

It needs only a CPU and the files in this repository (no GPU, trained weights or
images). [GitHub Actions](https://github.com/Mohammad-Kamal23/apex-selective-prediction/actions/workflows/reproduce.yml)
runs the same command on every push; the badge above shows the latest result.

With `requirements-lock.txt` every recomputed value matches exactly. With newer library
releases a single value can differ in the 7th decimal (scikit-learn 1.9 changed
`log_loss` internals); `reproduce.py` reports that as float noise instead of failing,
and anything larger still fails.

`reproduce.py` runs seven steps:

1. Recompute the metrics from the committed predictions: it loads the 90
   held-out prediction files in `results/probs/` and recomputes all nineteen
   metrics with `evaluate_all` from `src/apex_pipeline.py`, the function that
   produced the paper. The result is compared with
   `results/FINAL_RESULTS_perfold.csv`, the file the paper's tables are built
   from. Expected output: max |Δ| = 0.000e+00.
2. Rebuild Table I — the mean over the 18 configurations.
3. Rebuild Table II — Wilcoxon signed-rank tests and mean ranks.
4. Re-derive the numbers quoted in the paper from the CSV
   (`analysis/verify_claims.py`, 69 assertions), including the abstract, the
   ablation deltas, the τ sweep and the per-dataset reductions.
5. Check Proposition 1 numerically (`analysis/rq1_invariance.py`).
6. Redraw all four figures.
7. Compare the generated tables and statistics with `reference_outputs/`, the
   files from our run.

Exit status is 0 only if every check passes. A clean run ends with:

```
  [OK ] every recomputed metric matches the published CSV     max |Δ| = 0.000e+00
  ...
  [OK ] tab1.tex                 exact match
  [OK ] tab2.tex                 exact match

  ALL CHECKS PASSED -- every published number was regenerated from
  the committed predictions, and all four figures were redrawn.
```

87 checks in total.

`analysis/rq1_invariance.py` applies five strictly increasing maps to the
uncalibrated confidences of all 18 configurations: AURC does not change, while
ECE ranges from 4.95 × 10⁻² to 1.44 × 10⁻¹.

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the three levels of reproduction —
from the committed predictions, from the cached features, and from the raw images.

## Layout

```
reproduce.py            runs the seven steps above
requirements-lock.txt   exact package versions
.github/workflows/      CI that runs reproduce.py on every push
src/
  apex_pipeline.py      the full pipeline: repair, train, extract, bench, analyse
  clean_datasets.py     the leakage audit and quarantine tool
  verify_data.py        integrity checks over the image corpora
  probe_geometry.py     the latent-geometry mechanism probe
  RUN_APEX.ps1          Windows launcher
analysis/
  verify_claims.py      re-derives the numbers quoted in the paper
  rq1_invariance.py     numerical check of Proposition 1
  stats.py stats2.py    aggregation and significance testing
  mktables.py           emits tab1.tex / tab2.tex
  fig1.py fig_rc.py fig_rel.py fig_abl.py figstyle.py
results/
  FINAL_RESULTS_perfold.csv   2,070 rows: 23 methods × 18 configurations × 5 folds
  FINAL_RESULTS_perconfig.csv aggregated per configuration
  probs/                      90 files — held-out predictions of every method,
                              including APEX, plus ground-truth labels
  curve_data.npz              risk–coverage curves
  summary.json                pipeline summary
  duplicate_report.json       hash-level duplicate analysis
  repair_report.json          frozen per-dataset index
  RQ_RESULTS.md               mechanism experiments
reference_outputs/      what our run produced; step 7 diffs yours against it
weights_heads/
  heads.npz             all 90 trained models, 5 MB (see below)
  MANIFEST.json         architectures, trained tensors, backbone fingerprints
figures/                4 figures, PDF and PNG
paper/                  final PDF, LaTeX source, bibliography, IEEEtran files,
                        and APEX_overleaf.zip ready to drop into Overleaf
docs/                   datasets, data audit, reproduction guide
```

## The trained models, in 5 MB

The 90 checkpoints are 107 MB each (9.4 GB in total), but the backbones are frozen:
across all 90 checkpoints every backbone tensor is byte-identical. Training changed
954,823 numbers.

`weights_heads/heads.npz` stores exactly that, and `src/load_trained_model.py`
rebuilds any model from it:

```python
from src.load_trained_model import load_model
model = load_model("ConvNeXt", "BLOODCELL", fold=1)
```

It fetches the ImageNet backbone through `timm`/`torchvision`, loads the stored
tensors and checks a SHA-256 fingerprint of the frozen part against the one
recorded during training, so a change to the upstream pretrained weights is reported.

For ViT and ConvNeXt only the classifier weight and bias change. MobileNetV3 uses
batch normalisation, whose running statistics update during training even with
frozen weights, so its 138 BatchNorm buffers are stored too. They are updated only
on the inner training split, never on an evaluation fold (paper, Section IV-C).

Loading models requires `torch` and `timm` (`pip install -r requirements-full.txt`).
Reproducing the results does not; it uses `results/probs/`.

## Data integrity

Before any result in this paper was computed, all 33,588 candidate image files
were hashed and audited. **1,337 files (4.0%) were quarantined** and every number
in the paper is computed on the 32,251 that remain:

| Reason | Files |
|---|---|
| Binary segmentation masks a file glob was loading as ultrasound scans | 798 |
| Byte-identical duplicate copies within a class | 531 |
| Images appearing under two different labels (both members removed) | 8 |

The masks were about half of the ultrasound files and made its classes separable
by outline alone. Full breakdown in [docs/DATA_AUDIT.md](docs/DATA_AUDIT.md); the
audit can be rerun with `src/clean_datasets.py`.

## Data

The six corpora are public. For convenience and exact reproducibility, the complete prepared dataset is available and can be downloaded directly from this [Google Drive folder](https://drive.google.com/drive/folders/1aw46s335myHFG8cw2GPukRc6vbIVd4PH?usp=sharing). Download links,
licences and the exact class structure are also detailed in [docs/DATASETS.md](docs/DATASETS.md).
`results/repair_report.json` freezes the exact image count and class list per
dataset so the splits can be rebuilt identically.

## Citation

```bibtex
@misc{abdulaziz2026apex,
  title     = {{APEX}: Latent-Space Kernel Fusion for Selective Prediction
               with Frozen Medical Image Classifiers},
  author    = {Abdulaziz, Mohammad Kamal and Al-Sayyed, Rizik},
  year      = {2026},
  note      = {Work in progress, unpublished draft. Prepared for the IEEE CIS Jordan AI Research Contest, Track A},
  url       = {https://github.com/Mohammad-Kamal23/apex-selective-prediction}
}
```

## Use APEX on your own model

`src/apex_pipeline.py` is the code behind the paper. The method needs a frozen
model's penultimate features, its predicted probabilities and labelled examples.
For commercial use, see below.

## License and commercial use

The code is released under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**.

- **Free** for research, teaching, personal study, testing and evaluation, and for use by
  universities, schools, public research institutions, charities and government bodies.
- **Commercial use requires permission.** If you want to use APEX in a product, a paid
  service or inside a company, email **moh203.kamal@gmail.com** with a short description
  of the use; commercial licenses are granted case by case.

Copies of the code must keep the `Required Notice:` lines at the top of `LICENSE`.
The datasets are the property of their respective providers and are governed by their
own terms (see `docs/DATASETS.md`).
