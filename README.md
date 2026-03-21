# DELBERT

**Fingerprint Language Modeling for Generalizable Hit Discovery in DNA-Encoded Libraries**

[Paper](https://openreview.net/forum?id=wOOD4W3wJk) | [Data](https://huggingface.co/datasets/wanglab/delbert-data) | [Models](https://huggingface.co/wanglab/delbert)

*Published at the [ICLR 2026 Workshop on Machine Learning for Genomics Explorations](https://mlgenx.github.io/)*

## Overview

DELBERT is a self-supervised transformer encoder that treats molecular fingerprints (FPs) as a discrete token language, enabling masked language modeling (MLM) pretraining without requiring access to underlying chemical structures. This makes it uniquely suited for privacy-preserving settings such as the [AIRCHECK](https://www.aircheck.science/) initiative, where DEL screening data is released only as precomputed fingerprints.

Under rigorous library-based out-of-distribution (OOD) evaluation across four protein targets (WDR91, LRRK2, SETDB1, DCAF7), DELBERT significantly outperforms XGBoost and LightGBM ensemble baselines on three of four targets, with **1.6--2.7x improvements** in early-enrichment metrics.

## Installation

```bash
git clone https://github.com/bowang-lab/DELBERT.git
cd DELBERT
pip install -e .
```

Optional dependencies:

```bash
# For LoRA finetuning
pip install peft

# For baseline models (XGBoost, LightGBM)
pip install -e ".[baselines]"

# Everything
pip install -e ".[all]"
```

## Data

Datasets are hosted on HuggingFace: [wanglab/delbert-data](https://huggingface.co/datasets/wanglab/delbert-data)

Each dataset contains molecules represented as sparse molecular fingerprints (ECFP4, FCFP6, ATOMPAIR, TOPTOR) with binary enrichment labels from DEL screens against four protein targets: WDR91, LRRK2, SETDB1, and DCAF7.

## Pretrained Models

Pretrained checkpoints are available at: [wanglab/delbert](https://huggingface.co/wanglab/delbert)

## Quick Start

### 1. Prepare pretraining data

Tokenize molecular fingerprints into token sequences and build the vocabulary:

```bash
python scripts/prepare_pretrain_data.py --config-name pretrain/example
```

This creates `processed_data/pretrain/{experiment_id}/` with tokenized data and vocabulary.

### 2. Self-supervised pretraining (MLM)

Pretrain DELBERT via masked language modeling on all molecules (labels discarded):

```bash
python scripts/pretrain.py experiment=pretrain_example
```

### 3. Prepare supervised data

Prepare labeled data with library-based OOD splits:

```bash
python scripts/prepare_supervised_data.py --config-name supervised/example
```

### 4. Supervised finetuning

Finetune the pretrained model for binding prediction with LoRA:

```bash
python scripts/train_classifier.py experiment=classify_example \
    pretrained_checkpoint=outputs/pretrain_wdr91/.../checkpoints/best/epoch=XX-val_loss=X.XXX.ckpt
```

## Cross-Validation Evaluation

For rigorous OOD evaluation using library-based K-fold cross-validation:

**DELBERT transformer:**
```bash
python evals/library_cv/scripts/run_transformer_cv_orchestrator.py \
    --config evals/library_cv/configs/transformer_cv.yaml
```

**Baseline models (XGBoost, LightGBM):**
```bash
python evals/library_cv/scripts/run_baseline_cv.py \
    --config evals/library_cv/configs/baseline_cv.yaml
```

Both scripts use the same fold assignments for fair comparison. Results are saved to `evals/library_cv/results/`.

## Project Structure

```
DELBERT/
├── delbert/                  # Core package
│   ├── data/                 # Data loading, tokenization, splits
│   └── models/               # Model architecture, training modules
├── scripts/                  # Data prep and training entry points
├── evals/                    # Evaluation pipelines
│   └── library_cv/           # Library-based K-fold cross-validation
├── configs/                  # Hydra configuration files
│   ├── data_prep/            # Data preparation configs
│   └── experiment/           # Training experiment configs
├── pyproject.toml
└── requirements.txt
```

## Citation

```bibtex
@inproceedings{seyedahmadi2026delbert,
    title={{DELBERT}: Fingerprint Language Modeling for Generalizable Hit Discovery in {DNA}-Encoded Libraries},
    author={Arman Seyed-Ahmadi and Bing Hu and Armin Geraili and Anita Layton and Helen Chen and Shana O. Kelley and Bo Wang},
    booktitle={ICLR 2026 Workshop on Machine Learning for Genomics Explorations},
    year={2026},
    url={https://openreview.net/forum?id=wOOD4W3wJk}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
