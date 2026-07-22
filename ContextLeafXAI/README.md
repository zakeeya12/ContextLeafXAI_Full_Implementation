# ContextLeafXAI

A complete PyTorch implementation of **LeafContextReasonNet** for context-aware multimodal plant disease diagnosis.

## Features

- EfficientNet-B3 + ViT-B/16 hybrid visual encoder
- MLP context encoder for weather, soil, crop, region, season, and missingness variables
- Reliability-aware gated fusion
- Bidirectional visual-context feature interaction
- Context-consistency regularization
- Differentiable cross-modal agreement regularization
- Grad-CAM, Integrated Gradients, and SHAP utilities
- Visual-only, context-only, early-fusion, gated-fusion, and full-model modes
- Train/validation/test CLI, inference CLI, efficiency profiling, and synthetic smoke test
- NOAA and SoilGrids context acquisition scripts

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Manifest format

Each CSV row represents one image. Required columns:

```text
image_path,class_id,class_name
```

For multimodal training, add numerical context columns, optional categorical columns already integer-encoded, missingness flags, and quality variables. Configure their names in `configs/base.yaml`.

Example:

```text
image_path,class_id,class_name,temp_mean_7d,rain_sum_14d,soil_ph,miss_temp,weather_completeness
/path/a.jpg,0,Tomato___healthy,24.1,12.5,6.4,0,0.96
```

## Prepare splits

```bash
python scripts/create_splits.py --manifest data/metadata/all_samples.csv --out-dir data/processed
```

## Train

```bash
python train.py --config configs/base.yaml
```

## Evaluate

```bash
python evaluate.py --config configs/base.yaml --checkpoint outputs/checkpoints/best.pt
```

## Inference

```bash
python infer.py --config configs/base.yaml --checkpoint outputs/checkpoints/best.pt --image sample.jpg --context-json sample_context.json
```

## Explainability

```bash
python explain.py --config configs/base.yaml --checkpoint outputs/checkpoints/best.pt --image sample.jpg --context-json sample_context.json
```

## Synthetic end-to-end smoke test

```bash
python scripts/make_synthetic_demo.py
python train.py --config configs/synthetic.yaml
python evaluate.py --config configs/synthetic.yaml --checkpoint outputs/synthetic/checkpoints/best.pt
```

## Important scientific constraint

Do not assign NOAA or SoilGrids values to public images unless capture date and geographic coordinates are verified. Public datasets without such metadata should be used for visual-only experiments.
