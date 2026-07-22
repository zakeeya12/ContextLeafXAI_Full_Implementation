#  ContextLeafXAI
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21484943-blue)](https://doi.org/10.5281/zenodo.21484943)

> **Context-Aware Multimodal Plant Disease Diagnosis with Explainable Inference**

ContextLeafXAI is a **PyTorch-based research framework** for context-aware plant disease diagnosis using **leaf images** and **agro-ecological metadata**.

The framework introduces **LeafContextReasonNet**, a multimodal architecture that jointly learns visual disease symptoms together with contextual information such as:

- Weather
- Soil properties
- Crop identity
- Season
- Geographic location
- Metadata quality
- Missing contextual information

It supports visual-only, context-only, and multimodal learning together with explainable AI methods.

---

#  Features

- Hybrid EfficientNet-B3 + ViT-B/16 visual encoder
- Structured agro-ecological context encoder
- Reliability-aware multimodal fusion
- Bidirectional visual-context interaction
- Missing-context handling
- Context-consistency regularization
- Cross-modal explanation agreement loss
- Grad-CAM visualization
- Integrated Gradients
- SHAP-compatible explanations
- Visual-only experiments
- Context-only experiments
- Early Fusion
- Gated Fusion
- Full multimodal framework
- Cross-dataset evaluation
- Robustness evaluation
- Synthetic multimodal dataset generator
- NOAA weather utilities
- SoilGrids feature extraction
- Automatic checkpoint, metrics, and figure generation

---

# Table of Contents

- Overview
- Architecture
- Repository Structure
- Supported Datasets
- Context Features
- Installation
- Quick Start
- Preparing Real Data
- Training
- Evaluation
- Explainability
- Inference
- Baselines
- Ablation Studies
- Robustness
- Cross Dataset Evaluation
- Reproducibility
- Output Files
- Model Export
- Google Colab
- Scientific Notes
- Citation
- License

---

#  Proposed Architecture

LeafContextReasonNet contains five major components.

## 1. Hybrid Visual Encoder

- EfficientNet-B3
- Vision Transformer (ViT-B/16)

Extracts

- lesion
- texture
- color
- boundary
- global symptom relationships

---

## 2. Context Encoder

Processes

- Weather
- Soil
- Crop metadata
- Geographic information
- Missingness indicators
- Metadata quality

using a multilayer perceptron.

---

## 3. Bidirectional Interaction

Visual features influence contextual reasoning while contextual information recalibrates visual representations.

---

## 4. Reliability-Aware Fusion

Learns how much the prediction should rely on

- image information
- contextual information

according to data quality and completeness.

---

## 5. Explainable Inference

Produces

- Grad-CAM heatmaps
- Integrated Gradients
- SHAP explanations

for transparent prediction.

---

#  Repository Structure

```text
ContextLeafXAI/
│
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
├── infer.py
├── explain.py
├── export_model.py
│
├── configs/
├── data/
├── src/
├── scripts/
├── notebooks/
└── outputs/
```

---

#  Supported Datasets

## Visual datasets

- PlantVillage
- PlantDoc
- Cassava Leaf Disease

## Context-enabled Dataset

Each sample should contain

```text
sample_id
image_path
disease_label
crop_name
capture_date
latitude
longitude
location_id
field_id
metadata_verified
```

---

#  Context Features

## Weather

- Temperature
- Rainfall
- Humidity
- Wind Speed
- Weather completeness

## Soil

- pH
- Organic Carbon
- Clay
- Sand
- Nitrogen
- Bulk Density

## Auxiliary

- Crop
- Region
- Season
- Latitude
- Longitude
- Metadata quality
- Missingness indicators

---

# Installation

Clone repository

```bash
git clone [https://github.com/your-username/ContextLeafXAI.git](https://github.com/zakeeya12/ContextLeafXAI_Full_Implementation)
cd ContextLeafXAI
```

Create environment

```bash
conda create -n contextleafxai python=3.10 -y

conda activate contextleafxai
```

Install packages

```bash
pip install -r requirements.txt
```

---

#  Recommended Environment

| Component | Version |
|------------|----------|
| Python | 3.10 |
| PyTorch | 2.x |
| Torchvision | 0.16+ |
| CUDA | 11.8+ |
| GPU | RTX3090 (Recommended) |

---

#  Quick Start

Generate synthetic dataset

```bash
python scripts/make_synthetic_demo.py
```

Train

```bash
python train.py --config configs/synthetic.yaml
```

Evaluate

```bash
python evaluate.py \
--config configs/synthetic.yaml \
--checkpoint outputs/synthetic/checkpoints/best.pt
```

Inference

```bash
python infer.py \
--config configs/synthetic.yaml \
--checkpoint outputs/synthetic/checkpoints/best.pt \
--image path/to/image.jpg \
--context path/to/context.json
```

---

#  Training Configurations

Available experiments

- Visual-only
- Context-only
- Early Fusion
- Gated Fusion
- LeafContextReasonNet

Example

```bash
python train.py --config configs/leafcontextreasonnet.yaml
```

---

#  Evaluation Metrics

The framework reports

- Accuracy
- Precision
- Recall
- Macro F1
- Weighted F1
- ROC-AUC
- Brier Score
- Expected Calibration Error
- Confusion Matrix
- Throughput
- GPU Memory
- Latency

---

#  Explainability

## Grad-CAM

```bash
python explain.py --method gradcam
```

## Integrated Gradients

```bash
python explain.py --method integrated_gradients
```

## SHAP

```bash
python explain.py --method shap
```

---

#  Ablation Studies

| ID | Configuration |
|----|---------------|
| A1 | Visual Only |
| A2 | Context Only |
| A3 | Early Fusion |
| A4 | Gated Fusion |
| A5 | Reliability Fusion |
| A6 | Bidirectional Fusion |
| A7 | + Context Consistency |
| A8 | + Explanation Agreement |
| A9 | Without Missingness |
| A10 | Without Metadata Quality |
| A11 | Without EfficientNet |
| A12 | Without ViT |
| A13 | Full Model |

---

#  Output Structure

```text
outputs/

├── checkpoints/
├── predictions/
├── explanations/
├── figures/
├── metrics/
└── logs/
```

---

#  Google Colab

```python
!git clone https://github.com/your-username/ContextLeafXAI.git

%cd ContextLeafXAI

!pip install -r requirements.txt
```

Generate synthetic data

```python
!python scripts/make_synthetic_demo.py
```

Train

```python
!python train.py --config configs/synthetic.yaml
```

---

#  Citation

```bibtex
@article{contextleafxai,
title={ContextLeafXAI: Context-Aware Multimodal Plant Disease Diagnosis Integrating Agro-Ecological Metadata and Explainable Inference},
'''
---
## License

```text
MIT License
```
---


