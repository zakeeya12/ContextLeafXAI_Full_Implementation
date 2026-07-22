ContextLeafXAI
Context-Aware Multimodal Plant Disease Diagnosis with Explainable Inference
ContextLeafXAI is a PyTorch-based research framework for context-aware plant disease diagnosis using leaf images and agro-ecological metadata. The framework implements the proposed LeafContextReasonNet, which jointly learns visual disease symptoms and contextual factors such as weather, soil properties, crop identity, season, geographic information, metadata quality, and missingness.
The repository supports visual-only, context-only, and multimodal experiments, including reliability-aware gated fusion, bidirectional visual–context interaction, context-consistency regularization, Grad-CAM visualization, and contextual feature attribution.
________________________________________
Key Features
•	Hybrid EfficientNet-B3 and ViT-B/16 visual encoder 
•	Structured agro-ecological context encoder 
•	Reliability-aware multimodal fusion 
•	Bidirectional visual–context feature interaction 
•	Explicit handling of incomplete contextual metadata 
•	Context-consistency regularization 
•	Cross-modal explanation-agreement loss 
•	Grad-CAM visual explanations 
•	Integrated Gradients and SHAP-compatible context explanations 
•	Visual-only, context-only, early-fusion, gated-fusion, and full-model configurations 
•	Reproducible training with multiple random seeds 
•	Cross-dataset and robustness evaluation support 
•	NOAA weather-data retrieval utilities 
•	SoilGrids soil-feature extraction utilities 
•	Synthetic multimodal dataset for immediate pipeline testing 
•	Automated metric, prediction, checkpoint, and figure generation 
________________________________________
Proposed Architecture
LeafContextReasonNet contains five primary components:
1.	Hybrid Visual Encoder
An ImageNet-pretrained EfficientNet-B3 extracts local lesion, texture, color, and boundary information, while ViT-B/16 captures global dependencies and long-range symptom patterns.
2.	Context Encoder
A multilayer perceptron processes standardized weather and soil variables, categorical embeddings, geographic features, metadata-quality variables, and missingness indicators.
3.	Bidirectional Interaction
Contextual information recalibrates visual features, while visual evidence adjusts the importance assigned to contextual variables.
4.	Reliability-Aware Gated Fusion
The fusion module learns the relative contribution of image and contextual evidence according to their completeness, quality, and relevance.
5.	Explainable Inference
Grad-CAM identifies influential image regions, while Integrated Gradients or SHAP estimates the contribution of agro-ecological variables.
The model predicts the disease class, confidence score, modality gate values, modality reliability scores, visual heatmap, and contextual feature attributions.
________________________________________
Repository Structure
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
│   ├── base.yaml
│   ├── synthetic.yaml
│   ├── visual_only.yaml
│   ├── context_only.yaml
│   ├── early_fusion.yaml
│   ├── gated_fusion.yaml
│   └── leafcontextreasonnet.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── metadata/
│   └── splits/
│
├── src/
│   ├── datasets/
│   ├── context/
│   ├── models/
│   ├── losses/
│   ├── engine/
│   ├── explainability/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
│   ├── make_synthetic_demo.py
│   ├── prepare_plantvillage.py
│   ├── prepare_plantdoc.py
│   ├── prepare_cassava.py
│   ├── verify_context_metadata.py
│   ├── download_noaa.py
│   ├── extract_soilgrids.py
│   ├── build_context_table.py
│   └── create_splits.py
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_context_alignment.ipynb
│   ├── 03_visual_baselines.ipynb
│   ├── 04_multimodal_training.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_results_generation.ipynb
│
└── outputs/
    ├── checkpoints/
    ├── predictions/
    ├── explanations/
    ├── figures/
    ├── tables/
    └── logs/
________________________________________
Supported Datasets
Visual datasets
The repository is designed to support:
•	PlantVillage 
•	PlantDoc 
•	Cassava Leaf Disease 
These datasets can be used for visual-branch training, visual baseline evaluation, transfer learning, and cross-dataset generalization.
Context-enabled multimodal dataset
Primary multimodal experiments require plant images with verifiable sample-level metadata, including:
•	acquisition date 
•	latitude 
•	longitude 
•	crop identity 
•	disease label 
•	location or field identifier 
•	metadata verification status 
Public image datasets should not be assigned artificial weather or soil values when their acquisition time and location are unknown.
A recommended multimodal manifest is:
sample_id,image_path,disease_label,class_id,crop_name,capture_date,latitude,longitude,location_id,field_id,source_dataset,metadata_verified
________________________________________
Contextual Features
The framework can process the following types of contextual variables.
Weather features
•	7-day mean temperature 
•	7-day minimum and maximum temperature 
•	7-day temperature variability 
•	7-day accumulated precipitation 
•	14-day mean temperature 
•	14-day minimum and maximum temperature 
•	14-day accumulated precipitation 
•	humidity 
•	wind speed 
•	weather-station distance 
•	weather-record completeness 
Soil features
•	soil pH 
•	soil organic carbon 
•	clay fraction 
•	sand fraction 
•	silt fraction 
•	bulk density 
•	cation exchange capacity 
•	total nitrogen 
•	soil-layer depth 
•	soil-data quality 
Auxiliary features
•	crop identity 
•	region 
•	season 
•	month 
•	latitude zone 
•	longitude zone 
•	contextual missingness indicators 
•	metadata-quality indicators 
________________________________________
Installation
1. Clone the repository
git clone https://github.com/your-username/ContextLeafXAI.git
cd ContextLeafXAI
2. Create a virtual environment
Using Conda:
conda create -n contextleafxai python=3.10 -y
conda activate contextleafxai
Using venv:
python -m venv .venv
Linux or macOS:
source .venv/bin/activate
Windows:
.venv\Scripts\activate
3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
________________________________________
Recommended Environment
The implementation was designed for the following environment:
Python: 3.10
PyTorch: 2.x
Torchvision: 0.16 or newer
CUDA: 11.8 or newer
GPU: NVIDIA GPU with at least 12 GB memory
Recommended GPU: RTX 3090 or equivalent
CPU execution is supported for testing but is not recommended for full EfficientNet-B3 and ViT-B/16 training.
________________________________________
Quick Start with Synthetic Data
The repository includes a synthetic multimodal dataset generator for testing the complete pipeline without downloading external datasets.
1. Generate the synthetic dataset
python scripts/make_synthetic_demo.py
The script creates:
•	synthetic plant images 
•	disease labels 
•	numerical contextual features 
•	categorical contextual features 
•	missingness indicators 
•	training, validation, and test manifests 
2. Train the model
python train.py --config configs/synthetic.yaml
3. Evaluate the trained model
python evaluate.py \
  --config configs/synthetic.yaml \
  --checkpoint outputs/synthetic/checkpoints/best.pt
4. Run inference
python infer.py \
  --config configs/synthetic.yaml \
  --checkpoint outputs/synthetic/checkpoints/best.pt \
  --image path/to/image.jpg \
  --context path/to/context.json
The synthetic experiment validates software functionality only and should not be interpreted as a scientific plant-disease result.
________________________________________
Preparing a Real Dataset
Create a CSV manifest with one row per sample.
Example:
sample_id,image_path,class_id,class_name,crop_name,capture_date,latitude,longitude,temperature_mean_7d,precipitation_sum_7d,soil_ph,soil_organic_carbon,weather_missing,soil_missing,metadata_verified,split
0001,data/raw/context_enabled/img_0001.jpg,0,Tomato___Bacterial_spot,Tomato,2025-06-14,17.3850,78.4867,28.3,34.2,6.7,12.8,0,0,1,train
0002,data/raw/context_enabled/img_0002.jpg,1,Tomato___Healthy,Tomato,2025-06-16,17.3920,78.4750,29.1,2.0,6.5,11.9,0,0,1,val
Required fields depend on the selected configuration. At minimum:
image_path
class_id
split
For multimodal training:
context feature columns
metadata-quality columns
missingness columns
________________________________________
Dataset Splitting
The recommended default split is:
Training: 70%
Validation: 15%
Testing: 15%
For image-only datasets, use stratified splitting while preserving class proportions.
For context-enabled datasets, use location-grouped splitting so that samples from the same field or geographic site do not occur in both training and test partitions.
Duplicate and near-duplicate images should be removed before splitting.
Create splits using:
python scripts/create_splits.py \
  --manifest data/metadata/dataset_manifest.csv \
  --output-dir data/splits \
  --group-column location_id \
  --seed 42
________________________________________
NOAA Weather Retrieval
Weather variables can be collected for samples with verified acquisition dates and geographic coordinates.
python scripts/download_noaa.py \
  --manifest data/metadata/context_enabled_manifest.csv \
  --output data/interim/noaa_weather.csv
The retrieval module is designed to:
•	identify a valid weather station 
•	match records by acquisition date 
•	calculate 7-day weather aggregates 
•	calculate 14-day weather aggregates 
•	measure contextual completeness 
•	record station distance and retrieval status 
API keys or additional NOAA credentials may be required depending on the selected NOAA service.
________________________________________
SoilGrids Feature Extraction
Soil properties can be extracted using verified latitude and longitude coordinates.
python scripts/extract_soilgrids.py \
  --manifest data/metadata/context_enabled_manifest.csv \
  --output data/interim/soilgrids_features.csv
The utility extracts configured SoilGrids variables and records missing or unavailable values.
________________________________________
Build the Multimodal Context Table
After weather and soil retrieval:
python scripts/build_context_table.py \
  --image-manifest data/metadata/context_enabled_manifest.csv \
  --weather data/interim/noaa_weather.csv \
  --soil data/interim/soilgrids_features.csv \
  --output data/processed/multimodal_manifest.csv
All imputers, scalers, and categorical encoders must be fitted only on the training partition.
________________________________________
Training Configurations
Visual-only model
python train.py --config configs/visual_only.yaml
Context-only model
python train.py --config configs/context_only.yaml
Early-fusion baseline
python train.py --config configs/early_fusion.yaml
Conventional gated-fusion baseline
python train.py --config configs/gated_fusion.yaml
Full LeafContextReasonNet
python train.py --config configs/leafcontextreasonnet.yaml
________________________________________
Example Configuration
experiment:
  name: leafcontextreasonnet_main
  seed: 42
  output_dir: outputs/leafcontextreasonnet_main

data:
  train_manifest: data/processed/train.csv
  val_manifest: data/processed/val.csv
  test_manifest: data/processed/test.csv
  image_column: image_path
  target_column: class_id
  image_size: 224
  num_classes: 10

  numerical_context_columns:
    - temperature_mean_7d
    - temperature_max_7d
    - precipitation_sum_7d
    - temperature_mean_14d
    - precipitation_sum_14d
    - soil_ph
    - soil_organic_carbon
    - soil_clay_fraction
    - soil_sand_fraction

  categorical_context_columns:
    - crop_name
    - season
    - region

  missingness_columns:
    - weather_missing
    - soil_missing

  quality_columns:
    - weather_completeness_ratio
    - weather_station_distance_km
    - soil_data_quality
    - metadata_verified

model:
  mode: multimodal
  cnn_backbone: efficientnet_b3
  transformer_backbone: vit_b_16
  pretrained: true
  visual_projection_dim: 512
  context_embedding_dim: 128
  fusion_dim: 256
  dropout: 0.30
  bidirectional_interaction: true
  reliability_aware_fusion: true

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  backbone_learning_rate: 0.00001
  weight_decay: 0.0001
  warmup_epochs: 5
  early_stopping_patience: 15
  gradient_clip_norm: 5.0
  mixed_precision: true

loss:
  classification_weight: 1.0
  context_consistency_weight: 0.20
  explanation_agreement_weight: 0.10

evaluation:
  save_predictions: true
  save_confusion_matrix: true
  save_roc_curves: true
  save_calibration_metrics: true
________________________________________
Training Procedure
The default training procedure consists of:
1.	loading image and contextual inputs; 
2.	applying training-only image augmentation; 
3.	encoding the image using EfficientNet-B3 and ViT-B/16; 
4.	encoding agro-ecological metadata using the context MLP; 
5.	estimating visual and context reliability; 
6.	applying bidirectional cross-modal recalibration; 
7.	generating the reliability-aware gated representation; 
8.	predicting disease probabilities; 
9.	calculating the classification and regularization losses; 
10.	updating the model using AdamW; 
11.	evaluating validation performance; 
12.	saving the best checkpoint. 
The recommended optimization setup is:
Optimizer: AdamW
Initial learning rate: 1e-4
Backbone learning rate: 1e-5
Weight decay: 1e-4
Batch size: 32
Maximum epochs: 100
Warm-up: 5 epochs
Scheduler: cosine annealing
Early-stopping patience: 15 epochs
Gradient clipping: 5.0
________________________________________
Loss Function
The total loss is:
Total Loss =
Classification Loss
+ λcontext × Context-Consistency Loss
+ λagreement × Explanation-Agreement Loss
Default weights:
λcontext = 0.20
λagreement = 0.10
Classification loss
Weighted cross-entropy is used for disease classification.
Context-consistency loss
The model is encouraged to maintain stable predictions when contextual variables are mildly perturbed, masked, or made partially unavailable.
Explanation-agreement loss
A differentiable proxy encourages visual and contextual evidence to produce compatible disease representations.
________________________________________
Evaluation
Evaluate a trained model using:
python evaluate.py \
  --config configs/leafcontextreasonnet.yaml \
  --checkpoint outputs/leafcontextreasonnet_main/checkpoints/best.pt
The evaluator can report:
•	accuracy 
•	macro precision 
•	macro recall 
•	macro F1-score 
•	weighted F1-score 
•	micro F1-score 
•	multiclass ROC-AUC 
•	negative log-likelihood 
•	Brier score 
•	expected calibration error 
•	confusion matrix 
•	per-class metrics 
•	parameter count 
•	inference latency 
•	throughput 
•	peak GPU-memory usage 
________________________________________
Explainability
Grad-CAM
Generate visual explanations:
python explain.py \
  --config configs/leafcontextreasonnet.yaml \
  --checkpoint outputs/leafcontextreasonnet_main/checkpoints/best.pt \
  --method gradcam \
  --manifest data/processed/test.csv \
  --output-dir outputs/explanations/gradcam
Grad-CAM overlays indicate image regions that influenced the disease prediction.
Integrated Gradients
Generate contextual feature attributions:
python explain.py \
  --config configs/leafcontextreasonnet.yaml \
  --checkpoint outputs/leafcontextreasonnet_main/checkpoints/best.pt \
  --method integrated_gradients \
  --manifest data/processed/test.csv \
  --output-dir outputs/explanations/integrated_gradients
SHAP
Where supported by the selected model and environment:
python explain.py \
  --config configs/leafcontextreasonnet.yaml \
  --checkpoint outputs/leafcontextreasonnet_main/checkpoints/best.pt \
  --method shap \
  --manifest data/processed/test.csv \
  --output-dir outputs/explanations/shap
For computationally expensive SHAP analysis, use a stratified subset of the test set.
________________________________________
Inference
Prepare contextual values as JSON:
{
  "temperature_mean_7d": 28.6,
  "temperature_max_7d": 34.2,
  "precipitation_sum_7d": 42.8,
  "temperature_mean_14d": 27.9,
  "precipitation_sum_14d": 68.4,
  "soil_ph": 6.5,
  "soil_organic_carbon": 13.2,
  "soil_clay_fraction": 31.5,
  "soil_sand_fraction": 42.1,
  "crop_name": "Tomato",
  "season": "Monsoon",
  "region": "South",
  "weather_missing": 0,
  "soil_missing": 0,
  "weather_completeness_ratio": 0.93,
  "weather_station_distance_km": 8.4,
  "soil_data_quality": 0.91,
  "metadata_verified": 1
}
Run:
python infer.py \
  --config configs/leafcontextreasonnet.yaml \
  --checkpoint outputs/leafcontextreasonnet_main/checkpoints/best.pt \
  --image examples/tomato_leaf.jpg \
  --context examples/tomato_context.json
The inference output may include:
Predicted class
Confidence score
Top-k disease probabilities
Visual reliability
Context reliability
Average fusion-gate value
Grad-CAM output path
Context attribution output path
________________________________________
Baseline Experiments
The repository supports comparison with:
•	ResNet-50 
•	DenseNet-121 
•	EfficientNet-B3 
•	ViT-B/16 
•	context-only MLP 
•	early concatenation 
•	late probability fusion 
•	conventional sigmoid-gated fusion 
•	reliability-aware fusion 
•	full LeafContextReasonNet 
All models should use the same:
•	dataset partitions 
•	preprocessing 
•	image resolution 
•	augmentation policy 
•	optimizer 
•	stopping criteria 
•	metric implementation 
•	random seeds 
________________________________________
Ablation Studies
Recommended ablation configurations include:
ID	Configuration
A1	Visual-only model
A2	Context-only model
A3	Early concatenation
A4	Conventional gated fusion
A5	Reliability-aware fusion
A6	Fusion with bidirectional interaction
A7	A6 with context-consistency loss
A8	A7 with explanation-agreement loss
A9	Full model without missingness indicators
A10	Full model without metadata-quality variables
A11	Full model without EfficientNet-B3
A12	Full model without ViT-B/16
A13	Complete LeafContextReasonNet
Additional contextual ablations can compare:
weather only
soil only
weather and soil
weather, soil, and crop
complete context
________________________________________
Robustness Experiments
The implementation can be extended or configured to evaluate:
•	missing weather features 
•	missing soil features 
•	random context masking 
•	Gaussian perturbation of normalized context 
•	metadata incompleteness 
•	image blur 
•	illumination variation 
•	background complexity 
•	image compression 
•	contextual noise 
Recommended context masking levels:
0%, 10%, 20%, 30%, 40%, and 50%
________________________________________
Cross-Dataset Evaluation
Cross-dataset experiments should only compare harmonized crop–disease classes.
Recommended protocols:
PlantVillage → PlantDoc
PlantDoc → PlantVillage
Cassava has a crop-specific label space and is better suited for:
•	feature-transfer evaluation 
•	crop-specific fine-tuning 
•	out-of-domain representation analysis 
Avoid directly comparing incompatible label spaces.
________________________________________
Reproducibility
Recommended seeds:
42
123
2026
Example:
python train.py \
  --config configs/leafcontextreasonnet.yaml \
  --seed 42
Repeat with all three seeds and report mean and standard deviation.
The repository preserves:
•	configuration files 
•	dataset split files 
•	model checkpoints 
•	prediction files 
•	metric summaries 
•	feature preprocessors 
•	random seed information 
•	environment details 
________________________________________
Output Files
A typical experiment produces:
outputs/experiment_name/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── logs/
│   ├── training.csv
│   └── experiment.log
├── predictions/
│   ├── validation_predictions.csv
│   └── test_predictions.csv
├── metrics/
│   ├── validation_metrics.json
│   └── test_metrics.json
├── figures/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── calibration_curve.png
└── explanations/
    ├── gradcam/
    ├── integrated_gradients/
    └── shap/
________________________________________
Exporting the Model
Export a trained checkpoint using:
python export_model.py \
  --config configs/leafcontextreasonnet.yaml \
  --checkpoint outputs/leafcontextreasonnet_main/checkpoints/best.pt \
  --format torchscript \
  --output outputs/exported/leafcontextreasonnet.ts
Depending on model compatibility, supported export formats may include:
•	TorchScript 
•	ONNX 
•	PyTorch state dictionary 
________________________________________
Google Colab
The repository can be used in Google Colab.
!git clone https://github.com/your-username/ContextLeafXAI.git
%cd ContextLeafXAI
!pip install -r requirements.txt
Generate the synthetic demonstration:
!python scripts/make_synthetic_demo.py
Train:
!python train.py --config configs/synthetic.yaml
Evaluate:
!python evaluate.py \
    --config configs/synthetic.yaml \
    --checkpoint outputs/synthetic/checkpoints/best.pt
For Google Drive storage:
from google.colab import drive
drive.mount("/content/drive")
Update the data and output paths in the YAML configuration accordingly.
________________________________________
Scientific Usage Notes
•	Do not assign NOAA or SoilGrids values to images without verifiable dates and coordinates. 
•	Fit normalization and imputation parameters using the training partition only. 
•	Keep samples from the same location, plant, field, or sequence within one data partition. 
•	Remove duplicate and near-duplicate images before splitting. 
•	Use lesion annotations when reporting Grad-CAM localization IoU. 
•	Do not treat synthetic-data results as scientific findings. 
•	Report all principal results using repeated runs and fixed dataset partitions. 
•	Clearly separate visual-only public-dataset experiments from verified multimodal experiments. 
________________________________________
Citation
When using this repository in academic work, cite the associated manuscript:
@article{contextleafxai,
  title   = {ContextLeafXAI LeafContextReasonNet for Context-Aware Multimodal Plant Disease Diagnosis Integrating Agro-Ecological Metadata and Explainable Inference},
  author  = {Author names},
  journal = {Journal name},
  year    = {2026},
  volume  = {To be added},
  number  = {To be added},
  pages   = {To be added},
  doi     = {To be added}
}
Update the bibliographic information after publication.
________________________________________
License
Specify the intended license before making the repository public.
A commonly used option for research code is the MIT License:
MIT License
Dataset licenses remain governed by their respective dataset providers. This repository does not redistribute third-party datasets unless explicitly permitted.
________________________________________
Disclaimer
This software is intended for research and experimental decision-support purposes. It is not a substitute for diagnosis by agricultural experts, plant pathologists, or local extension services. Predictions should be interpreted together with field observations, laboratory testing, regional disease reports, and professional agronomic advice.
________________________________________
Acknowledgements
The framework is designed to support experiments using public plant-disease image datasets and agro-ecological data sources such as PlantVillage, PlantDoc, Cassava Leaf Disease, NOAA climate observations, and SoilGrids soil information. Users should acknowledge and cite the original dataset and data-source publications according to their respective terms.

