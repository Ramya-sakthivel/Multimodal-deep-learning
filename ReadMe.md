# Title: Multimodal Deep Learning for Automated Pressure Ulcer Risk Assessment

## Project Overview & Description

Pressure ulcers (bedsores) represent a significant healthcare challenge in hospital environments and long-term nursing care. Early risk assessment is essential for targeted clinical decision-making and prevention.

This repository contains the official implementation of a novel **multimodal deep learning architecture** that integrates structured clinical records and unstructured wound image data using a **Transformer-based Cross-Modal Multi-Head Attention Fusion** mechanism.

Unlike conventional unimodal approaches or naive feature concatenation, this framework models complex interactions across modalities to predict three clinically actionable pressure ulcer risk categories:

* **Label 0**: Low Risk  
* **Label 1**: Moderate Risk  
* **Label 2**: High Risk

The system integrates an **Explainable AI (XAI)** suite incorporating **Grad-CAM** for image focal region visualization, **Cross-Modal Attention Maps** for feature interaction analysis, and **Permutation Feature Importance** for clinical variables.

## Dataset Information

This research utilizes two publicly available, third-party datasets that are aligned at the label level into unified risk categories.

### 1\. MIMIC3d Aggregated Clinical Dataset (Tabular)

* **Description**: Derived from the Medical Information Mart for Intensive Care III (MIMIC-III) database. It contains structured patient-level hospital admission records, including demographics, length-of-stay groups, chart event counts, and care interactions.  
* **Size**: 58,976 total patient instances across 29 features.  
* **Class Distribution**: Low Risk (15,280), Moderate Risk (24,370), High Risk (19,326).  
* **Source & Download Link**: [Kaggle \- MIMIC3d Dataset](https://www.kaggle.com/datasets/drscarlat/mimic3d?resource=download)

### 2\. Pressure Injury Image Dataset (PIID / Mini UPP)

* **Description**: Clinical wound images of pressure ulcers across varying stages, lighting conditions, and tissue severities.  
* **Size**: 1,391 total images after harmonization (High Risk: 848, Moderate Risk: 313, Low Risk: 230).  
* **Source & Download Link**: [Kaggle \- Mini UPP Dataset](https://www.kaggle.com/datasets/marcoivanesposito/mini-upp-dataset)

### Dataset Harmonization & Risk Mapping Scheme

To harmonize disparate image labels across PIID sources into unified 3-class risk categories:

| Original PIID Stage / Category | Harmonized Risk Category | Target Numerical Label |
| :---- | :---- | :---: |
| Stage I / Stage 1 | Low Risk | **0** |
| Stage II / Stage 2 | Moderate Risk | **1** |
| Stage III / Stage IV / Stage 3 / Stage 4 | High Risk | **2** |
| Suspected Deep Tissue Injury (SDTI) | High Risk | **2** |
| Unstageable | High Risk | **2** |
| Invalid / Unclear | Excluded | — |

## Code Information & Repository Structure

### Directory Layout

```
.
├── data/
│   ├── raw/                       # Directory for downloaded MIMIC3d CSV & PIID image folders
│   └── processed/                 # Generated aligned multimodal dataset files
├── Pressure-ulcer-initial.ipynb   # Notebook for dataset importing, cleaning & label mapping
├── Pressure-Ulcer-MultiModal-Version1.ipynb # Complete end-to-end Python pipeline (Preprocessing,
│                                  # Feature Extraction, Transformer Fusion, Training, Evaluation,
│                                  # Statistical Tests, and XAI Explanations)
├── requirements.txt               # Required Python dependencies
└── README.md                      # Complete project documentation
```

### Core Architectural Modules

1. **Clinical Feature Extractor (DNN)**: 2-layer Deep Neural Network (128 and 64 units) with ReLU activations, Batch Normalization, and Dropout to process 29 structured clinical parameters into a clinical feature vector${z}_{clin}\in {R}^{128}$.&nbsp;  
2. **Image Feature Extractor (CNN)**: Pretrained **EfficientNet-B0** backbone taking $(224\times 224\times 3)$ input wound images, followed by Global Average Pooling (GAP) to extract visual feature vector  ${z}_{img}\in {R}^{128}$.&nbsp;  
3. **Transformer Cross-Modal Fusion**: Multi-Head Self-Attention mechanism (4 heads, embedding dimension 128\) that dynamically weights and cross-attends clinical features against visual spatial features.  
4. **Classification Head**: Dense feed-forward layers with Softmax output predicting 3-class probabilities (Low, Moderate, High risk).  
5. **XAI Interpretability Suite**:  
   * **Grad-CAM**: Generates class activation heatmaps overlaying wound images.  
   * **Attention Visualization**: Plots $N\times N$ cross-modal interaction weight matrices.  
   * **Permutation Feature Importance**: Quantifies drop in loss when clinical variables are permuted.

## Methodology Overview

1. **Preprocessing**: Missing values imputed via central tendency (numerical) and mode (categorical). Min-Max scaling applied to clinical features $[0,1]$. Images resized to $224\times 224$ and normalized ($\frac{X}{255}$). Data augmentation (rotation, flipping, zooming) applied to mitigate image class imbalance.  
2. **Multimodal Alignment**: Label-level synchronization joining clinical admission vectors and wound images sharing identical target risk categories.  
3. **Two-Stage Training Strategy**:  
   * *Stage 1*: Freeze EfficientNet-B0 backbone weights; train classification and transformer fusion layers at learning rate $\eta =1\times 1{0}^{-4}$.  
   * *Stage 2*: Unfreeze upper convolutional blocks for end-to-end fine-tuning at reduced learning rate $\eta =1\times 1{0}^{-5}$.  
4. **Optimization**: Categorical Cross-Entropy Loss weighted by class frequencies ${w}_{C}=\frac{N}{{n}_{c}}$ optimized using Adam.

## &nbsp;Experimental Performance & Baseline Comparison

### Ablation Study Results

| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| :---- | :---: | :---: | :---: | :---: |
| Image-Only (EfficientNet-B0) | 41.91% | 38.97% | 41.91% | 31.31% |
| Clinical-Only (DNN) | 89.97% | 92.74% | 89.97% | 90.01% |
| Naive Concatenation Fusion | 82.52% | 83.13% | 82.52% | 82.14% |
| **Proposed Transformer Fusion** | **92.56%** | **92.74%** | **92.56%** | **92.52%** |

### Statistical Significance & Robustness

* **Paired t-test**: $p=0.000999<0.001$, confirming statistically significant improvement over single-modality and naive fusion baselines.  
* **Bootstrap Analysis**: 1,000 bootstrap iterations demonstrated a stable normal accuracy gain distribution centered at $+10.09\%$ with low variance.

## &nbsp;Requirements & Dependencies

The codebase requires **Python 3.8+**. The primary dependencies are:

```
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
opencv-python>=4.5.0
pillow>=8.0.0
scipy>=1.7.0
```

## Usage Instructions

### Step 1: Environment Setup

Clone the repository and install dependencies:

```sh
git clone https://github.com/Ramya-sakthivel/Multimodal-deep-learning.git
cd Multimodal-deep-learning
pip install -r requirements.txt
```

### Step 2: Data Preparation

1. Download the MIMIC3d dataset from [Kaggle](https://www.kaggle.com/datasets/drscarlat/mimic3d?resource=download) and place `mimic3d.csv` into `data/raw/`.  
2. Download the PIID / Mini UPP image dataset from [Kaggle](https://www.kaggle.com/datasets/marcoivanesposito/mini-upp-dataset) and extract image folders into `data/raw/PIID/`.  
3. Open and run `Pressure-ulcer-initial.ipynb` in Jupyter Notebook or Google Colab to generate harmonized preprocessed datasets.

### Step 3: Model Training & Evaluation

Open and execute the main pipeline notebook `Pressure-Ulcer-MultiModal-Version1.ipynb` sequentially in Jupyter Notebook, JupyterLab, or Google Colab:

```sh
jupyter notebook Pressure-Ulcer-MultiModal-Version1.ipynb
```

This executes all steps end-to-end:

* Data preprocessing & loading  
* CNN (EfficientNet-B0) \+ DNN feature extraction  
* Transformer-based cross-modal multi-head attention fusion  
* Two-stage model training & hyperparameter configuration  
* Statistical significance testing & bootstrap analysis  
* XAI generation (Grad-CAM, cross-modal attention maps, feature importance)

## License & Contribution Guidelines-

### License

This project is open-source and licensed under the [MIT License](http://LICENSE). You are free to reuse, modify, and distribute the code for academic and commercial applications with appropriate attribution.

### Contribution Guidelines

Contributions, bug reports, and enhancements are welcome\!

1. Fork the repository.  
2. Create a feature branch (`git checkout -b feature/NewFeature`).  
3. Commit your changes (`git commit -m 'Add NewFeature'`).  
4. Push to the branch (`git push origin feature/NewFeature`).  
5. Open a Pull Request detailing your modifications.

## Citation

If you use this codebase, model architecture, or experimental protocol in your research, please cite our paper:

```bibtex
@article{chinnasamy2026multimodal,
  title={Multimodal Deep Learning for Automated Pressure Ulcer Risk Assessment in Nursing Care},
  author={Chinnasamy, Ramya and Subramanian, Malliga},
  journal={PeerJ Computer Science},
  year={2026},
  note={AI Application}
}
```