# Multimodal-deep-learning
================================================================================
MULTIMODAL DEEP LEARNING FOR AUTOMATED PRESSURE ULCER RISK ASSESSMENT
================================================================================

AUTHORS
-------
1. Ramya Chinnasamy – Department of Computer science and Engineering, Kongu Engineering College, Perundurai, India
2. Malliga Subramanian – Department of Computer science and Engineering, Kongu Engineering College, Perundurai, India  
 Email: ramya.me@gmail.com
DOI / Code Availability: 


1. ABSTRACT OVERVIEW
--------------------
Pressure ulcers (bedsores) represent a significant challenge in hospital environments and nursing care. Early risk assessment is essential for targeted prevention. This framework combines structured clinical indicators (29 patient attributes) with unstructured wound images to predict 3-class pressure ulcer risk levels:
  * Label 0: Low Risk
  * Label 1: Moderate Risk
  * Label 2: High Risk

Key Highlights:
  * Multimodal Integration: Fusion of clinical records (MIMIC3d, 58,976 records) 
    and wound images (PIID, 1,391 images).
  * Transformer-Based Cross-Modal Fusion: Captures complex cross-modal 
    interactions beyond simple concatenation.
  * Explainable AI (XAI): Multilevel interpretability using Grad-CAM for image focus, 
    Cross-Modal Attention Maps for feature interaction, and Permutation Feature 
    Importance for clinical variables.
  * Statistical Rigor: Performance validated via statistical significance testing 
    (p < 0.001) and bootstrap confidence analysis.


2. SYSTEM ARCHITECTURE
----------------------

  [ Wound Image ]                            [ Clinical Data ]
(Batch, 224, 224, 3)                      (Batch, 29 Features)
         |                                         |
         v                                         v
+-----------------------+              +-----------------------+
| EfficientNet-B0 (CNN) |              |     Clinical DNN      |
|  Feature Extraction   |              | (Dense + BatchNorm +  |
| Global Avg Pooling    |              |    ReLU + Dropout)    |
+-----------+-----------+              +-----------+-----------+
            |                                      |
            v                                      v
  [ Visual Vector z_img ]                [ Clinical Vector z_clin ]
            |                                      |
            +------------------+-------------------+
                               |
                               v
                   +-----------------------+
                   | Transformer Cross-    |
                   | Modal Fusion Block    |
                   | (Multi-Head Attention)|
                   +-----------+-----------+
                               |
                               v
                   +-----------------------+
                   | Fully Connected Head  |
                   +-----------+-----------+
                               |
                               v
                   +-----------------------+
                   | Softmax Classifier    |
                   | Low / Moderate / High |
                   +-----------------------+


3. EXPERIMENTAL RESULTS & ABLATION STUDY
----------------------------------------

Comprehensive Performance Comparison:
------------------------------------------------------------------
Model Architecture          | Accuracy | Precision | Recall | F1-Score
------------------------------------------------------------------
Image Only (CNN)            |  41.91%  |   38.97%  | 41.91% |  31.31%
Clinical Only (DNN)         |  89.97%  |   92.74%  | 89.97% |  90.01%
Naive Concatenation Fusion  |  82.52%  |   83.13%  | 82.52% |  82.14%
Proposed Transformer Fusion |  92.56%  |   92.74%  | 92.56% |  92.52%
------------------------------------------------------------------

Per-Class Classification Breakdown:
------------------------------------------------------------------
Risk Category   | Class Label | Precision | Recall | F1-Score | Support
------------------------------------------------------------------
Low Risk        |     0       |   0.94    |  0.97  |   0.96   |   153
Moderate Risk   |     1       |   0.95    |  0.87  |   0.91   |   264
High Risk       |     2       |   0.89    |  0.96  |   0.92   |   201
------------------------------------------------------------------
Macro Average   |             |   0.93    |  0.94  |   0.93   |   618
Weighted Avg    |             |   0.93    |  0.93  |   0.93   |   618
------------------------------------------------------------------


4. DATASET HARMONIZATION & MAPPING
-----------------------------------

Target Category Harmonization for PIID Image Dataset:
-------------------------------------------------------------------
Original Stage Label           | Mapped Risk Category | Target Label
-------------------------------------------------------------------
Stage I / Stage 1              | Low Risk             |      0
Stage II / Stage 2             | Moderate Risk        |      1
Stage III / IV / Stage 3 / 4   | High Risk            |      2
Suspected Deep Tissue (SDTI)   | High Risk            |      2
Unstageable                    | High Risk            |      2
Invalid                        | Excluded             |      -
-------------------------------------------------------------------

MIMIC3d Dataset:
  - 58,976 total patient instances across 29 features.
  - Class Distribution: Low Risk (15,280), Moderate Risk (24,370), High Risk (19,326).


5. MODEL HYPERPARAMETERS
------------------------
  * Image Backbone: EfficientNet-B0
  * Initial Learning Rate: 1e-4
  * Fine-tuning Learning Rate: 1e-5
  * Batch Size: 16
  * Epochs: 20
  * Optimizer: Adam
  * Loss Function: Categorical Cross-Entropy
  * Clinical DNN Layers: 2 Dense Layers (128, 64 neurons)
  * Transformer Attention Heads: 4
  * Embedding Dimension: 128


6. REPOSITORY STRUCTURE

-----------------------

.

├── data/

│   ├── raw/                      # MIMIC3d CSV and PIID image folder

│   └── processed/                # Preprocessed/aligned multimodal datasets

├── Pressure-Ulcer-MultiModal-Version1.ipynb            # Complete end-to-end Python pipeline

│                                 # (Preprocessing, CNN+DNN Extraction,

│                                 # Transformer Fusion, Training, Evaluation,

│                                 # Statistical Tests, and XAI Explanations)

7. EXECUTION & USAGE
--------------------
To run the complete pipeline using the single script:

1. Install dependencies:

   $ pip install -r requirements.txt

2. Run the main execution script:

   $ python main.py

8. CITATION & REFERENCE
-----------------------
If you utilize this architecture, code, or experimental protocol, please cite:

Ramya Chinnasamy, Malliga Subramanian. 
"Multimodal Deep Learning for Automated Pressure Ulcer Risk Assessment in Nursing Care." 

