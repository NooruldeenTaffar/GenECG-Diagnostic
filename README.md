# GenECG-Diagnostic

## Overview

**GenECG-Diagnostic** is a research-oriented pipeline for automated diagnosis of 12-lead ECG images using computer vision.
The project uses the **GenECG** dataset and applies a **two-stage deep learning workflow**:

1. **Lead localisation** using YOLOv8
2. **Multi-label diagnosis** using a Vision Transformer (ViT) with SNOMED-CT outputs

---

## Pipeline Summary

**Input:** Raw 12-lead ECG image (3×4 grid)
**Output:** Probabilistic SNOMED-CT diagnostic predictions

**Stages**

1. **Data acquisition** – Download GenECG from Hugging Face
2. **YOLO label generation** – Heuristic 3×4 grid → 12 lead bounding boxes
3. **YOLO training** – Detect individual ECG leads
4. **ViT classification** – Multi-label diagnosis from ECG images

---

## Repository Structure

```
.
├── src/
│   ├── Data_pipeline/
│   │   ├── download_data.py
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   ├── yolo_labels.py
│   │   └── run_mass_label_generation.py
│   ├── Training/
│   │   └── train_yolo.py
│   └── inference.py
├── notebooks/
│   └── GenECG_ViT_Training_Colab.ipynb
├── data_A.yaml
├── data_colab.yaml
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
HUGGING_FACE_TOKEN=your_hf_token
HF_TOKEN=your_hf_token
```

Download the dataset:

```bash
python src/Data_pipeline/download_data.py
```

---

## YOLO Label Generation

Generate bounding-box labels for ECG leads:

```bash
python src/Data_pipeline/run_mass_label_generation.py \
  --raw-dir data/Raw/GenECG/Dataset_A_ECGs_without_imperfections \
  --output-dir data/Processed/YOLO_Labels
```

---

## YOLO Training

```bash
python src/Training/train_yolo.py \
  --data data_A.yaml \
  --epochs 50 \
  --batch 16
```

---

## Inference (ViT)

Run diagnosis on a single ECG image:

```bash
python src/inference.py path/to/ecg.png \
  --model runs/vit/vit_multilabel_checkpoint.pt
```

Output includes:

* SNOMED-CT codes
* Prediction probabilities
* Threshold-based positives

---

## Notes

* Multi-label classification (`BCEWithLogitsLoss`)
* ViT backbone: `google/vit-base-patch16-224`
* Dataset includes ECGs **with and without imperfections**




