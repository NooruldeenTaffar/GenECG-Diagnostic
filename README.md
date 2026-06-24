# GenECG-Diagnostic

## Problem

Reading a 12-lead ECG image and turning it into a diagnosis is normally a two-step job for a clinician: find each of the 12 leads on the page, then read the waveform pattern across all of them to reach a diagnostic impression. This project automates both steps from a raw ECG image, with the goal of outputting standardised, machine-readable diagnostic codes (SNOMED-CT) rather than free text.

## Approach

A two-stage computer vision pipeline:

1. **Lead localisation (YOLOv8/YOLO11)** – detects the 12 individual lead panels within a raw 3×4 grid ECG image.
2. **Multi-label diagnosis (Vision Transformer)** – a `google/vit-base-patch16-224` backbone, fine-tuned to predict multiple SNOMED-CT diagnostic codes per ECG, trained with `BCEWithLogitsLoss` to allow more than one diagnosis per image.

**Input:** Raw 12-lead ECG image (3×4 grid)
**Output:** Probabilistic SNOMED-CT diagnostic predictions

## Status

Both stages have been trained end-to-end on the GenECG dataset, and `src/inference.py` runs a full image-to-diagnosis pass. Quantitative evaluation (mAP for lead detection, per-label F1/AUC for diagnosis) is tracked in the training notebook and is the next thing I want to formalise and report here.

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

## YOLO Label Generation

Generate bounding-box labels for ECG leads:

```bash
python src/Data_pipeline/run_mass_label_generation.py \
  --raw-dir data/Raw/GenECG/Dataset_A_ECGs_without_imperfections \
  --output-dir data/Processed/YOLO_Labels
```

## YOLO Training

```bash
python src/Training/train_yolo.py \
  --data data_A.yaml \
  --epochs 50 \
  --batch 16
```

## Inference (ViT)

Run diagnosis on a single ECG image:

```bash
python src/inference.py path/to/ecg.png \
  --model runs/vit/vit_multilabel_checkpoint.pt
```

Output includes:

- SNOMED-CT codes
- Prediction probabilities
- Threshold-based positives

## Skills Demonstrated

- Two-stage computer vision pipeline design (object detection feeding a downstream classifier)
- Object detection with YOLOv8/YOLO11 on a custom heuristic-labelled dataset
- Vision Transformer fine-tuning for multi-label classification
- Mapping model outputs to a real clinical coding standard (SNOMED-CT)
- End-to-end ML engineering: data acquisition, label generation, training, and a runnable inference script

## Notes

- Multi-label classification (`BCEWithLogitsLoss`)
- ViT backbone: `google/vit-base-patch16-224`
- Dataset includes ECGs with and without imperfections
