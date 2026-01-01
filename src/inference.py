#!/usr/bin/env python3
"""
GenECG Diagnostic Inference Script
===================================
Feed an ECG image, get SNOMED-CT diagnostic predictions.

Usage:
    python src/inference.py path/to/ecg_image.png
    python src/inference.py data/Raw/GenECG/Dataset_A_ECGs_without_imperfections/00000/00001_hr_1R.png
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def load_vit_model(checkpoint_path: str, device: torch.device):
    """Load the trained ViT model."""
    from transformers import ViTForImageClassification
    
    print(f"📦 Loading ViT model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=checkpoint['num_classes'],
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['snomed_cols']


def predict(model, image_path: str, snomed_cols: list, device: torch.device, top_k: int = 10):
    """Run prediction on a single ECG image."""
    
    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(img_tensor).logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    
    # Get top predictions
    top_indices = probs.argsort()[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        snomed_code = snomed_cols[idx].replace('SNOMED_', '')
        results.append({
            'snomed_code': snomed_code,
            'full_name': snomed_cols[idx],
            'probability': float(probs[idx]),
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="GenECG Diagnostic Inference")
    parser.add_argument("image", type=str, help="Path to ECG image")
    parser.add_argument("--model", type=str, 
                        default="runs/vit/vit_multilabel_checkpoint.pt",
                        help="Path to ViT checkpoint")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top predictions to show")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Probability threshold for positive prediction")
    args = parser.parse_args()
    
    # Check paths
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load model
    model, snomed_cols = load_vit_model(args.model, device)
    print(f"✅ Model loaded with {len(snomed_cols)} SNOMED classes\n")
    
    # Run prediction
    print(f"🔍 Analyzing: {args.image}\n")
    results = predict(model, args.image, snomed_cols, device, args.top_k)
    
    # Display results
    print("=" * 60)
    print("📋 SNOMED-CT DIAGNOSTIC PREDICTIONS")
    print("=" * 60)
    print(f"{'SNOMED Code':<15} {'Probability':>12} {'Status':<10}")
    print("-" * 60)
    
    for r in results:
        status = "🔴 POSITIVE" if r['probability'] >= args.threshold else ""
        print(f"{r['snomed_code']:<15} {r['probability']:>12.1%} {status:<10}")
    
    print("-" * 60)
    
    # Summary
    positive_count = sum(1 for r in results if r['probability'] >= args.threshold)
    print(f"\n📊 Summary: {positive_count} diagnoses above {args.threshold:.0%} threshold")
    
    return results


if __name__ == "__main__":
    main()
