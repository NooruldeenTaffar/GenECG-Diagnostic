# src/Data_pipeline/dataset.py

import os
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms


class GenECGHFDataset(Dataset):
    """
    Loads GenECG from Hugging Face with authentication to avoid rate limits.
    Requires environment variable HF_TOKEN to be set (Read access token).
    """

    def __init__(self, image_size: int = 224, split: str = "train"):
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set.\n"
                "In VS Code terminal run:\n"
                'export HF_TOKEN="YOUR_TOKEN_HERE"\n'
                "Then run your script again."
            )

        # Load dataset (cached automatically after first download)
        self.dataset = load_dataset(
            "edcci/GenECG",
            split=split,
            token=token,
        )

        self.label_names = self.dataset.features["label"].names  # e.g., ['NORM','MI','STTC','CD','HYP']

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            ),
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]

        # Hugging Face returns a PIL Image for the "image" column
        image = item["image"].convert("RGB")
        label = int(item["label"])

        image = self.transform(image)
        return image, label
