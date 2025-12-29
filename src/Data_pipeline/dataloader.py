from torch.utils.data import DataLoader
from src.Data_pipeline.dataset import GenECGHFDataset

def get_dataloader(batch_size=8, image_size=224, shuffle=True):
    dataset = GenECGHFDataset(image_size=image_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # important for Mac M1
    )

    return loader
