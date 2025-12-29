from src.Data_pipeline.dataloader import get_dataloader

loader = get_dataloader(batch_size=2)

x, y = next(iter(loader))
print("Images shape:", x.shape)
print("Labels:", y)

