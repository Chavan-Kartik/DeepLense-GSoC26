from src.data_loader import get_dataloaders
import torch

def test():
    root_path = "data/raw/Task_I_Classification"
    print(f"--- Checking data at {root_path} ---")
    
    try:
        train_loader, val_loader = get_dataloaders(root_path, batch_size=32)
        
        # Pull one batch
        images, labels = next(iter(train_loader))
        
        print(f"Success! Batch successfully loaded.")
        print(f"Image Batch Shape: {images.shape}  -> (Batch, Channels, Height, Width)")
        print(f"Labels Batch Shape: {labels.shape}")
        print(f"Unique labels in this batch: {torch.unique(labels)}")
        print(f"Max pixel value: {images.max():.4f}, Min pixel value: {images.min():.4f}")
        
    except Exception as e:
        print(f"Pipeline Error: {e}")

if __name__ == "__main__":
    test()