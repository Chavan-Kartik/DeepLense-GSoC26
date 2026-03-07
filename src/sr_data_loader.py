from torch.utils.data import random_split
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SuperResDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        """
        Custom Dataset for matching Low-Res and High-Res image pairs.
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        
        # Sort the files so LR and HR pairs align perfectly
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.npy')])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.npy')])
        
        # Sanity check: Ensure we have exact pairs
        assert len(self.lr_files) == len(self.hr_files), "Mismatch: Unequal number of LR and HR files!"

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        # Load the corresponding LR and HR numpy arrays
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        
        lr_image = np.load(lr_path).astype(np.float32)
        hr_image = np.load(hr_path).astype(np.float32)
        
        # Min-Max Normalization to scale pixel values between 0.0 and 1.0 safely
        lr_image = (lr_image - np.min(lr_image)) / (np.max(lr_image) - np.min(lr_image) + 1e-8)
        hr_image = (hr_image - np.min(hr_image)) / (np.max(hr_image) - np.min(hr_image) + 1e-8)

        # Convert to PyTorch tensors and add the channel dimension (1, H, W) for grayscale
       # Convert to PyTorch tensors
        lr_tensor = torch.tensor(lr_image, dtype=torch.float32)
        hr_tensor = torch.tensor(hr_image, dtype=torch.float32)

        # Bomb-proof dimension fix: squeeze out any extra 1s, then add exactly one channel dimension
        lr_tensor = lr_tensor.squeeze().unsqueeze(0)
        hr_tensor = hr_tensor.squeeze().unsqueeze(0)

        return lr_tensor, hr_tensor

def get_sr_dataloaders(lr_dir, hr_dir, val_split=0.2, batch_size=32):
    """
    Initializes the Dataset and mathematically splits it into Training and Validation sets.
    """
    # 1. Load the entire dataset from the single folders
    full_dataset = SuperResDataset(lr_dir, hr_dir)
    
    # 2. Calculate the split sizes (80% Train, 20% Validation)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    # 3. Split the data randomly, but use a fixed seed (42) so it's reproducible for science!
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # 4. Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader