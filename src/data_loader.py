import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class DeepLenseDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train'):
        """
        Args:
            root_dir (str): Path to the main dataset folder containing 'train' and 'val'.
            split (str): Either 'train' or 'val'.
        """
        self.split_dir = os.path.join(root_dir, split)
        
        # Explicit mapping mapping based on the directory names we found
        self.classes = ['no', 'sphere', 'vort']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.filepaths = []
        self.labels = []
        
        # Build the index
        for cls_name in self.classes:
            cls_folder = os.path.join(self.split_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
                
            for file in os.listdir(cls_folder):
                if file.endswith('.npy'): # Safely ignores .DS_Store
                    self.filepaths.append(os.path.join(cls_folder, file))
                    self.labels.append(self.class_to_idx[cls_name])
                    
        # Physics-informed augmentations (Rotation and flips don't change the physics of a lens)
        if split == 'train':
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                # Random 90, 180, 270 degree rotations
                T.RandomApply([T.RandomRotation((90, 90))], p=0.33),
                T.RandomApply([T.RandomRotation((180, 180))], p=0.33),
                T.RandomApply([T.RandomRotation((270, 270))], p=0.33)
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.filepaths[idx]
        label = self.labels[idx]

        arr = np.load(path)
        
        # Squeeze out any extra dimensions first, then add back exactly one
        # This turns [1, 150, 150] or [150, 150] into exactly [1, 150, 150]
        arr = np.squeeze(arr)
        arr = np.expand_dims(arr, axis=0)
        
        tensor_img = torch.from_numpy(arr).float()

        if self.transform:
            tensor_img = self.transform(tensor_img)

        return tensor_img, label

def get_dataloaders(root_dir: str, batch_size: int = 32) -> tuple[DataLoader, DataLoader]:
    """Creates and returns the train and validation dataloaders."""
    train_dataset = DeepLenseDataset(root_dir, split='train')
    val_dataset = DeepLenseDataset(root_dir, split='val')

    # pin_memory=True dramatically speeds up data transfer to the GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    return train_loader, val_loader