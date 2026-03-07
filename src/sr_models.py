import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        """
        Super-Resolution Convolutional Neural Network (SRCNN)
        Adapted with internal Bicubic Upscaling for 2x resolution enhancement.
        """
        super(SRCNN, self).__init__()
        
        # Layer 1: Patch Extraction
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2: Non-linear Mapping
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Layer 3: Reconstruction
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

    def forward(self, x):
        # THE FIX: Mathematically stretch the 75x75 image to 150x150 first
        x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        
        # Then let the network learn how to sharpen the blurry stretched pixels
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x