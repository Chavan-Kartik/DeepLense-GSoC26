import torch
import torch.nn as nn
import timm

class LensingClassifier(nn.Module):
    def __init__(self, model_name: str = 'convnext_nano', num_classes: int = 3, pretrained: bool = True):
        super(LensingClassifier, self).__init__()
        
        # Load the base model from timm library
        # 'in_chans=1' is crucial because our lensing data is grayscale/single-channel
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes, 
            in_chans=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def get_model(num_classes: int = 3) -> nn.Module:
    """Helper function to initialize and return the model."""
    model = LensingClassifier(num_classes=num_classes)
    return model

if __name__ == "__main__":
    # Quick shape test
    model = get_model()
    test_input = torch.randn(1, 1, 150, 150)
    output = model(test_input)
    print(f"Model successfully initialized.")
    print(f"Input Shape: {test_input.shape}")
    print(f"Output Shape: {output.shape} (Should be [1, 3])")