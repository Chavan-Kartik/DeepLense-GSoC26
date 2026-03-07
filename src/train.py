import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    all_labels = []
    all_probs = []
    running_loss = 0.0
    
    for images, labels in tqdm(dataloader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        probs = torch.softmax(outputs, dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        running_loss += loss.item()
        
    return running_loss / len(dataloader), np.array(all_labels), np.array(all_probs)