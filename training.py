import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from datasets import load_dataset
import torchvision.transforms.functional as F
import argparse
from huggingface_hub import HfApi, create_repo

def fourier_features(img):
    # Convert PIL Image to numpy array
    img = np.array(img)
    
    # Process each channel separately
    channels = []
    ffts = []
    
    for channel in cv2.split(img):
        # Normalize channel to [0,1]
        channel = channel.astype(np.float32) / 255.0
        
        # Apply FFT
        fft = np.fft.fft2(channel)
        fft_shift = np.fft.fftshift(fft)
        ffts.append(fft_shift)
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_shift)
        # Apply log transform to compress dynamic range
        log_magnitude = np.log1p(magnitude)
        
        # Normalize to [0,1] range
        log_magnitude = (log_magnitude - np.min(log_magnitude)) / (np.max(log_magnitude) - np.min(log_magnitude))
        channels.append(log_magnitude)
    
    return ffts, np.stack(channels, axis=-1)

def apply_fourier_filter_color(fft_shifts, filter_type="HighEmphasis", cutoff_freq=10, gain_low=5.0, gain_high=1.0):
    """Apply Fourier filter to each color channel separately"""
    filtered_channels = []
    
    for fft_shift in fft_shifts:
        # Create frequency distance grid
        rows, cols = fft_shift.shape
        crow, ccol = rows // 2, cols // 2
        x = np.arange(cols)
        y = np.arange(rows)
        u, v = np.meshgrid(x, y)
        D = np.sqrt((u - ccol)**2 + (v - crow)**2)
        
        # Create filter mask
        if filter_type == "HighEmphasis":
            hpf_base = 1 - np.exp(-D**2 / (2 * cutoff_freq**2))
            H = gain_low + gain_high * hpf_base
        
        # Apply filter
        fft_filtered_shifted = fft_shift * H
        fft_filtered = np.fft.ifftshift(fft_filtered_shifted)
        image_filtered = np.fft.ifft2(fft_filtered)
        image_filtered = np.real(image_filtered)
        
        # Normalize
        image_filtered = (image_filtered - np.min(image_filtered)) / (np.max(image_filtered) - np.min(image_filtered))
        filtered_channels.append(image_filtered)
    
    return np.stack(filtered_channels, axis=-1)

class NYUDepthDataset(Dataset):
    def __init__(self, dataset, image_transform=None, depth_transform=None, apply_fourier=True):
        self.dataset = dataset
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.apply_fourier = apply_fourier
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Load and convert images
        # NYU Depth V2 dataset has 'image' and 'depth_map' keys
        image = sample['image'].convert('RGB')
        depth = sample['depth_map'].convert('L')
        
        if self.apply_fourier:
            # Apply Fourier transform and filtering
            ffts, _ = fourier_features(image)
            image = apply_fourier_filter_color(ffts)
            # Convert back to PIL Image
            image = Image.fromarray((image * 255).astype(np.uint8))
        
        if self.image_transform:
            image = self.image_transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)
            
        return image, depth

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10):
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Training loop
        for images, depths in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            depths = depths.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Resize depths to match model output size
            depths = F.resize(depths, size=(378, 378))
            
            loss = torch.nn.functional.mse_loss(outputs.predicted_depth, depths)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, depths in val_loader:
                images = images.to(device)
                depths = depths.to(device)
                
                outputs = model(images)
                # Resize depths to match model output size
                depths = F.resize(depths, size=(378, 378))
                loss = torch.nn.functional.mse_loss(outputs.predicted_depth, depths)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
        # Log to wandb
        wandb.log({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'epoch': epoch + 1
        })

def push_to_hub(model, model_name, token):
    """Push the model to Hugging Face Hub"""
    print(f"Pushing model to Hugging Face Hub as '5524-Group/{model_name}'...")
    
    # Create repository if it doesn't exist
    repo_id = f"5524-Group/{model_name}"
    try:
        create_repo(repo_id, token=token, exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Save model locally first
    model.save_pretrained(f"./{model_name}")
    
    # Push to hub
    api = HfApi()
    try:
        api.upload_folder(
            folder_path=f"./{model_name}",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print(f"Successfully pushed model to {repo_id}")
    except Exception as e:
        print(f"Error pushing model: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Depth-Anything model with Fourier preprocessing')
    parser.add_argument('--model_name', type=str, required=True, help='Name for the model on Hugging Face Hub')
    parser.add_argument('--hf_token', type=str, required=True, help='Hugging Face authentication token')
    parser.add_argument('--wandb_key', type=str, required=True, help='Weights & Biases API key')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.login(key=args.wandb_key)
    wandb.init(project="depth-anything-training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load NYU Depth V2 dataset
    print("Loading NYU Depth V2 dataset...")
    dataset = load_dataset("sayakpaul/nyu_depth_v2", split="train", trust_remote_code=True)
    
    # Split dataset into train and validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    
    # Load model and processor
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)  # Enable fast processor
    
    model = model.to(device)
    
    # Define transforms for images
    image_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define transforms for depth maps
    depth_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        # Normalize depth values to [0, 1]
        transforms.Lambda(lambda x: x / 255.0)
    ])
    
    # Create datasets with Fourier preprocessing
    train_dataset = NYUDepthDataset(
        dataset=train_dataset,
        image_transform=image_transform,
        depth_transform=depth_transform,
        apply_fourier=True
    )
    
    val_dataset = NYUDepthDataset(
        dataset=val_dataset,
        image_transform=image_transform,
        depth_transform=depth_transform,
        apply_fourier=True
    )
    
    # Create data loaders with reduced number of workers
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1)
    
    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # Train model
    train_model(model, train_loader, val_loader, optimizer, device, num_epochs=args.num_epochs)
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    
    # Push model to Hugging Face Hub
    push_to_hub(model, args.model_name, args.hf_token)
    
    wandb.finish()

if __name__ == '__main__':
    main()
