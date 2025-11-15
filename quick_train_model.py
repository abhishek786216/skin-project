"""
Quick Model Training for Skin Cancer Classification
Fast training with optimized parameters - Results in 30-60 minutes
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
import os

# Import our data loading
from test_preprocessing import get_train_val_test_splits, SimpleDataset

class QuickSkinCancerTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_gpu()
        
    def setup_gpu(self):
        """Optimize GPU settings for fast training"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            
            # Get GPU memory and optimize batch size
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {gpu_memory:.1f}GB")
            
            # Optimized batch sizes for speed
            if gpu_memory >= 8:
                self.batch_size = 32
                self.num_workers = 0  # Windows compatibility
            elif gpu_memory >= 6:
                self.batch_size = 24
                self.num_workers = 0  # Windows compatibility
            else:
                self.batch_size = 16
                self.num_workers = 0  # Windows compatibility
                
            print(f"Optimized: batch_size={self.batch_size}, num_workers={self.num_workers}")
        else:
            print("WARNING: No GPU available - training will be slow!")
            self.batch_size = 8
            self.num_workers = 0

    def get_model(self, architecture, num_classes=7):
        """Get pre-trained model with minimal modifications for speed"""
        if architecture == 'resnet':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif architecture == 'mobilenet':
            model = models.mobilenet_v3_small(pretrained=True)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif architecture == 'efficientnet':
            # Use smaller EfficientNet for speed
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
        return model.to(self.device)

    def get_transforms(self):
        """Fast data augmentation - minimal for speed"""
        from PIL import Image
        
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform

    def train_single_model(self, architecture, epochs=8):
        """Train single model quickly with good parameters"""
        print(f"\n{'='*60}")
        print(f"QUICK TRAINING: {architecture.upper()}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Get data
        train_df, val_df, test_df = get_train_val_test_splits()
        train_transform, val_transform = self.get_transforms()
        
        train_dataset = SimpleDataset(train_df, train_df.img_dirs, transform=train_transform)
        val_dataset = SimpleDataset(val_df, val_df.img_dirs, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                 shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                               shuffle=False, num_workers=self.num_workers, pin_memory=True)
        
        # Model setup
        model = self.get_model(architecture)
        criterion = nn.CrossEntropyLoss()
        
        # Optimized learning rate and optimizer
        if architecture == 'resnet':
            lr = 0.001
        elif architecture == 'mobilenet':
            lr = 0.0005
        else:  # efficientnet
            lr = 0.0008
            
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        
        print(f"Training {architecture} for {epochs} epochs with lr={lr}")
        print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Training loop
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (data, target) in enumerate(train_bar):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
                
                # Update progress bar
                train_acc = 100. * train_correct / train_total
                train_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{train_acc:.2f}%'})
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            
            val_acc = 100. * val_correct / val_total
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            scheduler.step(val_loss / len(val_loader))
            
            # Early stopping if accuracy is very high
            if val_acc > 85:
                print(f"Early stopping at {val_acc:.2f}% accuracy")
                break
        
        total_time = time.time() - start_time
        
        # Save best model
        model_path = f'data/{architecture}_quick_model.pth'
        torch.save({
            'model_state_dict': best_model_state,
            'architecture': architecture,
            'val_accuracy': best_val_acc,
            'training_time': total_time,
            'epochs_trained': epoch + 1
        }, model_path)
        
        result = {
            'architecture': architecture,
            'val_accuracy': best_val_acc,
            'training_time': total_time,
            'epochs_trained': epoch + 1,
            'model_path': model_path
        }
        
        print(f"✓ {architecture} completed: {best_val_acc:.2f}% accuracy in {total_time/60:.1f} minutes")
        torch.cuda.empty_cache()  # Clear GPU memory
        
        return result

    def quick_comparison(self):
        """Train all models quickly and compare"""
        print("QUICK SKIN CANCER MODEL COMPARISON")
        print("Training 3 models with optimized parameters...")
        print("Expected time: 15-30 minutes total")
        
        architectures = ['resnet', 'mobilenet', 'efficientnet']
        results = []
        
        for arch in architectures:
            try:
                result = self.train_single_model(arch, epochs=6)  # Reduced epochs for speed
                results.append(result)
            except Exception as e:
                print(f"Error training {arch}: {e}")
                continue
        
        # Save results
        results_path = 'data/quick_model_comparison.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print comparison
        print("\n" + "="*80)
        print("QUICK TRAINING RESULTS")
        print("="*80)
        
        results.sort(key=lambda x: x['val_accuracy'], reverse=True)
        
        for i, result in enumerate(results):
            status = "🥇 BEST" if i == 0 else "🥈 GOOD" if i == 1 else "🥉 OK"
            print(f"{status} {result['architecture'].upper():12}: "
                  f"{result['val_accuracy']:5.2f}% accuracy, "
                  f"{result['training_time']/60:4.1f}min, "
                  f"{result['epochs_trained']:2d} epochs")
        
        best_model = results[0]
        print(f"\n✓ RECOMMENDED: {best_model['architecture'].upper()} with {best_model['val_accuracy']:.2f}% accuracy")
        print(f"  Model saved as: {best_model['model_path']}")
        
        return results

def main():
    """Main training function"""
    trainer = QuickSkinCancerTrainer()
    results = trainer.quick_comparison()
    
    print(f"\n{'='*60}")
    print("QUICK TRAINING COMPLETED!")
    print(f"{'='*60}")
    print("Next steps:")
    print("1. Run: streamlit run simple_app.py")
    print("2. Check model comparison in the web interface")
    print("3. Use the best model for production")
    
    return results

if __name__ == "__main__":
    results = main()