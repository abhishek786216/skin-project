"""
Advanced Model Training with Hyperparameter Tuning and Multiple Architectures
Includes ResNet, Xception, and MobileNet with GPU optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import os
import json
from test_preprocessing import SimpleDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, ParameterGrid
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ResNetModel(nn.Module):
    """ResNet-based model for skin cancer classification"""
    
    def __init__(self, num_classes=7, dropout_rate=0.5, hidden_size=512):
        super(ResNetModel, self).__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.fc.in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class XceptionModel(nn.Module):
    """Xception-inspired model for skin cancer classification"""
    
    def __init__(self, num_classes=7, dropout_rate=0.5, hidden_size=512):
        super(XceptionModel, self).__init__()
        # Use EfficientNet as Xception alternative (PyTorch doesn't have native Xception)
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-15]:
            param.requires_grad = False
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.classifier[1].in_features, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class MobileNetModel(nn.Module):
    """MobileNet model for skin cancer classification"""
    
    def __init__(self, num_classes=7, dropout_rate=0.5, hidden_size=512):
        super(MobileNetModel, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier[0].in_features, hidden_size),
            nn.Hardswish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Hardswish(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class ModelTrainer:
    """Advanced model trainer with hyperparameter tuning"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.best_models = {}
        self.training_history = {}
        
    def create_data_loaders(self, train_df, val_df, test_df, img_dirs, batch_size=32, num_workers=4):
        """Create optimized data loaders with augmentation"""
        
        # Enhanced transforms for training
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Standard transforms for validation/test
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets with transforms
        train_dataset = SimpleDataset(train_df, img_dirs, transform=train_transform)
        val_dataset = SimpleDataset(val_df, img_dirs, transform=val_transform)
        test_dataset = SimpleDataset(test_df, img_dirs, transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, model, train_loader, val_loader, params, model_name):
        """Train model with given hyperparameters"""
        
        model = model.to(self.device)
        
        # Optimizer setup
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        elif params['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=params['weight_decay'])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        
        # Scheduler setup
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        # Loss function with class weights for imbalanced data
        class_weights = torch.tensor([1.5, 3.0, 1.2, 4.0, 0.3, 2.5, 4.5]).to(self.device)  # Adjusted weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        best_val_acc = 0
        patience_counter = 0
        
        print(f"\nTraining {model_name} with parameters: {params}")
        
        # Print GPU memory info before training
        if self.device.type == 'cuda':
            print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
        
        for epoch in range(params['epochs']):
            start_time = time.time()
            
            # Clear GPU cache at start of each epoch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Training phase
            model.train()
            train_loss = 0
            
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['epochs']}")):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            epoch_time = time.time() - start_time
            
            # GPU memory monitoring
            gpu_memory_info = ""
            if self.device.type == 'cuda':
                gpu_memory_info = f", GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB"
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.2f}%, Time: {epoch_time:.2f}s{gpu_memory_info}")
            
            # Early stopping
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'epoch': epoch,
                    'params': params,
                    'model_name': model_name
                }, f'best_{model_name.lower()}_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= params['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return model, best_val_acc, history
    
    def evaluate_model(self, model, test_loader, class_names):
        """Comprehensive model evaluation"""
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, 
                                     output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        return accuracy, report, cm, all_predictions, all_labels, all_probabilities
    
    def hyperparameter_tuning(self, model_class, train_loader, val_loader, param_grid, model_name):
        """Perform hyperparameter tuning"""
        
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING FOR {model_name.upper()}")
        print(f"{'='*60}")
        
        best_score = 0
        best_params = None
        best_model = None
        tuning_results = []
        
        param_combinations = list(ParameterGrid(param_grid))
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for i, params in enumerate(param_combinations):
            print(f"\nCombination {i+1}/{len(param_combinations)}: {params}")
            
            # Create model instance
            model = model_class(
                num_classes=7, 
                dropout_rate=params['dropout_rate'],
                hidden_size=params['hidden_size']
            )
            
            try:
                # Train model
                trained_model, val_acc, history = self.train_model(
                    model, train_loader, val_loader, params, model_name
                )
                
                # Store results
                result = {
                    'params': params,
                    'val_accuracy': val_acc,
                    'model_name': model_name
                }
                tuning_results.append(result)
                
                # Update best model
                if val_acc > best_score:
                    best_score = val_acc
                    best_params = params
                    best_model = trained_model
                
                print(f"Validation Accuracy: {val_acc:.2f}%")
                
            except Exception as e:
                print(f"Error training with params {params}: {str(e)}")
                continue
        
        # Store best model and results
        self.best_models[model_name] = {
            'model': best_model,
            'accuracy': best_score,
            'params': best_params,
            'tuning_results': tuning_results
        }
        
        print(f"\nBest {model_name} - Accuracy: {best_score:.2f}%, Params: {best_params}")
        return best_model, best_score, best_params

def main():
    print("Advanced Skin Cancer Classification Model Training")
    print("="*60)
    
    # Enhanced GPU setup and optimization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Set memory optimization flags
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input size
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Enable memory growth
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        
        print("GPU optimizations enabled:")
        print("  - CUDNN benchmark mode: ON")
        print("  - Memory fraction limit: 90%")
        print("  - GPU cache cleared")
    else:
        print("WARNING: CUDA not available. Training will be much slower on CPU!")
        print("Install CUDA-enabled PyTorch for GPU acceleration.")
    
    # Load and prepare data
    print("\nPreparing data...")
    data_dir = r"c:\Users\abhis\OneDrive\Desktop\skin\data"
    metadata_path = os.path.join(data_dir, 'raw', 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)
    
    # Encode labels
    le = LabelEncoder()
    df['dx_encoded'] = le.fit_transform(df['dx'])
    class_names = list(le.classes_)
    
    print(f"Classes: {class_names}")
    print(f"Total samples: {len(df)}")
    
    # Find image directories
    img_dirs = []
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        dir_path = os.path.join(data_dir, 'raw', part)
        if os.path.exists(dir_path):
            img_dirs.append(dir_path)
    
    # Create splits
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['dx'], random_state=42)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize trainer
    trainer = ModelTrainer(device)
    
    # Optimize batch size and num_workers based on hardware
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 12:
            batch_size, num_workers = 64, 8  # High-end GPUs (RTX 4080+, A100)
        elif gpu_memory >= 8:
            batch_size, num_workers = 48, 6  # Mid-range GPUs (RTX 3080, 4070)
        elif gpu_memory >= 6:
            batch_size, num_workers = 32, 4  # Entry GPUs (RTX 3060, 4060)
        else:
            batch_size, num_workers = 16, 2  # Low VRAM GPUs (GTX 1660, etc.)
        
        print(f"Optimized for {gpu_memory:.1f}GB GPU: batch_size={batch_size}, num_workers={num_workers}")
    else:
        batch_size, num_workers = 8, 2  # Conservative for CPU
        print(f"CPU optimization: batch_size={batch_size}, num_workers={num_workers}")
    
    # Create optimized data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        train_df, val_df, test_df, img_dirs, batch_size=batch_size, num_workers=num_workers
    )
    
    # Define hyperparameter grids for each model (using optimized batch size)
    param_grids = {
        'ResNet': {
            'lr': [0.001, 0.0005, 0.0001],
            'batch_size': [batch_size],
            'epochs': [15 if device.type == 'cuda' else 10],
            'dropout_rate': [0.3, 0.5, 0.7],
            'hidden_size': [256, 512, 1024],
            'weight_decay': [1e-4, 1e-5],
            'optimizer': ['adam', 'adamw'],
            'patience': [5]
        },
        'MobileNet': {
            'lr': [0.001, 0.0005],
            'batch_size': [batch_size],
            'epochs': [15 if device.type == 'cuda' else 10],
            'dropout_rate': [0.3, 0.5],
            'hidden_size': [256, 512],
            'weight_decay': [1e-4, 1e-5],
            'optimizer': ['adam', 'adamw'],
            'patience': [5]
        },
        'Xception': {
            'lr': [0.0005, 0.0001],
            'batch_size': [batch_size],
            'epochs': [15 if device.type == 'cuda' else 10],
            'dropout_rate': [0.4, 0.6],
            'hidden_size': [512, 1024],
            'weight_decay': [1e-4],
            'optimizer': ['adamw'],
            'patience': [5]
        }
    }
    
    # Model classes
    model_classes = {
        'ResNet': ResNetModel,
        'MobileNet': MobileNetModel,
        'Xception': XceptionModel
    }
    
    # Train and tune each model
    final_results = {}
    
    for model_name, model_class in model_classes.items():
        print(f"\n{'='*80}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*80}")
        
        # Hyperparameter tuning
        best_model, best_score, best_params = trainer.hyperparameter_tuning(
            model_class, train_loader, val_loader, param_grids[model_name], model_name
        )
        
        # Final evaluation on test set
        print(f"\nFinal evaluation of best {model_name} on test set...")
        test_accuracy, test_report, cm, predictions, labels, probabilities = trainer.evaluate_model(
            best_model, test_loader, class_names
        )
        
        # Store results
        final_results[model_name] = {
            'model': best_model,
            'test_accuracy': test_accuracy,
            'val_accuracy': best_score,
            'params': best_params,
            'test_report': test_report,
            'confusion_matrix': cm,
            'model_size': sum(p.numel() for p in best_model.parameters()),
            'trainable_params': sum(p.numel() for p in best_model.parameters() if p.requires_grad)
        }
        
        print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
        print(f"{model_name} Parameters: {final_results[model_name]['model_size']:,}")
        print(f"{model_name} Trainable: {final_results[model_name]['trainable_params']:,}")
    
    # Select best model
    best_model_name = max(final_results.keys(), key=lambda x: final_results[x]['test_accuracy'])
    best_overall_model = final_results[best_model_name]
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL SELECTION")
    print(f"{'='*80}")
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {best_overall_model['test_accuracy']:.4f}")
    print(f"Validation Accuracy: {best_overall_model['val_accuracy']:.2f}%")
    print(f"Parameters: {best_overall_model['params']}")
    
    # Save best model for production
    production_model_path = os.path.join(data_dir, 'skin_cancer_model.pth')
    torch.save({
        'model_state_dict': best_overall_model['model'].state_dict(),
        'class_names': class_names,
        'model_name': best_model_name,
        'test_accuracy': best_overall_model['test_accuracy'],
        'val_accuracy': best_overall_model['val_accuracy'],
        'params': best_overall_model['params'],
        'model_size': best_overall_model['model_size'],
        'trainable_params': best_overall_model['trainable_params'],
        'training_date': datetime.now().isoformat()
    }, production_model_path)
    
    # Save model comparison results
    comparison_results = {}
    for name, results in final_results.items():
        comparison_results[name] = {
            'test_accuracy': float(results['test_accuracy']),
            'val_accuracy': float(results['val_accuracy']),
            'params': results['params'],
            'model_size': int(results['model_size']),
            'trainable_params': int(results['trainable_params'])
        }
    
    with open(os.path.join(data_dir, 'model_comparison.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\nModel saved to: {production_model_path}")
    print(f"Model comparison saved to: {os.path.join(data_dir, 'model_comparison.json')}")
    
    # Create visualization of results
    create_model_comparison_plot(final_results, data_dir)
    
    return final_results, best_model_name

def create_model_comparison_plot(results, data_dir):
    """Create visualization comparing all models"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    models = list(results.keys())
    test_accs = [results[model]['test_accuracy'] for model in models]
    val_accs = [results[model]['val_accuracy'] for model in models]
    model_sizes = [results[model]['model_size'] / 1e6 for model in models]  # In millions
    trainable_params = [results[model]['trainable_params'] / 1e6 for model in models]  # In millions
    
    # Test accuracy comparison
    bars1 = ax1.bar(models, test_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(test_accs):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Validation accuracy comparison
    bars2 = ax2.bar(models, val_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy (%)')
    for i, v in enumerate(val_accs):
        ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Model size comparison
    bars3 = ax3.bar(models, model_sizes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_title('Total Model Parameters', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Parameters (Millions)')
    for i, v in enumerate(model_sizes):
        ax3.text(i, v + 0.1, f'{v:.1f}M', ha='center', fontweight='bold')
    
    # Trainable parameters comparison
    bars4 = ax4.bar(models, trainable_params, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax4.set_title('Trainable Parameters', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Trainable Parameters (Millions)')
    for i, v in enumerate(trainable_params):
        ax4.text(i, v + 0.1, f'{v:.1f}M', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'model_comparison_plot.png'), dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to: {os.path.join(data_dir, 'model_comparison_plot.png')}")
    plt.close()

if __name__ == "__main__":
    results, best_model = main()
    print(f"\nTraining completed! Best model: {best_model}")