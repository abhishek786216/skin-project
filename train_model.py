import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from test_preprocessing import SimpleDataset, main as prep_main
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class SkinCancerModel(nn.Module):
    """Simple CNN model for skin cancer classification"""
    
    def __init__(self, num_classes=7):
        super(SkinCancerModel, self).__init__()
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, num_epochs=5, device='cpu'):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    model.to(device)
    
    print(f"Training on device: {device}")
    print(f"Training for {num_epochs} epochs...")
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"   Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        scheduler.step()
    
    return train_losses, val_accuracies

def evaluate_model(model, test_loader, class_names, device='cpu'):
    """Evaluate the model on test data"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    print("Evaluating model on test set...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    return accuracy, report, all_predictions, all_labels

def main():
    print("Starting Skin Cancer Classification Model Training")
    print("="*60)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Create splits (smaller datasets for faster training)
    sample_df = df.sample(n=min(2000, len(df)), random_state=42)  # Use subset for demo
    train_df, test_df = train_test_split(sample_df, test_size=0.2, stratify=sample_df['dx'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['dx'], random_state=42)
    
    print(f"Using subset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Create datasets and loaders
    train_dataset = SimpleDataset(train_df, img_dirs)
    val_dataset = SimpleDataset(val_df, img_dirs)
    test_dataset = SimpleDataset(test_df, img_dirs)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\n🧠 Initializing model...")
    model = SkinCancerModel(num_classes=len(class_names))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n🏋️ Training model...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=3,  # Small number for demo
        device=device
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, report, predictions, labels = evaluate_model(
        model, test_loader, class_names, device
    )
    
    # Save model
    model_path = os.path.join(data_dir, 'skin_cancer_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'accuracy': accuracy
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Final test accuracy: {accuracy:.4f}")
    
    # Print class-wise performance
    print("\nClass-wise Performance:")
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1 = report[str(i)]['f1-score']
            print(f"  {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    print("\nTraining completed successfully!")
    
    return model, accuracy, report

if __name__ == "__main__":
    model, accuracy, report = main()