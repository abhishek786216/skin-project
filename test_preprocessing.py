import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class SimpleDataset(Dataset):
    def __init__(self, df, img_dirs, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def find_image_path(self, img_id):
        for img_dir in self.img_dirs:
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                return img_path
        return None
    
    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'image_id']
        label = self.df.loc[idx, 'dx_encoded']
        
        img_path = self.find_image_path(img_id)
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_id}.jpg")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default preprocessing
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1)
        
        return image, torch.tensor(label, dtype=torch.long)

def get_train_val_test_splits():
    """Get train, val, test splits for the skin cancer dataset"""
    # Load metadata
    data_dir = r"c:\Users\abhis\OneDrive\Desktop\skin\data"
    metadata_path = os.path.join(data_dir, 'raw', 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)
    
    # Encode labels
    le = LabelEncoder()
    df['dx_encoded'] = le.fit_transform(df['dx'])
    
    # Find image directories
    img_dirs = []
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        dir_path = os.path.join(data_dir, 'raw', part)
        if os.path.exists(dir_path):
            img_dirs.append(dir_path)
    
    # Create splits
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['dx'], random_state=42)
    
    # Add image directories info
    train_df.img_dirs = img_dirs
    val_df.img_dirs = img_dirs
    test_df.img_dirs = img_dirs
    
    return train_df, val_df, test_df

def main():
    print("Starting simple preprocessing...")
    
    # Load metadata
    data_dir = r"c:\Users\abhis\OneDrive\Desktop\skin\data"
    metadata_path = os.path.join(data_dir, 'raw', 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)
    
    # Encode labels
    le = LabelEncoder()
    df['dx_encoded'] = le.fit_transform(df['dx'])
    
    print(f"Loaded {len(df)} samples")
    print(f"Classes: {list(le.classes_)}")
    
    # Find image directories
    img_dirs = []
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        dir_path = os.path.join(data_dir, 'raw', part)
        if os.path.exists(dir_path):
            img_dirs.append(dir_path)
            print(f"Found: {dir_path}")
    
    # Create splits
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['dx'], random_state=42)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = SimpleDataset(train_df, img_dirs)
    val_dataset = SimpleDataset(val_df, img_dirs)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print("Data loaders created successfully!")
    
    # Test loading
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch[0].shape}, Labels: {batch[1].shape}")
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()