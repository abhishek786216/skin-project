"""
Comprehensive Analysis of HAM10000 Skin Lesion Dataset
This script provides detailed analysis of the dataset including class distribution,
demographics, and data quality assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from PIL import Image
import cv2

def load_and_analyze_data():
    """Load metadata and perform comprehensive analysis"""
    
    # Load metadata
    metadata_path = r"c:\Users\abhis\OneDrive\Desktop\skin\data\raw\HAM10000_metadata.csv"
    df = pd.read_csv(metadata_path)
    
    print("="*60)
    print("HAM10000 DATASET COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Basic dataset information
    print(f"\nDATASET OVERVIEW")
    print(f"Dataset shape: {df.shape}")
    print(f"Total images: {len(df)}")
    print(f"Total unique lesions: {df['lesion_id'].nunique()}")
    print(f"Columns: {list(df.columns)}")
    
    # Data types and missing values
    print(f"\nDATA QUALITY ASSESSMENT")
    print("Data types:")
    print(df.dtypes)
    print(f"\nMissing values per column:")
    missing_data = df.isnull().sum()
    for col, missing in missing_data.items():
        percentage = (missing / len(df)) * 100
        print(f"  {col}: {missing} ({percentage:.2f}%)")
    
    # Class distribution
    print(f"\nDIAGNOSTIC CLASS DISTRIBUTION")
    class_counts = df['dx'].value_counts()
    class_info = {
        'akiec': 'Actinic keratoses and intraepithelial carcinoma',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis-like lesions',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi (moles)',
        'vasc': 'Vascular lesions'
    }
    
    for class_code, count in class_counts.items():
        percentage = (count / len(df)) * 100
        risk_level = "HIGH RISK" if class_code in ['mel', 'bcc', 'akiec'] else "LOW/MODERATE RISK"
        print(f"  {class_code}: {count} images ({percentage:.1f}%) - {class_info.get(class_code, 'Unknown')} [{risk_level}]")
    
    # Diagnosis type distribution
    print(f"\nDIAGNOSIS TYPE DISTRIBUTION")
    dx_type_counts = df['dx_type'].value_counts()
    dx_type_info = {
        'histo': 'Histopathology (definitive diagnosis)',
        'follow_up': 'Follow-up examination',
        'consensus': 'Expert consensus',
        'confocal': 'Confocal microscopy'
    }
    
    for dx_type, count in dx_type_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {dx_type}: {count} ({percentage:.1f}%) - {dx_type_info.get(dx_type, 'Unknown')}")
    
    # Age distribution
    print(f"\nAGE DEMOGRAPHICS")
    age_stats = df['age'].describe()
    print(f"Age statistics:")
    for stat, value in age_stats.items():
        print(f"  {stat}: {value:.1f}")
    
    # Gender distribution
    print(f"\nGENDER DISTRIBUTION")
    gender_counts = df['sex'].value_counts()
    for gender, count in gender_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {gender}: {count} ({percentage:.1f}%)")
    
    # Body localization
    print(f"\nBODY LOCALIZATION DISTRIBUTION")
    loc_counts = df['localization'].value_counts()
    for location, count in loc_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {location}: {count} ({percentage:.1f}%)")
    
    return df, class_counts, age_stats

def analyze_image_data():
    """Analyze image data characteristics"""
    print(f"\nIMAGE DATA ANALYSIS")
    
    # Count images in directories
    part1_dir = r"c:\Users\abhis\OneDrive\Desktop\skin\data\raw\HAM10000_images_part_1"
    part2_dir = r"c:\Users\abhis\OneDrive\Desktop\skin\data\raw\HAM10000_images_part_2"
    
    part1_count = len([f for f in os.listdir(part1_dir) if f.endswith('.jpg')])
    part2_count = len([f for f in os.listdir(part2_dir) if f.endswith('.jpg')])
    
    print(f"Images in part 1: {part1_count}")
    print(f"Images in part 2: {part2_count}")
    print(f"Total image files: {part1_count + part2_count}")
    
    # Sample a few images to get dimensions
    print(f"\nIMAGE CHARACTERISTICS (Sample Analysis)")
    sample_images = []
    
    # Get sample from part 1
    part1_files = [f for f in os.listdir(part1_dir) if f.endswith('.jpg')][:5]
    for img_file in part1_files:
        try:
            img_path = os.path.join(part1_dir, img_file)
            img = Image.open(img_path)
            sample_images.append({
                'file': img_file,
                'size': img.size,
                'mode': img.mode,
                'format': img.format
            })
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
    
    if sample_images:
        print(f"Sample image analysis:")
        for img_info in sample_images:
            print(f"  {img_info['file']}: {img_info['size']} pixels, Mode: {img_info['mode']}")

def create_visualizations(df, class_counts):
    """Create data visualizations"""
    print(f"\nCREATING DATA VISUALIZATIONS")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Class distribution
    axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Diagnostic Class Distribution')
    
    # Age distribution
    df['age'].dropna().hist(bins=20, ax=axes[0, 1], alpha=0.7)
    axes[0, 1].set_title('Age Distribution')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Frequency')
    
    # Gender distribution
    gender_counts = df['sex'].value_counts()
    axes[1, 0].bar(gender_counts.index, gender_counts.values)
    axes[1, 0].set_title('Gender Distribution')
    axes[1, 0].set_ylabel('Count')
    
    # Localization (top 10)
    loc_counts = df['localization'].value_counts().head(10)
    axes[1, 1].barh(loc_counts.index, loc_counts.values)
    axes[1, 1].set_title('Top 10 Body Locations')
    axes[1, 1].set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig(r'c:\Users\abhis\OneDrive\Desktop\skin\data_analysis_plots.png', dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to data_analysis_plots.png")
    plt.close()  # Close the plot instead of showing it

def analyze_class_imbalance(df):
    """Analyze class imbalance and suggest strategies"""
    print(f"\nCLASS IMBALANCE ANALYSIS")
    
    class_counts = df['dx'].value_counts()
    total_samples = len(df)
    
    print(f"Class imbalance metrics:")
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    imbalance_ratio = max_class_count / min_class_count
    
    print(f"  Largest class: {class_counts.index[0]} ({class_counts.iloc[0]} samples)")
    print(f"  Smallest class: {class_counts.index[-1]} ({class_counts.iloc[-1]} samples)")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    # Calculate weights for balanced training
    print(f"\n🏋️ SUGGESTED CLASS WEIGHTS FOR TRAINING:")
    class_weights = {}
    for class_name, count in class_counts.items():
        weight = total_samples / (len(class_counts) * count)
        class_weights[class_name] = weight
        print(f"  {class_name}: {weight:.3f}")
    
    return class_weights

def medical_insights(df):
    """Provide medical insights about the dataset"""
    print(f"\nMEDICAL INSIGHTS AND IMPLICATIONS")
    
    # High-risk vs low-risk distribution
    high_risk_classes = ['mel', 'bcc', 'akiec']
    low_risk_classes = ['nv', 'bkl', 'df', 'vasc']
    
    high_risk_count = df[df['dx'].isin(high_risk_classes)].shape[0]
    low_risk_count = df[df['dx'].isin(low_risk_classes)].shape[0]
    
    print(f"Risk Assessment:")
    print(f"  High-risk lesions (melanoma, BCC, akiec): {high_risk_count} ({high_risk_count/len(df)*100:.1f}%)")
    print(f"  Low-risk lesions (nevi, benign): {low_risk_count} ({low_risk_count/len(df)*100:.1f}%)")
    
    # Age analysis for high-risk lesions
    high_risk_df = df[df['dx'].isin(high_risk_classes)]
    if not high_risk_df.empty:
        mean_age_high_risk = high_risk_df['age'].mean()
        print(f"\nAge patterns:")
        print(f"  Average age for high-risk lesions: {mean_age_high_risk:.1f} years")
        
        # Gender analysis for melanoma specifically
        melanoma_df = df[df['dx'] == 'mel']
        if not melanoma_df.empty:
            melanoma_gender = melanoma_df['sex'].value_counts()
            print(f"\nMelanoma gender distribution:")
            for gender, count in melanoma_gender.items():
                percentage = (count / len(melanoma_df)) * 100
                print(f"  {gender}: {count} ({percentage:.1f}%)")

def main():
    """Main analysis function"""
    try:
        # Load and analyze data
        df, class_counts, age_stats = load_and_analyze_data()
        
        # Analyze image data
        analyze_image_data()
        
        # Create visualizations
        create_visualizations(df, class_counts)
        
        # Analyze class imbalance
        class_weights = analyze_class_imbalance(df)
        
        # Medical insights
        medical_insights(df)
        
        print(f"\nANALYSIS COMPLETE!")
        print(f"Key Findings:")
        print(f"   - Dataset contains {len(df)} dermatoscopic images of skin lesions")
        print(f"   - 7 diagnostic categories with significant class imbalance")
        print(f"   - Melanocytic nevi (nv) is the most common class ({class_counts['nv']} samples)")
        print(f"   - High-risk lesions represent important diagnostic targets")
        print(f"   - Age range from {df['age'].min():.0f} to {df['age'].max():.0f} years")
        print(f"   - Balanced gender representation in the dataset")
        
        return df, class_weights
        
    except Exception as e:
        print(f"ERROR: Error during analysis: {e}")
        return None, None

if __name__ == "__main__":
    main()