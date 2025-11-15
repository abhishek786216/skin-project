# Skin Disease Classification Project

## File Structure
```
skin/
├── data/                           # Dataset storage
│   ├── raw/HAM10000_metadata.csv  # Dataset labels
│   └── raw/HAM10000_images_part_1/ # Skin lesion images
├── models/                         # Saved model files
├── data_analysis.py               # Dataset exploration
├── test_preprocessing.py          # Data preprocessing
├── train_model.py                 # Model training
├── simple_app.py                  # Web application
├── start_app.bat                  # Windows launcher
├── requirements.txt               # Dependencies
├── USER_GUIDE.md                 # Detailed guide
└── PROJECT_ANALYSIS_REPORT.md    # Analysis report
```

## Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run the Web App (Easiest Way)
```powershell
streamlit run simple_app.py
```
- Opens web interface at http://localhost:8501
- Upload skin images for classification
- Get instant predictions with confidence scores

## Complete Workflow

### Step 1: Analyze Dataset
```powershell
python data_analysis.py
```
- Explores HAM10000 dataset statistics
- Generates class distribution plots
- Shows sample images from each category

### Step 2: Train Model (Optional - Pre-trained model included)
```powershell
python train_model.py
```
- Trains ResNet-18 on HAM10000 dataset
- Saves model to `models/skin_cancer_model.pth`
- Takes ~30 minutes on GPU, longer on CPU

### Step 3: Launch Web Application
```powershell
streamlit run simple_app.py
```
- Interactive web interface
- Upload and classify skin lesion images
- View detailed medical information

### Alternative: Use Windows Launcher
```powershell
start_app.bat
```
- Automated setup and launch
- Checks dependencies
- Starts web application

## Dataset Classes
- **MEL** - Melanoma (dangerous)
- **BCC** - Basal cell carcinoma
- **AKIEC** - Actinic keratoses
- **BKL** - Benign keratosis-like lesions
- **DF** - Dermatofibroma
- **NV** - Melanocytic nevi (moles)
- **VASC** - Vascular lesions

## Requirements
- Python 3.8+
- 4GB RAM minimum
- HAM10000 dataset in `data/raw/`
- See `requirements.txt` for full dependencies

## Results
- **Test Accuracy**: 73.5%
- **Model**: ResNet-18 with transfer learning
- **Dataset**: 10,015 dermatoscopic images

⚠️ **Medical Disclaimer**: For educational purposes only. Not for medical diagnosis. Consult medical professionals for health decisions.