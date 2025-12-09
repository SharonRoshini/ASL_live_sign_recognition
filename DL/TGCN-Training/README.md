# TGCN Training Module

This folder contains the training infrastructure, model checkpoints, and dataset configurations for the Temporal Graph Convolutional Network (TGCN) models used in Gesture2Globe.

## 🎯 Overview

The TGCN-Training module is the **foundation** of the Gesture2Globe ASL recognition system. It contains:

- **Trained Model Checkpoints**: Pre-trained TGCN models for ASL recognition (ASL100, ASL300, ASL1000, ASL2000)
- **Training Code**: Complete implementation for training TGCN models on ASL datasets
- **Dataset Splits**: JSON files defining train/test/validation splits for different ASL vocabularies
- **Model Configurations**: INI files specifying model architecture and training parameters

## 🔗 Integration with Gesture2Globe

### How This Module is Used

The `TGCN-Training` folder is **essential** for the Gesture2Globe application:

1. **Model Source**: Provides the PyTorch checkpoints that are converted to ONNX format for inference
2. **Model Configuration**: Contains configuration files that define model architecture (number of classes, keypoints, etc.)
3. **Label Mappings**: Contains JSON files with label mappings for each ASL vocabulary size
4. **Training Infrastructure**: Allows for retraining or fine-tuning models if needed

### Integration Flow

```
TGCN Checkpoint → ONNX Conversion → Model Loading → ASL Recognition
```

The conversion process:
1. Loads PyTorch checkpoint from `TGCN-Training/code/TGCN/archived/asl100/ckpt.pth`
2. Converts to ONNX format using `video-integration/convert_to_onnx.py`
3. Saves ONNX model to `video-integration/backend/models/asl100.onnx`
4. Backend loads ONNX model for real-time inference

### File Structure and Usage

#### Model Checkpoints
- **Location**: `code/TGCN/archived/{model_name}/ckpt.pth`
- **Models Available**:
  - `asl100/ckpt.pth` - 100 ASL signs (currently active)
  - `asl300/ckpt.pth` - 300 ASL signs
  - `asl1000/ckpt.pth` - 1000 ASL signs
  - `asl2000/ckpt.pth` - 2000 ASL signs

#### Configuration Files
- **Location**: `code/TGCN/configs/{model_name}.ini` and `code/TGCN/archived/{model_name}/{model_name}.ini`
- **Purpose**: Define model architecture parameters:
  - Number of classes
  - Input dimensions (NUM_SAMPLES, NUM_NODES)
  - Hidden layer sizes
  - Training hyperparameters

#### Dataset Splits
- **Location**: `data/splits/{model_name}.json`
- **Purpose**: Define which videos belong to train/test/validation sets
- **Used By**: Training scripts and label mapping

### Importance

- **Core Functionality**: Without these checkpoints, the application cannot recognize ASL signs
- **Model Flexibility**: Provides multiple model sizes (100, 300, 1000, 2000 signs) for different use cases
- **Extensibility**: Allows training new models or fine-tuning existing ones
- **Research Foundation**: Contains the complete training infrastructure for reproducibility

### Model Architecture

The TGCN (Temporal Graph Convolutional Network) models:
- **Input**: Temporal sequences of pose keypoints (body + hands)
- **Architecture**: Graph convolutional layers with temporal modeling
- **Output**: Probability distribution over ASL sign classes
- **Keypoints**: 55 total (13 body + 21 left hand + 21 right hand)

### Current Active Model

**ASL100** is currently active in Gesture2Globe:
- **Checkpoint**: `code/TGCN/archived/asl100/ckpt.pth`
- **Config**: `code/TGCN/archived/asl100/asl100.ini`
- **Labels**: `data/splits/asl100.json`
- **ONNX Output**: `video-integration/backend/models/asl100.onnx`

### Switching Models

To switch to a different model (e.g., ASL2000):

1. Update `video-integration/convert_to_onnx.py`:
   ```python
   # Change checkpoint path
   checkpoint_path = 'TGCN-Training/code/TGCN/archived/asl2000/ckpt.pth'
   ```

2. Update `video-integration/backend/config.py`:
   ```python
   # Change model paths and labels
   MODEL_CHECKPOINT = 'TGCN-Training/code/TGCN/archived/asl2000/ckpt.pth'
   LABELS_FILE = 'TGCN-Training/data/splits/asl2000.json'
   ```

3. Run conversion:
   ```bash
   cd video-integration
   python convert_to_onnx.py
   ```

4. Restart the backend server

### Training New Models

If you need to train a new model:

1. **Prepare Dataset**: Organize ASL videos with proper labeling
2. **Update Config**: Create/modify INI file in `code/TGCN/configs/`
3. **Update Splits**: Create JSON file in `data/splits/` with train/test splits
4. **Run Training**: Use `code/TGCN/train_tgcn.py` with appropriate parameters
5. **Convert to ONNX**: Use `video-integration/convert_to_onnx.py` to create inference model

### Key Files

- `code/TGCN/train_tgcn.py` - Main training script
- `code/TGCN/tgcn_model.py` - TGCN model definition
- `code/TGCN/sign_dataset.py` - Dataset loading and preprocessing
- `code/TGCN/configs.py` - Configuration parsing
- `code/TGCN/test_tgcn.py` - Model testing/evaluation

### Notes

- **Large Files**: Model checkpoints can be large (100MB+), ensure they're not committed to Git
- **GPU Recommended**: Training requires GPU for reasonable training times
- **Data Requirements**: Training requires substantial ASL video datasets
- **Model Conversion**: ONNX conversion is required for inference (PyTorch models are for training only)

---

**This module is essential for the Gesture2Globe application to function. Without the model checkpoints, ASL recognition would not be possible.**

