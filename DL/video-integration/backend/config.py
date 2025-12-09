"""
Configuration for ASL recognition system.
All model parameters are loaded dynamically from the ONNX model or config files.
"""

import os
import configparser

# Model configuration paths - ASL100 (active)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TGCN_ROOT = os.path.join(REPO_ROOT, 'TGCN-Training', 'code', 'TGCN', 'archived', 'asl100')
CONFIG_FILE = os.path.join(TGCN_ROOT, 'asl100.ini')
# ASL2000 paths (commented out)
# TGCN_ROOT_2000 = os.path.join(REPO_ROOT, 'TGCN-Training', 'code', 'TGCN', 'archived', 'asl2000')
# CONFIG_FILE_2000 = os.path.join(TGCN_ROOT_2000, 'asl2000.ini')

# Default values (will be overridden by model/config)
NUM_SAMPLES = 50  # Will be inferred from model
NUM_NODES = 55  # Fixed: 13 body + 21 left hand + 21 right hand
NUM_CLASSES = 100  # ASL100
# NUM_CLASSES = 2000  # ASL2000 (commented out)

# Body keypoints to exclude (matching training)
BODY_POSE_EXCLUDE = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}

# Frame capture settings
DEFAULT_FPS = 10  # Frames per second for capture
MAX_FRAMES = 200  # Maximum frames to process
TARGET_FRAME_SIZE = 256  # Target size for normalization

# Model paths - ASL100 (active)
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'asl100.onnx')
LABELS_PATH = os.path.join(REPO_ROOT, 'TGCN-Training', 'data', 'splits', 'asl100.json')
# ASL2000 paths (commented out)
# ONNX_MODEL_PATH_2000 = os.path.join(os.path.dirname(__file__), 'models', 'asl2000.onnx')
# LABELS_PATH_2000 = os.path.join(REPO_ROOT, 'TGCN-Training', 'data', 'splits', 'asl2000.json')

# TTS and Translation paths
TTS_PIPELINE_DIR = os.path.join(REPO_ROOT, 'asl-tts-pipeline', 'src')
# WORD2SENTENCE_PATH removed - using word2sentence folder directly

# Audio output directory
AUDIO_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'static', 'audio')

# Load config from INI file if available
def load_config():
    """Load configuration from INI file."""
    config = {
        'num_samples': NUM_SAMPLES,
        'num_nodes': NUM_NODES,
        'num_classes': NUM_CLASSES,
        'body_pose_exclude': BODY_POSE_EXCLUDE
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            cp = configparser.ConfigParser()
            cp.read(CONFIG_FILE)
            config['num_samples'] = int(cp.get('TRAIN', 'NUM_SAMPLES', fallback=str(NUM_SAMPLES)))
        except Exception as e:
            print(f"[CONFIG] Could not load config file: {e}")
    
    return config

