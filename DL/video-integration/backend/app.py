"""
Flask backend for video-based ASL recognition.
Processes uploaded videos, extracts pose keypoints, runs inference,
and integrates TTS, translation, and sentence formation.
"""

from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
import os
import sys
import cv2
import numpy as np
import tempfile
from werkzeug.utils import secure_filename
import json
import base64
from datetime import datetime
import requests
import subprocess
from pose_extractor import PoseExtractor
from letter_translator import LetterDetectionSystem
import uuid

# Auto-install piper-tts if not available
def _ensure_piper_tts():
    """Check if piper-tts is installed, install if missing."""
    try:
        import piper.voice
        print("[SETUP] piper-tts is already installed")
        return True
    except ImportError:
        print("[SETUP] piper-tts not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "piper-tts>=1.0.0", "--quiet"])
            print("[SETUP] piper-tts installed successfully")
            # Reload the module after installation
            import importlib
            if 'piper' in sys.modules:
                importlib.reload(sys.modules['piper'])
            return True
        except subprocess.CalledProcessError as e:
            print(f"[SETUP ERROR] Failed to install piper-tts: {e}")
            print("[SETUP] Please install manually: pip install piper-tts")
            return False

# Check and install piper-tts on startup
_ensure_piper_tts()

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
except Exception:
    ort = None

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # For session management
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "https://asl-frontend-0tid.onrender.com",  # deployed frontend
                "http://localhost:3001",                   # local dev (if you use it)
                "http://localhost:5173",                   # Vite default dev port
            ]
        }
    },
)

# Configuration
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'webm', 'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global model variables
pretrained_model = None
MODEL_LOAD_INFO = {
    'loaded': False,
    'error': None,
    'details': ''
}
NUM_SAMPLES = 50  # Default for ASL100
# NUM_SAMPLES = 50  # Default for ASL2000 (commented out)
NUM_NODES = 55

# Pose extractor
pose_extractor = None


def _setup_model():
    """Load the ONNX model for inference."""
    global pretrained_model, NUM_SAMPLES
    
    if ort is None:
        print('ONNX Runtime not available')
        MODEL_LOAD_INFO['error'] = 'ONNX Runtime not installed'
        return
    
    # Get model path - ASL100 (active)
    backend_dir = os.path.dirname(__file__)
    onnx_path = os.path.join(backend_dir, 'models', 'asl100.onnx')
    # ASL2000 model path (commented out)
    # onnx_path = os.path.join(backend_dir, 'models', 'asl2000.onnx')
    
    if not os.path.exists(onnx_path):
        print(f'ONNX model not found at {onnx_path}')
        MODEL_LOAD_INFO['error'] = f'ONNX model not found. Please run convert_to_onnx.py first.'
        return
    
    try:
        providers = ['CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        pretrained_model = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
        
        # Get input shape
        input_shape = pretrained_model.get_inputs()[0].shape
        if len(input_shape) >= 3:
            feature_len = input_shape[2] if input_shape[2] > 0 else 100
            NUM_SAMPLES = feature_len // 2
        
        MODEL_LOAD_INFO['loaded'] = True
        MODEL_LOAD_INFO['details'] = f'Loaded ONNX model from {onnx_path}'
        print(f'ONNX model loaded: {input_shape}, NUM_SAMPLES={NUM_SAMPLES}')
        
    except Exception as e:
        print(f'Failed loading ONNX model: {e}')
        import traceback
        traceback.print_exc()
        MODEL_LOAD_INFO['loaded'] = False
        MODEL_LOAD_INFO['error'] = str(e)
        pretrained_model = None


def _load_labels():
    """Load ASL100 label mappings."""
    # Load ASL100 labels (active)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    splits_path = os.path.join(repo_root, 'TGCN-Training', 'data', 'splits', 'asl100.json')
    # ASL2000 labels path (commented out)
    # splits_path = os.path.join(repo_root, 'TGCN-Training', 'data', 'splits', 'asl2000.json')
    
    try:
        with open(splits_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        glosses = [entry.get('gloss', '') for entry in content]
        glosses_sorted = sorted(glosses)
        print(f'Loaded {len(glosses_sorted)} labels')
        return glosses_sorted
    except Exception as e:
        print(f'Could not load labels: {e}')
        return None


def _preprocess_keypoints(frames_data):
    """
    Preprocess keypoint frames for model input.
    Matches Sign_Dataset.read_pose_file processing exactly.
    
    Training preprocessing:
    1. Exclude body keypoints: {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}
    2. Normalize: 2 * ((x / 256.0) - 0.5) to get [-1, 1] range
    3. Result: 55 keypoints (13 body + 21 left hand + 21 right hand)
    4. Format: (55, num_samples*2) where features are [x1,y1,x2,y2,...] across time
    """
    body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}
    processed = []
    
    # Reduced logging for speed - only log if processing many frames
    if len(frames_data) > 30:
        print(f"[PREPROCESS] Processing {len(frames_data)} frames, target: {NUM_SAMPLES} frames")
    
    for frame_idx, frame_data in enumerate(frames_data):
        people = frame_data.get('people', [])
        if not people:
            # Empty frame: fill with zeros (55 keypoints * 2 = 110 values)
            processed.append(np.zeros((NUM_NODES, 2), dtype=np.float32))
            continue
        
        p = people[0]
        body = p.get('pose_keypoints_2d', [])
        left = p.get('hand_left_keypoints_2d', [])
        right = p.get('hand_right_keypoints_2d', [])
        
        # Combine: body (25) + left hand (21) + right hand (21) = 67 total landmarks
        combined = list(body) + list(left) + list(right)
        
        # Extract x, y coordinates, excluding specified body keypoints
        # Format: [x, y, confidence, x, y, confidence, ...]
        num_landmarks = len(combined) // 3 if len(combined) >= 3 else 0
        x_list = []
        y_list = []
        
        for j in range(num_landmarks):
            # Skip excluded body keypoints (indices 0-24 are body)
            if j < 25 and j in body_pose_exclude:
                continue
            
            xi = combined[j*3 + 0] if j*3 + 0 < len(combined) else 0.0
            yi = combined[j*3 + 1] if j*3 + 1 < len(combined) else 0.0
            
            # Normalize to [-1, 1] range: 2 * ((x / 256) - 0.5)
            x_norm = 2.0 * ((float(xi) / 256.0) - 0.5)
            y_norm = 2.0 * ((float(yi) / 256.0) - 0.5)
            
            x_list.append(x_norm)
            y_list.append(y_norm)
        
        # Should have exactly 55 keypoints: 13 body + 21 left + 21 right
        expected_keypoints = NUM_NODES
        if len(x_list) != expected_keypoints:
            print(f"[WARNING] Frame {frame_idx}: Expected {expected_keypoints} keypoints, got {len(x_list)}")
            # Pad or truncate to 55
            while len(x_list) < expected_keypoints:
                x_list.append(0.0)
                y_list.append(0.0)
            x_list = x_list[:expected_keypoints]
            y_list = y_list[:expected_keypoints]
        
        # Stack as (55, 2) - 55 keypoints with x,y coordinates
        xy_frame = np.stack([np.array(x_list), np.array(y_list)], axis=1).astype(np.float32)
        processed.append(xy_frame)
    
    print(f"[PREPROCESS] Extracted {len(processed)} frames with keypoints")
    
    # Pad or sample to exactly NUM_SAMPLES frames
    if len(processed) < NUM_SAMPLES:
        # Pad with last frame
        last_frame = processed[-1] if processed else np.zeros((NUM_NODES, 2), dtype=np.float32)
        num_padding = NUM_SAMPLES - len(processed)
        for _ in range(num_padding):
            processed.append(last_frame.copy())
        print(f"[PREPROCESS] Padded {num_padding} frames to reach {NUM_SAMPLES}")
    elif len(processed) > NUM_SAMPLES:
        # Uniformly sample NUM_SAMPLES frames
        original_count = len(processed)
        indices = np.linspace(0, len(processed) - 1, NUM_SAMPLES).astype(int)
        processed = [processed[i] for i in indices]
        print(f"[PREPROCESS] Sampled {NUM_SAMPLES} frames from {original_count}")
    
    # Reshape to model input format: (1, num_nodes, feature_len)
    # feature_len = num_samples * 2 (x,y coordinates across time)
    # Each node has: [x_t0, y_t0, x_t1, y_t1, ..., x_t49, y_t49]
    feature_len = NUM_SAMPLES * 2
    input_data = np.zeros((1, NUM_NODES, feature_len), dtype=np.float32)
    
    for node_idx in range(NUM_NODES):
        for t in range(NUM_SAMPLES):
            frame_xy = processed[t]  # Shape: (55, 2)
            x_val = frame_xy[node_idx, 0]
            y_val = frame_xy[node_idx, 1]
            input_data[0, node_idx, t*2 + 0] = x_val
            input_data[0, node_idx, t*2 + 1] = y_val
    
    # Reduced logging for speed
    # print(f"[PREPROCESS] Final input shape: {input_data.shape}")
    # print(f"[PREPROCESS] Input range: x=[{input_data[0, :, ::2].min():.3f}, {input_data[0, :, ::2].max():.3f}], "
    #       f"y=[{input_data[0, :, 1::2].min():.3f}, {input_data[0, :, 1::2].max():.3f}]")
    
    return input_data


def _run_tts_pipeline(text_to_speak: str, language: str = 'en'):
    """
    Run TTS using Piper Python package directly.
    Returns dict with 'audio_url' pointing to generated audio file.
    """
    if not text_to_speak or not text_to_speak.strip():
        return None
    
    try:
        from piper.voice import PiperVoice
    except ImportError:
        print('[TTS ERROR] piper-tts not installed. Run: pip install piper-tts')
        return None
    
    try:
        # Define paths
        backend_dir = os.path.dirname(__file__)
        # Go up two levels: backend -> video-integration -> root, then to asl-tts-pipeline
        model_dir = os.path.abspath(os.path.join(backend_dir, '..', '..', 'asl-tts-pipeline', 'src', 'piper_models'))
        output_dir = os.path.join(backend_dir, 'static', 'audio')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Model mapping - need both .onnx and .onnx.json files
        models = {
            "en": "en_US-lessac-medium.onnx",
            "es": "es_ES-sharvard-medium.onnx"
        }
        
        if language not in models:
            print(f"[TTS ERROR] Unsupported language: {language}")
            return None
        
        model_path = os.path.join(model_dir, models[language])
        config_path = model_path + ".json"
        
        # Debug: Print resolved paths
        print(f"[TTS] Model directory: {model_dir}")
        print(f"[TTS] Looking for model: {model_path}")
        print(f"[TTS] Looking for config: {config_path}")
        
        if not os.path.exists(model_path):
            print(f"[TTS ERROR] Model not found: {model_path}")
            print(f"[TTS ERROR] Model directory exists: {os.path.exists(model_dir)}")
            if os.path.exists(model_dir):
                print(f"[TTS ERROR] Files in model directory: {os.listdir(model_dir)}")
            return None
        
        if not os.path.exists(config_path):
            print(f"[TTS ERROR] Model config not found: {config_path}")
            return None
        
        print(f"[TTS] Found model and config files")
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{language}_piper_{timestamp}.wav"
        output_path = os.path.join(output_dir, filename)
        
        # Load voice and synthesize
        try:
            # PiperVoice.load() defaults config_path to model_path + ".json" if None
            voice = PiperVoice.load(model_path, config_path)
            import wave
            with wave.open(output_path, 'wb') as wav_file:
                voice.synthesize_wav(text_to_speak, wav_file)
        except Exception as api_error:
            print(f"[TTS ERROR] Piper synthesis error: {api_error}")
            import traceback
            traceback.print_exc()
            return None
        
        if os.path.exists(output_path):
            # Return URL path for serving
            filename_only = os.path.basename(output_path)
            print(f"[TTS] Generated: {filename_only}")
            return {'audio_url': f"http://localhost:5001/audio/{filename_only}"}
        
        return None
        
    except Exception as e:
        print(f'[TTS ERROR] Pipeline error: {e}')
        import traceback
        traceback.print_exc()
        return None


def _translate_text(text: str, target_lang: str = 'es'):
    """Translate text using Google Translate API with fallback."""
    if not text or not text.strip():
        return ""
    
    try:
        # Try Google Translate API
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': 'en',
            'tl': target_lang,
            'dt': 't',
            'q': text
        }
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result and len(result) > 0 and len(result[0]) > 0:
                translated = ''.join([item[0] for item in result[0] if item[0]])
                return translated
    except Exception as e:
        print(f"[TRANSLATE] API error: {e}, using fallback")
    
    # Fallback dictionary for common words
    fallback = {
        "hello": "hola", "goodbye": "adiós", "thank you": "gracias",
        "please": "por favor", "yes": "sí", "no": "no", "help": "ayuda",
        "bed": "cama", "kiss": "beso", "man": "hombre", "woman": "mujer"
    }
    return fallback.get(text.lower(), text)


def _form_sentence(words):
    """
    Form sentence from words using word2sentence module.
    Uses FLAN-T5 for sentence generation and GPT-2 for fluency scoring.
    If only one word is provided, returns that word directly without calling the API.
    """
    if not words:
        return ""
    
    # Filter out empty words
    words = [w for w in words if w and w.strip()]
    if not words:
        return ""
    
    # Deduplicate words while preserving order (handle repeated words)
    seen = set()
    unique_words = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)
    
    if len(unique_words) < len(words):
        print(f"[SENTENCE] Deduplicated {len(words)} words to {len(unique_words)} unique words")
    
    # If only one unique word, return it directly (no need to form a sentence)
    if len(unique_words) == 1:
        print(f"[SENTENCE] Only one word recognized: {unique_words[0]}. Using it directly as sentence.")
        return unique_words[0]
    
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    word2sentence_dir = os.path.join(repo_root, 'word2sentence')
    if word2sentence_dir not in sys.path:
        sys.path.insert(0, word2sentence_dir)
    
    try:
        from sentence_former import best_sentence  # type: ignore
        
        print(f"[SENTENCE] Forming sentence from {len(unique_words)} words: {unique_words}")
        
        # Calculate target length: 2 * num_words + 1 (as per notebook)
        target_len = 2 * len(unique_words) + 1
        n = max(6, 2 * len(unique_words))  # Number of candidates to generate
        
        # Get best sentence with verbose logging
        sentence, top_candidates = best_sentence(unique_words, n=n, target_len=target_len, verbose=True)
        
        # Check if sentence is just concatenated words (fallback behavior)
        simple_join = " ".join(unique_words).capitalize() + "."
        if sentence == simple_join or sentence.lower() == simple_join.lower():
            print(f"[SENTENCE WARNING] Sentence appears to be simple concatenation. This may indicate models failed to load.")
            print(f"[SENTENCE] Attempting to use sentence_former anyway, but this might be a fallback result.")
        
        print(f"[SENTENCE] Generated sentence: {sentence}")
        if top_candidates:
            print(f"[SENTENCE] Top candidates: {top_candidates[:3]}")
        
        return sentence
        
    except ImportError as e:
        print(f'[SENTENCE ERROR] Failed to import sentence_former: {e}')
        import traceback
        traceback.print_exc()
        print('[SENTENCE] Falling back to simple join')
        # Fallback: simple join with capitalization
        sentence = " ".join(words)
        if sentence:
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            if not sentence.endswith(('.', '!', '?')):
                sentence += "."
        return sentence
    except Exception as e:
        print(f'[SENTENCE ERROR] Error forming sentence: {e}')
        import traceback
        traceback.print_exc()
        # Fallback: simple join with capitalization
        sentence = " ".join(words)
        if sentence:
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            if not sentence.endswith(('.', '!', '?')):
                sentence += "."
        return sentence


# Create necessary directories on startup
os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'audio'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

# Initialize
_setup_model()
pose_extractor = PoseExtractor()
LABELS = _load_labels()

# Letter detection systems (session-based)
letter_detection_systems = {}  # session_id -> LetterDetectionSystem


@app.route('/model-status', methods=['GET'])
def model_status():
    """Return model load status."""
    status = {
        'onnxruntime_available': ort is not None,
        'model_loaded': MODEL_LOAD_INFO.get('loaded', False),
        'error': MODEL_LOAD_INFO.get('error'),
        'details': MODEL_LOAD_INFO.get('details'),
        'num_samples': NUM_SAMPLES,
        'num_nodes': NUM_NODES,
        'labels_count': len(LABELS) if LABELS else 0
    }
    
    # Add model input shape info
    if pretrained_model is not None:
        try:
            inputs = pretrained_model.get_inputs()
            outputs = pretrained_model.get_outputs()
            status['input_shape'] = [list(inp.shape) for inp in inputs]
            status['output_shape'] = [list(out.shape) for out in outputs]
        except:
            pass
    
    return jsonify(status)


@app.route('/process-video', methods=['POST'])
def process_video():
    """
    Main endpoint: Process uploaded video for ASL recognition.
    
    Expected: multipart/form-data with 'video' file
    Returns: JSON with recognized words, sentence, translation, and TTS audio
    """
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        print(f"Processing video: {filename}")
        
        # Extract pose keypoints from video
        print(f"[VIDEO] Extracting pose keypoints from: {filename}")
        print(f"[VIDEO] Processing entire video as one sequence (no frame batching)")
        
        # Extract ALL frames from video (no max_frames limit, no batching)
        # The video will be processed as one continuous sequence
        frames_data = pose_extractor.extract_from_video(video_path, max_frames=None, sample_rate=1)
        
        if not frames_data:
            return jsonify({"error": "No keypoints extracted from video"}), 400
        
        print(f"[VIDEO] Extracted {len(frames_data)} frames from video")
        
        # Check first frame structure
        if frames_data:
            first_frame = frames_data[0]
            people = first_frame.get('people', [])
            if people:
                p = people[0]
                body_count = len(p.get('pose_keypoints_2d', [])) // 3
                left_count = len(p.get('hand_left_keypoints_2d', [])) // 3
                right_count = len(p.get('hand_right_keypoints_2d', [])) // 3
                print(f"[VIDEO] First frame keypoints: body={body_count}, left_hand={left_count}, right_hand={right_count}")
        
        # Preprocess keypoints
        input_data = _preprocess_keypoints(frames_data)
        
        # Run inference
        if pretrained_model is None or not MODEL_LOAD_INFO['loaded']:
            return jsonify({"error": "Model not loaded"}), 503
        
        input_name = pretrained_model.get_inputs()[0].name
        output_name = pretrained_model.get_outputs()[0].name
        
        logits = pretrained_model.run([output_name], {input_name: input_data})[0]
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Get top prediction
        predicted_idx = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0, predicted_idx])
        
        # Map to label
        recognized_word = None
        if LABELS and 0 <= predicted_idx < len(LABELS):
            recognized_word = LABELS[predicted_idx]
        
        print(f"Recognized: {recognized_word} (idx={predicted_idx}, conf={confidence:.3f})")
        
        # Form sentence (if multiple words detected, for now just use single word)
        words = [recognized_word] if recognized_word else []
        sentence = _form_sentence(words) if words else recognized_word or ""
        
        # Translate to Spanish
        translated_text = _translate_text(sentence, 'es') if sentence else ""
        
        # Generate TTS for both English and Spanish
        tts_result_en = None
        tts_result_es = None
        
        if sentence:
            print(f"[TTS] Generating English TTS for: {sentence}")
            tts_result_en = _run_tts_pipeline(sentence, 'en')
            if tts_result_en:
                print(f"[TTS] English audio generated: {tts_result_en.get('audio_url')}")
            else:
                print("[TTS] Failed to generate English audio")
        
        if translated_text:
            print(f"[TTS] Generating Spanish TTS for: {translated_text}")
            tts_result_es = _run_tts_pipeline(translated_text, 'es')
            if tts_result_es:
                print(f"[TTS] Spanish audio generated: {tts_result_es.get('audio_url')}")
            else:
                print("[TTS] Failed to generate Spanish audio")
        
        # Build response
        response = {
            "word": recognized_word or str(predicted_idx),
            "confidence": confidence,
            "sentence": sentence,
            "translated_text": translated_text,
            "audio_urls": {}
        }
        
        if tts_result_en and tts_result_en.get('audio_url'):
            response['audio_urls']['en'] = tts_result_en['audio_url']
        
        if tts_result_es and tts_result_es.get('audio_url'):
            response['audio_urls']['es'] = tts_result_es['audio_url']
        
        # Cleanup
        try:
            os.remove(video_path)
        except:
            pass
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files from TTS pipeline."""
    try:
        filename = secure_filename(filename)
        backend_dir = os.path.dirname(__file__)
        
        # Check static/audio directory first (new location)
        static_audio_dir = os.path.join(backend_dir, 'static', 'audio')
        filepath = os.path.join(static_audio_dir, filename)
        
        # If not found, check old location
        if not os.path.exists(filepath):
            repo_root = os.path.abspath(os.path.join(backend_dir, '..', '..'))
            audio_dir = os.path.join(repo_root, 'asl-tts-pipeline', 'src', 'output_audio')
            filepath = os.path.join(audio_dir, filename)
        
        if not os.path.exists(filepath):
            return jsonify({"error": f"Audio file not found: {filename}"}), 404
        
        return send_file(filepath, mimetype="audio/wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process-live', methods=['POST'])
def process_live():
    """
    Process live captured frames for ASL recognition.
    
    Expected: JSON with 'frames' array of base64-encoded images
    Returns: JSON with recognized words, sentence, translation, and TTS audio
    """
    try:
        data = request.get_json()
        if not data or 'frames' not in data:
            return jsonify({"error": "No frames provided"}), 400
        
        frames_base64 = data['frames']
        if not isinstance(frames_base64, list) or len(frames_base64) == 0:
            return jsonify({"error": "Frames must be a non-empty array"}), 400
        
        # Reduced logging for speed
        # print(f"[LIVE] Processing {len(frames_base64)} frames from live capture")
        
        # Convert base64 frames to OpenCV format and extract keypoints
        # Process frames more efficiently
        frames_data = []
        for idx, frame_b64 in enumerate(frames_base64):
            try:
                # Decode base64 image
                if frame_b64.startswith('data:image'):
                    # Remove data URL prefix if present
                    frame_b64 = frame_b64.split(',')[1]
                
                image_data = base64.b64decode(frame_b64)
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    # Skip invalid frames silently for speed
                    continue
                
                # Extract keypoints from frame (this is the main bottleneck)
                keypoints = pose_extractor.extract_keypoints(frame)
                if keypoints:  # Only append if keypoints were found
                    frames_data.append(keypoints)
                
            except Exception as e:
                # Skip frames with errors to maintain speed
                continue
        
        if not frames_data:
            return jsonify({"error": "No valid frames extracted"}), 400
        
        # Reduced logging for speed - only log summary
        # print(f"[LIVE] Extracted keypoints from {len(frames_data)} frames")
        
        # Preprocess keypoints
        input_data = _preprocess_keypoints(frames_data)
        
        # Run inference
        if pretrained_model is None or not MODEL_LOAD_INFO['loaded']:
            return jsonify({"error": "Model not loaded"}), 503
        
        input_name = pretrained_model.get_inputs()[0].name
        output_name = pretrained_model.get_outputs()[0].name
        
        logits = pretrained_model.run([output_name], {input_name: input_data})[0]
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Get top prediction
        predicted_idx = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0, predicted_idx])
        
        # Map to label
        recognized_word = None
        if LABELS and 0 <= predicted_idx < len(LABELS):
            recognized_word = LABELS[predicted_idx]
        
        print(f"Recognized: {recognized_word} (idx={predicted_idx}, conf={confidence:.3f})")
        
        # Form sentence (if multiple words detected, for now just use single word)
        words = [recognized_word] if recognized_word else []
        sentence = _form_sentence(words) if words else recognized_word or ""
        
        # Translate to Spanish
        translated_text = _translate_text(sentence, 'es') if sentence else ""
        
        # Generate TTS for both English and Spanish
        tts_result_en = None
        tts_result_es = None
        
        if sentence:
            print(f"[TTS] Generating English TTS for: {sentence}")
            tts_result_en = _run_tts_pipeline(sentence, 'en')
            if tts_result_en:
                print(f"[TTS] English audio generated: {tts_result_en.get('audio_url')}")
            else:
                print("[TTS] Failed to generate English audio")
        
        if translated_text:
            print(f"[TTS] Generating Spanish TTS for: {translated_text}")
            tts_result_es = _run_tts_pipeline(translated_text, 'es')
            if tts_result_es:
                print(f"[TTS] Spanish audio generated: {tts_result_es.get('audio_url')}")
            else:
                print("[TTS] Failed to generate Spanish audio")
        
        # Build response
        response = {
            "word": recognized_word or str(predicted_idx),
            "confidence": confidence,
            "sentence": sentence,
            "translated_text": translated_text,
            "audio_urls": {}
        }
        
        if tts_result_en and tts_result_en.get('audio_url'):
            response['audio_urls']['en'] = tts_result_en['audio_url']
        
        if tts_result_es and tts_result_es.get('audio_url'):
            response['audio_urls']['es'] = tts_result_es['audio_url']
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing live frames: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/form-sentence-and-tts', methods=['POST'])
def form_sentence_and_tts():
    """
    Form sentence from words, translate, and generate TTS.
    
    Expected: JSON with 'words' array (can be strings or objects with 'word' property)
    Returns: JSON with sentence, translation, and audio URLs
    """
    try:
        data = request.get_json()
        if not data or 'words' not in data:
            return jsonify({"error": "No words provided"}), 400
        
        words = data['words']
        if not isinstance(words, list) or len(words) == 0:
            return jsonify({"error": "Words must be a non-empty array"}), 400
        
        # Extract word strings from word objects
        word_list = []
        for w in words:
            if isinstance(w, dict):
                word_list.append(w.get('word', ''))
            elif isinstance(w, str):
                word_list.append(w)
            else:
                word_list.append(str(w))
        
        # Filter out empty words
        word_list = [w for w in word_list if w and w.strip()]
        
        if not word_list:
            return jsonify({"error": "No valid words provided"}), 400
        
        print(f"[SENTENCE] Forming sentence from {len(word_list)} words: {word_list}")
        
        # Form sentence
        sentence = _form_sentence(word_list)
        print(f"[SENTENCE] Formed sentence: {sentence}")
        
        # Translate to Spanish
        translated_text = _translate_text(sentence, 'es') if sentence else ""
        print(f"[SENTENCE] Translated to Spanish: {translated_text}")
        
        # Generate TTS for both English and Spanish
        tts_result_en = None
        tts_result_es = None
        
        if sentence:
            print(f"[TTS] Generating English TTS for: {sentence}")
            tts_result_en = _run_tts_pipeline(sentence, 'en')
            if tts_result_en:
                print(f"[TTS] English audio generated: {tts_result_en.get('audio_url')}")
            else:
                print("[TTS] Failed to generate English audio")
        
        if translated_text:
            print(f"[TTS] Generating Spanish TTS for: {translated_text}")
            tts_result_es = _run_tts_pipeline(translated_text, 'es')
            if tts_result_es:
                print(f"[TTS] Spanish audio generated: {tts_result_es.get('audio_url')}")
            else:
                print("[TTS] Failed to generate Spanish audio")
        
        # Build response
        response = {
            "sentence": sentence,
            "translated_text": translated_text,
            "audio_urls": {}
        }
        
        if tts_result_en and tts_result_en.get('audio_url'):
            response['audio_urls']['en'] = tts_result_en['audio_url']
        
        if tts_result_es and tts_result_es.get('audio_url'):
            response['audio_urls']['es'] = tts_result_es['audio_url']
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error forming sentence and generating TTS: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==================== LETTER DETECTION ENDPOINTS ====================

def get_letter_detection_system(session_id: str = None) -> LetterDetectionSystem:
    """Get or create a letter detection system for a session."""
    # Use provided session_id or get from request, or create new one
    if session_id is None:
        # Try to get from request JSON or headers
        try:
            data = request.get_json(silent=True) or {}
            session_id = data.get('session_id')
        except:
            pass
        
        if not session_id:
            # Try to get from session (if available)
            try:
                session_id = session.get('letter_detection_id')
            except:
                pass
        
        if not session_id:
            # Create new session ID
            session_id = str(uuid.uuid4())
    
    if session_id not in letter_detection_systems:
        backend_dir = os.path.dirname(__file__)
        model_path = os.path.join(backend_dir, 'models', 'asl_model.joblib')
        letter_detection_systems[session_id] = LetterDetectionSystem(model_path)
    
    return letter_detection_systems[session_id]


@app.route('/api/letter-detection/process-frame', methods=['POST'])
def letter_detection_process_frame():
    """
    Process a single frame for letter detection.
    Expected: JSON with 'frame' (base64 encoded image) and optional 'session_id'
    Returns: JSON with current_letter, text, current_word, suggestions, etc.
    """
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame provided"}), 400
        
        session_id = data.get('session_id')
        frame_b64 = data['frame']
        
        # Decode base64 image
        if frame_b64.startswith('data:image'):
            frame_b64 = frame_b64.split(',')[1]
        
        image_data = base64.b64decode(frame_b64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400
        
        # Get system for this session
        system = get_letter_detection_system(session_id)
        
        # Process frame (returns result dict and drawn frame)
        # Don't draw on frame - we draw on frontend canvas for better performance
        result, drawn_frame = system.process_frame(frame, draw_on_frame=False)
        
        # Skip encoding drawn frame to reduce latency - landmarks are drawn on frontend canvas
        # Encode drawn frame as base64 (only if needed, but we're using canvas overlay now)
        # if drawn_frame is not None:
        #     _, buffer = cv2.imencode('.jpg', drawn_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        #     frame_base64 = base64.b64encode(buffer).decode('utf-8')
        #     result['drawn_frame'] = f'data:image/jpeg;base64,{frame_base64}'
        
        # Include session_id in response so client can reuse it
        result['session_id'] = session_id or list(letter_detection_systems.keys())[-1] if letter_detection_systems else None
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing letter detection frame: {error_msg}")
        import traceback
        traceback.print_exc()
        # Return more detailed error for debugging
        return jsonify({
            "error": error_msg,
            "error_type": type(e).__name__,
            "message": "Failed to process frame. Check backend logs for details."
        }), 500


@app.route('/api/letter-detection/upload', methods=['POST'])
def letter_detection_upload():
    """
    Process uploaded video for letter detection.
    Expected: multipart/form-data with 'video' file
    Returns: JSON with detected letters, text, words, suggestions
    """
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        print(f"Processing letter detection video: {filename}")
        
        # Get session_id from request if provided
        session_id = None
        try:
            form_data = request.form.to_dict()
            session_id = form_data.get('session_id')
        except:
            pass
        
        # Get system for this session
        system = get_letter_detection_system(session_id)
        system.clear()  # Clear previous state
        
        # Process video frame by frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Failed to open video"}), 400
        
        detected_letters = []
        all_results = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Process every 5th frame to avoid too much processing
            if frame_count % 5 == 0:
                result = system.process_frame(frame)
                all_results.append(result)
                if result.get('letter_committed'):
                    detected_letters.append(result.get('current_letter'))
        
        cap.release()
        
        # Get final state
        final_state = system.get_state()
        
        # Cleanup
        try:
            os.remove(video_path)
        except:
            pass
        
        current_session_id = session_id or (list(letter_detection_systems.keys())[-1] if letter_detection_systems else None)
        
        return jsonify({
            "text": final_state["text"],
            "current_word": final_state["current_word"],
            "detected_letters": detected_letters,
            "frames_processed": frame_count,
            "session_id": current_session_id
        })
        
    except Exception as e:
        print(f"Error processing letter detection video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/letter-detection/start-motion', methods=['POST'])
def letter_detection_start_motion():
    """Start motion tracking mode."""
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id')
        system = get_letter_detection_system(session_id)
        system.start_motion_mode()
        return jsonify({"status": "motion_mode_started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/letter-detection/set-motion-target', methods=['POST'])
def letter_detection_set_motion_target():
    """Set motion target (J or Z)."""
    try:
        data = request.get_json()
        target = data.get('target', '').upper()
        if target not in ['J', 'Z']:
            return jsonify({"error": "Target must be 'J' or 'Z'"}), 400
        
        session_id = data.get('session_id')
        system = get_letter_detection_system(session_id)
        system.set_motion_target(target)
        return jsonify({"status": "motion_target_set", "target": target})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/letter-detection/finish-motion', methods=['POST'])
def letter_detection_finish_motion():
    """Finish motion tracking and add letter if valid."""
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id')
        system = get_letter_detection_system(session_id)
        letter = system.finish_motion()
        state = system.get_state()
        
        return jsonify({
            "letter_added": letter is not None,
            "letter": letter,
            "text": state["text"],
            "current_word": state["current_word"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/letter-detection/add-space', methods=['POST'])
def letter_detection_add_space():
    """Add space to text."""
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id')
        system = get_letter_detection_system(session_id)
        system.add_space()
        state = system.get_state()
        return jsonify({
            "text": state["text"],
            "current_word": state["current_word"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/letter-detection/backspace', methods=['POST'])
def letter_detection_backspace():
    """Remove last character."""
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id')
        system = get_letter_detection_system(session_id)
        system.backspace()
        state = system.get_state()
        return jsonify({
            "text": state["text"],
            "current_word": state["current_word"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/letter-detection/clear', methods=['POST'])
def letter_detection_clear():
    """Clear all text."""
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id')
        system = get_letter_detection_system(session_id)
        system.clear()
        return jsonify({"status": "cleared", "text": ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/letter-detection/state', methods=['GET'])
def letter_detection_state():
    """Get current letter detection state."""
    try:
        session_id = request.args.get('session_id')
        system = get_letter_detection_system(session_id)
        state = system.get_state()
        return jsonify(state)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "Video ASL Recognition Server is running"})


if __name__ == "__main__":
    print("=" * 60)
    print("Video-based ASL Recognition Server")
    print("=" * 60)
    print("Available endpoints:")
    print("  POST /process-video - Upload and process video for ASL recognition")
    print("  POST /process-live - Process live captured frames for ASL recognition")
    print("  POST /form-sentence-and-tts - Form sentence from words, translate, and generate TTS")
    print("  GET  /model-status - Get model loading status")
    print("  GET  /health - Health check")
    print("  POST /api/letter-detection/process-frame - Process frame for letter detection")
    print("  POST /api/letter-detection/upload - Upload video for letter detection")
    print("  POST /api/letter-detection/start-motion - Start motion tracking mode")
    print("  POST /api/letter-detection/set-motion-target - Set motion target (J/Z)")
    print("  POST /api/letter-detection/finish-motion - Finish motion tracking")
    print("  POST /api/letter-detection/add-space - Add space to text")
    print("  POST /api/letter-detection/backspace - Remove last character")
    print("  POST /api/letter-detection/clear - Clear all text")
    print("  GET  /api/letter-detection/state - Get current state")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5001))
    app.run(
    host="0.0.0.0",
    port=port,
    debug=False,
    use_reloader=False,
    threaded=True
)