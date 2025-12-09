# Gesture2Globe

A comprehensive American Sign Language (ASL) recognition and translation system that processes sign language gestures from video or live camera feeds, recognizes words, forms sentences, translates to multiple languages, and generates audio output.

## 🎯 Overview

Gesture2Globe is a complete end-to-end ASL recognition system that:

- **Recognizes ASL Signs**: Processes video or live camera feeds to identify American Sign Language gestures
- **Forms Sentences**: Converts recognized words into natural sentences
- **Translates**: Automatically translates to Spanish (and other languages)
- **Generates Speech**: Creates audio output in both English and Spanish using text-to-speech
- **Real-time Processing**: Supports both video upload and live camera capture modes
- **Letter Detection**: Real-time letter-by-letter ASL recognition

## 🏗️ System Architecture

The application consists of three main components:

### 1. Video Integration Module (`video-integration/`)
The core ASL recognition system that:
- Extracts pose keypoints from video frames using MediaPipe
- Runs inference using trained TGCN (Temporal Graph Convolutional Network) models
- Recognizes ASL words from video or live camera feeds
- Integrates with sentence formation and translation services

**Key Features:**
- Video upload processing
- Live camera capture and recognition
- Real-time letter detection
- Pose keypoint extraction (MediaPipe → OpenPose BODY_25 format)
- ONNX model inference (ASL100 - 100 sign classes)

### 2. ASL TTS Pipeline (`asl-tts-pipeline/`)
Text-to-speech and translation services that:
- Converts recognized text to speech using PiperTTS
- Translates text to Spanish using Google Translate API
- Generates audio files in multiple languages
- Provides high-quality neural TTS output

**Key Features:**
- English and Spanish TTS generation
- Automatic translation via Google Translate API
- Multiple voice model support
- Comprehensive logging and audio file management

### 3. Word-to-Sentence Module (`word2sentence/`)
Sentence formation system that:
- Converts recognized ASL words into natural sentences
- Uses FLAN-T5 for sentence generation
- Uses GPT-2 for fluency scoring
- Handles single-word cases (returns word directly without API call)

## 📁 Project Structure

```
CS59000-Course-Project/
├── README.md                          # This file (overview)
├── requirements.txt                   # Consolidated Python dependencies
├── .gitignore                         # Git ignore patterns
│
├── video-integration/                 # Main ASL recognition system
│   ├── backend/
│   │   ├── app.py                     # Flask backend server
│   │   ├── pose_extractor.py          # MediaPipe pose extraction
│   │   ├── letter_translator.py       # Letter detection system
│   │   ├── config.py                  # Configuration settings
│   │   └── models/                    # ONNX model directory
│   │       ├── asl100.onnx            # ASL100 model (100 classes)
│   │       └── asl2000.onnx           # ASL2000 model (2000 classes)
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── App.jsx                # Main React component
│   │   │   ├── LetterDetection.jsx    # Letter detection component
│   │   │   └── App.css                # Styles
│   │   ├── package.json               # Frontend dependencies
│   │   └── vite.config.js             # Vite configuration
│   └── convert_to_onnx.py            # PyTorch to ONNX conversion
│
├── asl-tts-pipeline/                   # TTS and translation services
│   ├── README.md                      # TTS pipeline documentation
│   ├── src/
│   │   ├── asl_piper_pipeline.py     # Main TTS orchestrator
│   │   ├── piper_tts_pipeline.py     # PiperTTS synthesis engine
│   │   ├── translator.py             # Google Translate integration
│   │   ├── piper_models/             # Voice models
│   │   ├── output_audio/             # Generated audio files
│   │   └── logs/                      # Pipeline logs
│   └── piper/                         # PiperTTS engine
│
├── word2sentence/                     # Sentence formation module
│   ├── sentence_former.py            # Sentence generation logic
│   └── requirements.txt              # Module dependencies
│
└── TGCN-Training/                     # Model training data
    ├── code/TGCN/                     # Training code
    └── data/splits/                   # Dataset splits
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed
- **Node.js 16+** and **npm** installed
- **Git** installed
- Access to the project directories:
  - `TGCN-Training/` - Contains model checkpoints
  - `asl-tts-pipeline/` - For TTS and translation (optional)

### Installation

1. **Install Python Dependencies** (from project root):
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Frontend Dependencies**:
   ```bash
   cd video-integration/frontend
   npm install
   ```

3. **Convert Model to ONNX** (first time only):
   ```bash
   cd video-integration
   python convert_to_onnx.py
   ```

### Running the Application

You need **two terminal windows**:

#### Terminal 1: Start Backend Server
```bash
cd video-integration/backend
python app.py
```

The backend will start on `http://localhost:5001`

#### Terminal 2: Start Frontend Server
```bash
cd video-integration/frontend
npm run dev
```

The frontend will start on `http://localhost:3001`

### Using the Application

1. Open `http://localhost:3001` in your browser
2. Verify model status shows "Ready to recognize your signs"
3. Choose your mode:
   - **Upload Video**: Upload and process a video file
   - **Live Capture**: Real-time recognition from webcam
   - **Letter Detection**: Real-time letter-by-letter recognition
4. View results: recognized words, formed sentences, translations, and audio

## 🔄 Processing Pipeline

### Video/Live Capture Flow

1. **Input**: User uploads video or captures live frames
2. **Pose Extraction**: MediaPipe extracts keypoints from each frame
3. **Keypoint Mapping**: MediaPipe keypoints mapped to OpenPose BODY_25 format
4. **Preprocessing**: Keypoints normalized and formatted for model input
5. **Model Inference**: ONNX model predicts sign class
6. **Label Mapping**: Predicted index mapped to ASL gloss
7. **Sentence Formation**: 
   - If single word: Returns word directly
   - If multiple words: Uses word2sentence module to form natural sentence
8. **Translation**: Sentence translated to Spanish via Google Translate API
9. **TTS Generation**: Audio generated for both English and Spanish
10. **Response**: Results returned to frontend with audio URLs

### Letter Detection Flow

1. **Frame Capture**: Camera captures frames at ~15 FPS
2. **Hand Detection**: MediaPipe Hands detects hand landmarks
3. **Letter Recognition**: Letter detection model processes hand landmarks
4. **Stability Check**: Letter must be stable for 12 frames before committing
5. **Display**: Current prediction and stable letter shown in real-time

## 🎯 Features

### Video Upload Mode
- Upload video files (mp4, webm, avi, mov, mkv)
- Process entire video as one sequence
- Extract pose keypoints from all frames
- Recognize ASL signs and form sentences

### Live Capture Mode
- Real-time webcam feed
- Hand detection before capturing frames
- Batch frame processing
- Two-column layout: camera on left, results on right

### Letter Detection Mode
- Real-time letter-by-letter recognition
- Hand skeleton visualization
- Current prediction and stable letter display
- Real-time hand landmark tracking

### Translation & TTS
- Automatic translation to Spanish
- High-quality TTS in English and Spanish
- Audio playback in browser
- Support for multiple languages

## 📊 Model Information

### ASL100 Model (Active)
- **Classes**: 100 ASL signs
- **Input**: 50 frames × 100 features (55 keypoints × 2 coordinates)
- **Architecture**: TGCN (Temporal Graph Convolutional Network)
- **Keypoints**: 55 total (13 body + 21 left hand + 21 right hand)
- **Checkpoint**: `TGCN-Training/code/TGCN/archived/asl100/ckpt.pth`
- **ONNX Model**: `video-integration/backend/models/asl100.onnx`

### ASL2000 Model (Available)
- **Classes**: 2000 ASL signs
- Same architecture as ASL100
- Can be activated by updating configuration files

## 🛠️ Configuration

### Backend Configuration
- **Port**: 5001 (configurable in `backend/app.py`)
- **Model**: ASL100 (configurable in `backend/config.py`)
- **Max Frames**: 200 per video (configurable)

### Frontend Configuration
- **Port**: 3001 (configurable in `frontend/vite.config.js`)
- **API Base**: `http://localhost:5001` (configurable in `App.jsx`)
- **Frame Rate**: ~15 FPS for letter detection

## 🐛 Troubleshooting

### Model Not Loading
- Verify ONNX model exists: `backend/models/asl100.onnx`
- Check backend logs for errors
- Ensure ONNX Runtime is installed: `pip install onnxruntime`

### Camera Not Working
- Grant camera permissions in browser
- Check browser console (F12) for errors
- Try different browser (Chrome, Firefox, Edge)

### TTS/Translation Not Working
- This is optional - recognition still works without it
- Ensure `asl-tts-pipeline` folder exists
- Check internet connection for Google Translate API
- Verify espeak-ng is installed (for TTS)

### Port Conflicts
- Backend: Change port in `backend/app.py`
- Frontend: Change port in `frontend/vite.config.js`
- Update API_BASE in `frontend/src/App.jsx` if backend port changes

## 📝 API Endpoints

### Backend Endpoints

- `POST /process-video` - Process uploaded video
- `POST /process-live` - Process live captured frames
- `POST /form-sentence-and-tts` - Form sentence and generate TTS
- `POST /api/letter-detection/process-frame` - Process letter detection frame
- `GET /model-status` - Get model loading status
- `GET /health` - Health check
- `GET /audio/<filename>` - Serve generated audio files

## 🔧 Dependencies

### Backend
- Flask 3.0.0
- ONNX Runtime 1.16.3
- MediaPipe 0.10.11
- OpenCV 4.8.1.78
- NumPy 1.24.3
- Requests 2.31.0

### Frontend
- React 18.2.0
- Vite 5.0.0
- Axios 1.6.0
- MediaPipe Hands (for hand detection)

### External Modules
- `asl-tts-pipeline` - TTS and translation
- `word2sentence` - Sentence formation

## 📚 Documentation

- **This README**: Overall application overview
- **`asl-tts-pipeline/README.md`**: Detailed TTS pipeline documentation

## 🎓 Key Design Decisions

1. **Single Word Handling**: If only one word is recognized, it's used directly as the sentence (no API call needed)
2. **Two-Column Layout**: Camera/controls on left, results on right for better UX
3. **Consolidated Dependencies**: Single `requirements.txt` at project root
4. **ONNX Models**: Converted from PyTorch for faster inference
5. **MediaPipe Integration**: Uses MediaPipe for pose and hand detection
6. **Responsive Design**: Professional light/dark mode UI

## 📄 License

This project is part of the CS59000 Course Project.

---

**Ready to get started?** Follow the Quick Start guide above to begin recognizing ASL signs! 🚀

