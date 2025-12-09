# ASL TTS Pipeline

A comprehensive Text-to-Speech pipeline that converts ASL (American Sign Language) text to both English and Spanish audio using PiperTTS. This pipeline provides high-quality speech synthesis with **automatic Google Translate integration** for professional-quality translations.

## 🎯 Overview

This project implements a complete ASL-to-Speech pipeline that:
- Takes ASL text input (in English)
- Generates high-quality English speech using PiperTTS
- **Automatically translates** the text to Spanish using Google Translate API
- Generates Spanish speech using PiperTTS
- Provides comprehensive logging and audio file management

## 🔗 Integration with Gesture2Globe

### How This Module is Used

The `asl-tts-pipeline` module is a **critical component** of the Gesture2Globe application. It is automatically invoked by the backend server (`video-integration/backend/app.py`) to:

1. **Generate Audio Output**: After ASL recognition identifies words and forms sentences, this pipeline converts the text to speech
2. **Enable Multilingual Support**: Automatically translates recognized ASL text to Spanish and generates audio in both languages
3. **Provide Audio Playback**: Creates audio files that are served to the frontend for user playback

### Integration Flow

```
ASL Recognition → Sentence Formation → TTS Pipeline → Audio Files → Frontend Playback
```

The backend calls the TTS pipeline functions:
- `_run_tts_pipeline()` - Generates TTS audio for a given text and language
- `_translate_text()` - Translates text using Google Translate API
- Functions from `asl_piper_pipeline.py` and `piper_tts_pipeline.py` are imported and used directly

### Importance

- **User Experience**: Provides audio feedback, making the application accessible to users who prefer audio output
- **Multilingual Support**: Enables the application to serve Spanish-speaking users
- **Complete Pipeline**: Transforms the application from text-only to a full multimedia experience
- **Accessibility**: Makes ASL recognition results accessible through audio, not just visual text

### Module Location

The backend server imports functions from this module:
```python
# In video-integration/backend/app.py
sys.path.insert(0, asl_tts_pipeline_dir)
from asl_piper_pipeline import ASLPiperPipeline
from piper_tts_pipeline import PiperSpeechSynthesizer
from translator import TextTranslator
```

### Optional but Recommended

While the TTS pipeline is **optional** (the application will work without it), it significantly enhances the user experience by providing:
- Audio playback of recognized signs
- Multilingual support
- Professional-quality speech synthesis

## 📁 Project Structure

```
asl-tts-pipeline/
├── README.md                     # This comprehensive documentation
├── piper/                        # PiperTTS engine and build files
│   ├── build/                    # Compiled Piper executable
│   ├── src/                      # Piper source code
│   ├── etc/                      # Test files and resources
│   ├── notebooks/                # Jupyter notebooks and examples
│   └── CMakeLists.txt            # Build configuration
└── src/                          # ASL TTS Pipeline source code
    ├── __init__.py               # Package initialization
    ├── asl_piper_pipeline.py     # Main ASL pipeline orchestrator
    ├── piper_tts_pipeline.py     # PiperTTS synthesis engine
    ├── translator.py             # Google Translate integration
    ├── logs/                     # Pipeline execution logs
    │   ├── piper_log.txt         # Main pipeline logs
    │   └── translation_log.txt   # Translation-specific logs
    ├── output_audio/             # Generated audio files
    │   ├── en_piper_*.wav        # English speech files
    │   └── es_piper_*.wav        # Spanish speech files
    └── piper_models/             # Voice models for TTS
        ├── en_US-lessac-medium.onnx
        ├── en_US-lessac-medium.onnx.json
        ├── es_ES-sharvard-medium.onnx
        └── es_ES-sharvard-medium.onnx.json
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.7+** installed on your system
2. **espeak-ng** installed (for text-to-phoneme conversion)
3. **Piper TTS binary** (included in the project - located in `piper/build/`)
4. **Internet connection** (for Google Translate API)
5. **requests library** (for API calls)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd CS59000-Course-Project/asl-tts-pipeline
   ```

2. **Install espeak-ng (macOS):**
   ```bash
   brew install espeak-ng
   ```

3. **Install espeak-ng (Ubuntu/Debian):**
   ```bash
   sudo apt-get install espeak-ng
   ```

4. **Install espeak-ng (Windows):**
   - Download from: https://github.com/espeak-ng/espeak-ng/releases
   - Or use: `choco install espeak-ng`

5. **Install Python dependencies:**
   ```bash
   pip install requests
   ```

## 🎯 How to Run

### Option 1: Run the Full Pipeline (Recommended)

```bash
cd src
python asl_piper_pipeline.py
```

This will:
- Convert "I need help with my homework" to English speech
- **Automatically translate** it to Spanish using Google Translate API
- Convert the Spanish text to Spanish speech
- Save both audio files to `src/output_audio/`
- Log the process to `src/logs/piper_log.txt`

### Option 2: Run with Custom Text

```bash
cd src
python -c "
from asl_piper_pipeline import ASLPiperPipeline
pipeline = ASLPiperPipeline()
pipeline.run('I am learning Spanish and it is very interesting')
"
```

**✨ Try any English sentence!** The system automatically translates it to Spanish:
- "The weather is beautiful today and I want to go outside"
- "Can you please help me understand this difficult math problem?"
- "I would like to eat an apple everyday"
- "I do not go to school regularly"

### Option 3: Use Individual TTS Component

```bash
cd src
python -c "
from piper_tts_pipeline import PiperSpeechSynthesizer
tts = PiperSpeechSynthesizer()

# Generate English speech
en_audio = tts.synthesize('Hello world!', 'en', 'hello.wav')
print(f'English audio: {en_audio}')

# Generate Spanish speech
es_audio = tts.synthesize('Hola mundo!', 'es', 'hola.wav')
print(f'Spanish audio: {es_audio}')
"
```

### Option 4: Test Translation Only

```bash
cd src
python -c "
from translator import TextTranslator
translator = TextTranslator()
result = translator.translate('Hello, how are you today?', 'es')
print(f'Translation: {result}')
"
```

### Option 5: Create a Custom Script

Create a file called `my_script.py` in the `src` directory:

```python
#!/usr/bin/env python3
from asl_piper_pipeline import ASLPiperPipeline
from piper_tts_pipeline import PiperSpeechSynthesizer
from translator import TextTranslator

def main():
    # Example 1: Full pipeline
    print("Running full ASL pipeline...")
    pipeline = ASLPiperPipeline()
    pipeline.run("Hello, this is a test!")
    
    # Example 2: Individual TTS
    print("\nRunning individual TTS...")
    tts = PiperSpeechSynthesizer()
    result = tts.synthesize("Testing individual TTS", "en", "test.wav")
    print(f"Generated: {result}")
    
    # Example 3: Translation only
    print("\nTesting translation...")
    translator = TextTranslator()
    spanish = translator.translate("Good morning, have a great day!", "es")
    print(f"Spanish: {spanish}")

if __name__ == "__main__":
    main()
```

Then run it:
```bash
cd src
python my_script.py
```

## 🎧 Output Files

The pipeline generates audio files in `src/output_audio/`:

- **English audio**: `en_piper_YYYYMMDD_HHMMSS.wav`
- **Spanish audio**: `es_piper_YYYYMMDD_HHMMSS.wav`

Example output:
```
============================================================
🎯 ASL Text: I need help with my homework
============================================================

⏳ Step 1: Generating English Speech
----------------------------------------
✅ Generated: en_piper_20251028_203500.wav (45.2 KB)

⏳ Step 2: Translating to Spanish
----------------------------------------
🌍 Translated: Necesito ayuda con mi tarea

⏳ Step 3: Generating Spanish Speech
----------------------------------------
✅ Generated: es_piper_20251028_203500.wav (42.8 KB)

⏳ Step 4: Logging Results
----------------------------------------
📝 Logged pipeline output to /path/to/src/logs/piper_log.txt

============================================================
📊 PIPELINE SUMMARY
============================================================
✅ ASL-to-Speech completed via PiperTTS!
🎧 English: /path/to/src/output_audio/en_piper_20251028_203500.wav
🎧 Spanish: /path/to/src/output_audio/es_piper_20251028_203500.wav
============================================================
```

## 📝 Logging

All pipeline runs are logged to `src/logs/piper_log.txt` with:
- Timestamp
- Original English text
- Spanish translation
- Paths to generated audio files

Example log entry:
```
=== 2025-10-28 20:35:00 ===
English Text: I need help with my homework
Spanish Translation: Necesito ayuda con mi tarea
English Audio: /path/to/src/output_audio/en_piper_20251028_203500.wav
Spanish Audio: /path/to/src/output_audio/es_piper_20251028_203500.wav
```

## 🔧 Configuration

### Available Voice Models

The pipeline supports:
- **English**: `en_US-lessac-medium.onnx` (high quality, natural pronunciation)
- **Spanish**: `es_ES-sharvard-medium.onnx` (medium quality, clear Spanish accent)

### Adding New Languages

1. Download voice models from: https://huggingface.co/rhasspy/piper-voices
2. Add the `.onnx` and `.onnx.json` files to `src/piper_models/`
3. Update the `models` dictionary in `src/piper_tts_pipeline.py`:

```python
self.models = {
    "en": "en_US-lessac-medium.onnx",
    "es": "es_ES-sharvard-medium.onnx",
    "fr": "fr_FR-siwis-medium.onnx",  # Add French
    "de": "de_DE-thorsten-medium.onnx"  # Add German
}
```

### Automatic Translation System

The system uses **Google Translate API** for automatic translation:

- ✅ **No manual configuration needed** - translates any English text automatically
- ✅ **Professional quality** - Google Translate provides accurate, grammatically correct translations
- ✅ **Multi-language support** - Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese
- ✅ **Fallback system** - works even without internet connection using local dictionary
- ✅ **Real-time translation** - instant translation without delays
- ✅ **Error handling** - graceful fallback if API is unavailable

### Supported Languages

The Google Translate integration supports:
- **Spanish** (es) - Primary target language
- **French** (fr)
- **German** (de)
- **Italian** (it)
- **Portuguese** (pt)
- **Russian** (ru)
- **Japanese** (ja)
- **Korean** (ko)
- **Chinese** (zh)

## 🏗️ Architecture

### Core Components

1. **ASLPiperPipeline** (`asl_piper_pipeline.py`)
   - Main orchestrator class
   - Coordinates the entire pipeline workflow
   - Handles logging and file management

2. **PiperSpeechSynthesizer** (`piper_tts_pipeline.py`)
   - Text-to-speech synthesis engine
   - Auto-detects Piper executable
   - Supports multiple voice models
   - Handles audio file generation

3. **TextTranslator** (`translator.py`)
   - Google Translate API integration
   - Fallback dictionary system
   - Multi-language support
   - Error handling and recovery

### Pipeline Flow

```
ASL Text Input
     ↓
English TTS (PiperTTS)
     ↓
Google Translate API
     ↓
Spanish TTS (PiperTTS)
     ↓
Audio Files + Logging
```

## 🐛 Troubleshooting

### Common Issues

1. **"No such file or directory: espeak-ng-data"**
   ```bash
   # macOS
   brew install espeak-ng
   
   # Ubuntu/Debian
   sudo apt-get install espeak-ng
   
   # Windows
   choco install espeak-ng
   ```

2. **"Piper executable not found"**
   - The system auto-detects Piper in the included `piper/build/` directory
   - If not found, ensure the piper folder is properly included in the project
   - Alternative: install system-wide Piper:
     ```bash
     # macOS
     brew install piper-tts
     
     # Ubuntu/Debian
     sudo apt install piper-tts
     ```

3. **"Piper model not found"**
   - Ensure voice models are in `src/piper_models/`
   - Check file names match the configuration
   - Download models from: https://huggingface.co/rhasspy/piper-voices

4. **"Permission denied"**
   ```bash
   chmod +x /path/to/piper/build/piper
   ```

5. **Import errors**
   - Make sure you're running from the `src` directory
   - Check that all Python files are in the correct location
   - Verify Python path includes the project directory

6. **Translation errors**
   - Ensure you have internet connection for Google Translate API
   - Install requests library: `pip install requests`
   - Check firewall settings if API calls fail
   - Test translation separately:
     ```bash
     python -c "from translator import TextTranslator; print(TextTranslator().translate('hello', 'es'))"
     ```

7. **Audio generation fails**
   - Check espeak-ng installation
   - Verify Piper models are complete (.onnx and .json files)
   - Ensure output directory has write permissions

### Debug Mode

Add debug output by modifying the pipeline:

```python
# In src/asl_piper_pipeline.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable:
export PYTHONPATH=/path/to/asl-tts-pipeline/src
```

### Testing Individual Components

Test each component separately:

```bash
# Test TTS only
cd src
python -c "
from piper_tts_pipeline import PiperSpeechSynthesizer
tts = PiperSpeechSynthesizer()
print('TTS Test:', tts.synthesize('test', 'en'))
"

# Test translation only
python -c "
from translator import TextTranslator
t = TextTranslator()
print('Translation Test:', t.translate('hello', 'es'))
"
```

## 📊 Performance

- **English TTS**: ~0.05 seconds real-time factor (20x faster than real-time)
- **Spanish TTS**: ~0.04 seconds real-time factor (25x faster than real-time)
- **Translation**: ~0.1-0.5 seconds (Google Translate API)
- **Audio Quality**: High-quality neural TTS with natural pronunciation
- **File Sizes**: ~50-150KB per 2-5 second audio clip
- **Translation Quality**: Professional-grade (Google Translate)
- **Memory Usage**: Low memory footprint (~50-100MB)
- **CPU Usage**: Efficient processing with PiperTTS

## 🔄 Updates and Maintenance

### Updating Voice Models

1. Download new models from https://huggingface.co/rhasspy/piper-voices
2. Replace files in `src/piper_models/`
3. Update model names in `piper_tts_pipeline.py` if needed
4. Test with sample text

### Updating Dependencies

```bash
# Update Python packages
pip install --upgrade requests

# Update espeak-ng
brew upgrade espeak-ng  # macOS
sudo apt upgrade espeak-ng  # Ubuntu/Debian
```

### Backup and Recovery

- **Audio files**: Backup `src/output_audio/` directory
- **Logs**: Backup `src/logs/` directory
- **Models**: Backup `src/piper_models/` directory
- **Configuration**: Backup Python source files

## 📄 License

This project is part of the CS59000 Course Project.

## 🆘 Support

If you encounter issues:

1. **Check the troubleshooting section above**
2. **Review the logs** in `src/logs/piper_log.txt`
3. **Verify dependencies** are installed:
   ```bash
   pip install requests
   brew install espeak-ng  # macOS
   ```
4. **Test individual components**:
   ```bash
   # Test translation
   python -c "from translator import TextTranslator; print(TextTranslator().translate('hello', 'es'))"
   
   # Test TTS
   python -c "from piper_tts_pipeline import PiperSpeechSynthesizer; print(PiperSpeechSynthesizer().synthesize('test', 'en'))"
   ```
5. **Check file paths and permissions**
6. **Verify internet connection** for Google Translate API
7. **Check Piper installation** and model files

## 🌟 Key Features

- **🎤 High-Quality TTS**: Professional neural text-to-speech with PiperTTS
- **🌐 Automatic Translation**: Google Translate API integration for any language
- **📱 Easy to Use**: Simple one-command execution
- **🔧 Scalable**: No manual configuration needed for new sentences
- **🌍 Multi-Language**: Support for 9+ languages
- **📁 Organized**: Clean directory structure with all files in `src/`
- **📝 Logging**: Complete pipeline logging and audio file management
- **🛡️ Robust**: Error handling and fallback systems
- **⚡ Fast**: Real-time processing with efficient algorithms
- **🔍 Debuggable**: Comprehensive logging and error reporting

## 🎓 Educational Value

This project demonstrates:
- **Text-to-Speech Integration**: Using state-of-the-art TTS engines
- **API Integration**: Google Translate API usage
- **Error Handling**: Robust fallback systems
- **File Management**: Organized audio and log file handling
- **Multi-language Support**: Internationalization concepts
- **Pipeline Architecture**: Modular, maintainable code design

---

**Ready to get started?** Run `cd src && python asl_piper_pipeline.py` to see the magic happen! 🚀