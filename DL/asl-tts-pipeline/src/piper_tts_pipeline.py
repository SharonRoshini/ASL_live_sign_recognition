import os
import subprocess
from datetime import datetime


class PiperSpeechSynthesizer:
    """
    Text-to-speech synthesis using PiperTTS (lightweight, high-quality).
    Requires Piper binaries and voice models.
    """

    def __init__(self):
        # Define Piper models (downloaded voice files)
        # You can list available voices at https://github.com/rhasspy/piper
        self.models = {
            "en": "en_US-lessac-medium.onnx",  
            "es": "es_ES-sharvard-medium.onnx" 
        }

        # Define output and model directories
        self.base_dir = os.path.dirname(__file__)
        self.model_dir = os.path.join(self.base_dir, "piper_models")
        self.output_dir = os.path.join(self.base_dir, "output_audio")

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Path to Piper executable - auto-detect based on project structure
        self.piper_bin = self._find_piper_executable()

    def _find_piper_executable(self):
        """Find the Piper executable across different systems and installations."""
        import shutil
        
        # List of possible Piper executable names
        piper_names = ["piper", "piper.exe"]
        
        # List of possible locations to search
        search_paths = [
            # Local project build directory (piper folder is now inside asl-tts-pipeline)
            os.path.join(os.path.dirname(os.path.dirname(self.base_dir)), "piper", "build"),
            # Alternative local build path
            os.path.join(os.path.dirname(self.base_dir), "..", "piper", "build"),
            # Direct path from src directory
            os.path.join(os.path.dirname(self.base_dir), "piper", "build"),
            # System-wide installation
            "/usr/local/bin",
            "/usr/bin", 
            "/opt/homebrew/bin",
            # Windows common locations
            "C:\\Program Files\\piper",
            "C:\\Program Files (x86)\\piper",
            # Current directory and PATH
            "."
        ]
        
        # First, try to find in PATH
        for piper_name in piper_names:
            piper_path = shutil.which(piper_name)
            if piper_path and os.path.isfile(piper_path):
                return piper_path
        
        # Then search in specific directories
        for search_path in search_paths:
            if os.path.exists(search_path):
                for piper_name in piper_names:
                    piper_path = os.path.join(search_path, piper_name)
                    if os.path.isfile(piper_path):
                        return piper_path
        
        # If not found, raise an error with helpful message
        raise FileNotFoundError(
            "Piper executable not found. Please ensure Piper is installed:\n"
            "  - macOS: brew install piper-tts\n"
            "  - Ubuntu/Debian: sudo apt install piper-tts\n"
            "  - Or build from source in the piper/ directory"
        )

    def synthesize(self, text: str, lang: str = "en", filename: str = None, speed: float = 1.0):
        """
        Generate speech using PiperTTS with high accuracy and natural pronunciation.
        """
        if lang not in self.models:
            raise ValueError(f"No Piper model found for language: {lang}")

        model_path = os.path.join(self.model_dir, self.models[lang])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Piper model not found at {model_path}")

        # Generate output file path
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{lang}_piper_{timestamp}.wav"
        output_path = os.path.join(self.output_dir, filename)

        # Piper doesn't have a direct speed parameter — we emulate by changing sample rate later
        cmd = [
            self.piper_bin,
            "--model", model_path,
            "--espeak_data", "/opt/homebrew/share/espeak-ng-data",
            "--output_file", output_path
        ]


        try:
            # Run Piper via subprocess and feed text via stdin
            result = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
            if result.returncode == 0:
                size_kb = os.path.getsize(output_path) / 1024
                print(f"✅ Generated: {os.path.basename(output_path)} ({size_kb:.1f} KB)")
                return output_path
            else:
                print(f"[ERROR] Piper failed: {result.stderr.decode('utf-8')}")
                return None
        except Exception as e:
            print(f"[ERROR] Piper synthesis failed: {e}")
            return None

    def synthesize_with_quality(self, text: str, lang: str = "en", quality: str = "high"):
        """
        Apply quality profiles. Piper doesn’t use speed/quality params like Coqui,
        but you can prepare alternate models (low/medium/high) for comparison.
        """
        print(f"[INFO] Synthesizing with '{quality}' quality preset...")
        return self.synthesize(text, lang=lang)
