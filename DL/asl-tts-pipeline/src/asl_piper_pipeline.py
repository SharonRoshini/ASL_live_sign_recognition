import os
from datetime import datetime

# Handle both relative and absolute imports
try:
    from .translator import TextTranslator
    from .piper_tts_pipeline import PiperSpeechSynthesizer
except ImportError:
    from translator import TextTranslator
    from piper_tts_pipeline import PiperSpeechSynthesizer


class ASLPiperPipeline:
    """
    ASL Text → English TTS → Translation → Spanish TTS (via PiperTTS)
    """

    def __init__(self):
        self.translator = TextTranslator()
        self.tts = PiperSpeechSynthesizer()

        self.base_dir = os.path.dirname(__file__)
        self.output_dir = os.path.join(self.base_dir, "output_audio")
        os.makedirs(self.output_dir, exist_ok=True)

        self.log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def run(self, asl_text: str):
        print("\n" + "="*60)
        print(f"🎯 ASL Text: {asl_text}")
        print("="*60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Step 1: English speech
        print("\n⏳ Step 1: Generating English Speech")
        print("-" * 40)
        en_filename = f"en_piper_{timestamp}.wav"
        en_audio = self.tts.synthesize(asl_text, "en", en_filename)

        # Step 2: Translate
        print("\n⏳ Step 2: Translating to Spanish")
        print("-" * 40)
        translated = self.translator.translate(asl_text, "es")
        print(f"🌍 Translated: {translated}")

        # Step 3: Spanish speech
        print("\n⏳ Step 3: Generating Spanish Speech")
        print("-" * 40)
        es_filename = f"es_piper_{timestamp}.wav"
        es_audio = self.tts.synthesize(translated, "es", es_filename)

        # Step 4: Log
        print("\n⏳ Step 4: Logging Results")
        print("-" * 40)
        self._log(asl_text, translated, en_audio, es_audio)
        
        # Final summary
        print("\n" + "="*60)
        print("📊 PIPELINE SUMMARY")
        print("="*60)
        print("✅ ASL-to-Speech completed via PiperTTS!")
        print(f"🎧 English: {en_audio}")
        print(f"🎧 Spanish: {es_audio}")
        print("="*60)

    def _log(self, en_text, es_text, en_audio, es_audio):
        log_file = os.path.join(self.log_dir, "piper_log.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"English Text: {en_text}\n")
            f.write(f"Spanish Translation: {es_text}\n")
            f.write(f"English Audio: {en_audio}\n")
            f.write(f"Spanish Audio: {es_audio}\n")
        print(f"📝 Logged pipeline output to {log_file}")


# Quick test
if __name__ == "__main__":
    pipeline = ASLPiperPipeline()
    pipeline.run("I need help with my homework")
