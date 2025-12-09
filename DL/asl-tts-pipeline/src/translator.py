"""
Advanced text translator for the ASL TTS pipeline.
Uses Google Translate API for automatic translation with fallback to local dictionary.
"""

import requests
import json
import os

class TextTranslator:
    """
    Advanced text translator with Google Translate API and fallback dictionary.
    Automatically translates any text without manual configuration.
    """
    
    def __init__(self):
        # Google Translate API configuration
        self.google_translate_url = "https://translate.googleapis.com/translate_a/single"
        self.fallback_translations = {
            "en_to_es": {
                # Essential fallback translations for common words
                "hello": "hola",
                "goodbye": "adiós",
                "thank you": "gracias",
                "please": "por favor",
                "yes": "sí",
                "no": "no",
                "help": "ayuda",
                "school": "escuela",
                "homework": "tarea",
                "water": "agua",
                "food": "comida",
                "eat": "comer",
                "drink": "beber",
                "sleep": "dormir",
                "family": "familia",
                "friend": "amigo",
                "today": "hoy",
                "tomorrow": "mañana",
                "now": "ahora"
            }
        }
    
    def translate(self, text: str, target_lang: str) -> str:
        """
        Translate text to target language using Google Translate API with fallback.
        
        Args:
            text: Input text to translate
            target_lang: Target language code (e.g., 'es' for Spanish)
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text
            
        # Try Google Translate API first
        try:
            translated = self._google_translate(text, target_lang)
            if translated and translated.strip():
                return translated
        except Exception as e:
            print(f"[WARNING] Google Translate failed: {e}")
        
        # Fallback to local dictionary
        return self._fallback_translate(text, target_lang)
    
    def _google_translate(self, text: str, target_lang: str) -> str:
        """
        Translate using Google Translate API (free, no API key required).
        """
        try:
            # Map language codes
            lang_map = {
                "es": "es",  # Spanish
                "en": "en",  # English
                "fr": "fr",  # French
                "de": "de",  # German
                "it": "it",  # Italian
                "pt": "pt",  # Portuguese
                "ru": "ru",  # Russian
                "ja": "ja",  # Japanese
                "ko": "ko",  # Korean
                "zh": "zh",  # Chinese
            }
            
            target_code = lang_map.get(target_lang, target_lang)
            source_code = "en"  # Assume English input
            
            # Google Translate API parameters
            params = {
                "client": "gtx",
                "sl": source_code,
                "tl": target_code,
                "dt": "t",
                "q": text
            }
            
            response = requests.get(self.google_translate_url, params=params, timeout=5)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            if result and len(result) > 0 and result[0]:
                translated_text = ""
                for item in result[0]:
                    if item[0]:  # If translation exists
                        translated_text += item[0]
                return translated_text.strip()
            
        except Exception as e:
            print(f"[ERROR] Google Translate API error: {e}")
            raise e
        
        return None
    
    def _fallback_translate(self, text: str, target_lang: str) -> str:
        """
        Fallback translation using local dictionary.
        """
        text_lower = text.lower().strip()
        
        if target_lang == "es":
            # Check for exact matches in fallback dictionary
            if text_lower in self.fallback_translations["en_to_es"]:
                return self.fallback_translations["en_to_es"][text_lower]
            
            # Try word-by-word translation for simple cases
            words = text_lower.split()
            translated_words = []
            
            for word in words:
                clean_word = word.strip('.,!?;:')
                if clean_word in self.fallback_translations["en_to_es"]:
                    translated_words.append(self.fallback_translations["en_to_es"][clean_word])
                else:
                    # Keep original word if no translation found
                    translated_words.append(word)
            
            # If we translated some words, return the result
            if any(word in self.fallback_translations["en_to_es"] for word in [w.strip('.,!?;:') for w in words]):
                return " ".join(translated_words)
            
            # Final fallback
            return f"[Translation unavailable: {text}]"
        
        # For other languages
        return f"[Translation to {target_lang} not implemented: {text}]"
    
    def add_translation(self, source_text: str, target_text: str, target_lang: str):
        """
        Add a new translation to the dictionary.
        """
        if target_lang == "es":
            self.translations["en_to_es"][source_text.lower()] = target_text
