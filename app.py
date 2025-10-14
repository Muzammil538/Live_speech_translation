import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sounddevice as sd
import numpy as np
import json
import queue
import threading
import vosk
import argostranslate.package
import argostranslate.translate
import pyttsx3
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import librosa
import tempfile
import wave
# from indictrans2 import IndicProcessor

class OfflineSpeechTranslator:
    def __init__(self):
        self.setup_gui()
        self.setup_models()
        self.audio_queue = queue.Queue()
        self.recording = False
        
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Offline Speech Translator - Indian Languages")
        self.root.geometry("800x600")
        
        # Language selection
        lang_frame = ttk.Frame(self.root)
        lang_frame.pack(pady=10)
        
        ttk.Label(lang_frame, text="Target Language:").pack(side=tk.LEFT)
        self.target_lang = ttk.Combobox(lang_frame, values=[
            "English", "Hindi", "Bengali", "Tamil", "Telugu", 
            "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"
        ])
        self.target_lang.set("Hindi")
        self.target_lang.pack(side=tk.LEFT, padx=10)
        
        # Control buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.record_btn = ttk.Button(btn_frame, text="Start Recording", 
                                   command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        self.translate_btn = ttk.Button(btn_frame, text="Translate", 
                                      command=self.translate_audio)
        self.translate_btn.pack(side=tk.LEFT, padx=5)
        
        self.speak_btn = ttk.Button(btn_frame, text="Speak Translation", 
                                  command=self.speak_translation)
        self.speak_btn.pack(side=tk.LEFT, padx=5)
        
        # Text areas
        text_frame = ttk.Frame(self.root)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(text_frame, text="Recognized Text:").pack(anchor=tk.W)
        self.source_text = scrolledtext.ScrolledText(text_frame, height=8)
        self.source_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        ttk.Label(text_frame, text="Translation:").pack(anchor=tk.W)
        self.target_text = scrolledtext.ScrolledText(text_frame, height=8)
        self.target_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_models(self):
        """Initialize all required models"""
        self.status_var.set("Loading models...")
        
        try:
            # 1. Setup Vosk for Speech Recognition
            self.setup_vosk_model()
            
            # 2. Setup Argos Translate for offline translation
            self.setup_translation_models()
            
            # 3. Setup TTS engine
            self.tts_engine = pyttsx3.init()
            self.setup_tts_voices()
            
            # 4. Setup language detection
            self.setup_language_detection()
            
            self.status_var.set("All models loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def setup_vosk_model(self):
        """Setup Vosk speech recognition models for multiple languages"""
        # Download Vosk models for different languages
        # You need to download these models separately
        self.vosk_models = {
            'en': vosk.Model("vosk-model-small-en-in-0.4"),  # English
            'hi': vosk.Model("vosk-model-small-hi-0.22"),     # Hindi     # Bengali
        }
        
        # Default recognizer (English)
        self.recognizer = vosk.KaldiRecognizer(self.vosk_models['en'], 16000)
        
    def setup_translation_models(self):
        """Setup offline translation models"""
        # Update package index
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        
        # Language mappings
        self.lang_codes = {
            "English": "en",
            "Hindi": "hi",
            "Bengali": "bn",
            "Tamil": "ta",
            "Telugu": "te",
            "Marathi": "mr",
            "Gujarati": "gu",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Punjabi": "pa"
        }
        
        # Install required translation packages
        for lang in self.lang_codes.values():
            if lang != "en":
                # Install en->lang and lang->en packages
                try:
                    en_to_lang = next((pkg for pkg in available_packages 
                                     if pkg.from_code == "en" and pkg.to_code == lang), None)
                    if en_to_lang:
                        argostranslate.package.install_from_path(en_to_lang.download())
                    
                    lang_to_en = next((pkg for pkg in available_packages 
                                     if pkg.from_code == lang and pkg.to_code == "en"), None)
                    if lang_to_en:
                        argostranslate.package.install_from_path(lang_to_en.download())
                except:
                    print(f"Warning: Could not install translation package for {lang}")
        
        # Alternative: Load IndicTrans2 models for better Indian language support
        self.setup_indic_trans()
    
    def setup_indic_trans(self):
        """Setup IndicTrans2 for better Indian language translation"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Load IndicTrans2 models
            self.indic_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B")
            self.indic_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B")
            
            self.en_indic_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-en-1B")
            self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-1B")
            
        except Exception as e:
            print(f"Warning: Could not load IndicTrans2 models: {e}")
            self.indic_model = None
    
    def setup_tts_voices(self):
        """Setup TTS voices for different languages"""
        voices = self.tts_engine.getProperty('voices')
        self.tts_voices = {}
        
        # Map available voices to languages
        for voice in voices:
            if 'english' in voice.name.lower():
                self.tts_voices['en'] = voice.id
            elif 'hindi' in voice.name.lower():
                self.tts_voices['hi'] = voice.id
    
    def setup_language_detection(self):
        """Setup language detection"""
        # Simple language detection based on script/characters
        self.lang_patterns = {
            'hi': ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ'],
            'bn': ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'এ', 'ঐ', 'ও', 'ঔ'],
            'ta': ['அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ'],
            'te': ['అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఎ', 'ఏ', 'ఐ', 'ఒ'],
        }
    
    def detect_language(self, text):
        """Detect language of input text"""
        if not text:
            return 'en'
        
        for lang, patterns in self.lang_patterns.items():
            if any(char in text for char in patterns):
                return lang
        
        return 'en'  # Default to English
    
    def toggle_recording(self):
        """Toggle audio recording"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start audio recording"""
        self.recording = True
        self.record_btn.config(text="Stop Recording")
        self.status_var.set("Recording... Speak now!")
        
        # Start recording in a separate thread
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.start()
    
    def stop_recording(self):
        """Stop audio recording"""
        self.recording = False
        self.record_btn.config(text="Start Recording")
        self.status_var.set("Processing audio...")
    
    def record_audio(self):
        """Record audio from microphone"""
        sample_rate = 16000
        channels = 1
        
        def audio_callback(indata, frames, time, status):
            if self.recording:
                self.audio_queue.put(indata.copy())
        
        # Record audio
        audio_data = []
        with sd.InputStream(samplerate=sample_rate, channels=channels, 
                           callback=audio_callback, dtype=np.float32):
            while self.recording:
                try:
                    data = self.audio_queue.get(timeout=0.1)
                    audio_data.append(data)
                except queue.Empty:
                    continue
        
        # Save audio data
        if audio_data:
            self.audio_array = np.concatenate(audio_data)
            self.sample_rate = sample_rate
    
    def translate_audio(self):
        """Translate recorded audio"""
        if not hasattr(self, 'audio_array'):
            messagebox.showwarning("Warning", "No audio recorded!")
            return
        
        try:
            # Convert audio to text
            self.status_var.set("Converting speech to text...")
            text = self.speech_to_text(self.audio_array, self.sample_rate)
            
            if not text:
                messagebox.showwarning("Warning", "No speech detected!")
                return
            
            self.source_text.delete(1.0, tk.END)
            self.source_text.insert(tk.END, text)
            
            # Detect source language
            detected_lang = self.detect_language(text)
            
            # Translate text
            self.status_var.set("Translating...")
            target_lang_code = self.lang_codes[self.target_lang.get()]
            
            translated_text = self.translate_text(text, detected_lang, target_lang_code)
            
            self.target_text.delete(1.0, tk.END)
            self.target_text.insert(tk.END, translated_text)
            
            self.status_var.set("Translation completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Translation failed: {str(e)}")
            self.status_var.set("Translation failed!")
    
    def speech_to_text(self, audio_data, sample_rate):
        """Convert speech to text using Vosk"""
        # Convert float32 to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Reset recognizer
        self.recognizer = vosk.KaldiRecognizer(self.vosk_models['en'], sample_rate)
        
        # Process audio
        if self.recognizer.AcceptWaveform(audio_int16.tobytes()):
            result = json.loads(self.recognizer.Result())
            return result.get('text', '')
        else:
            partial_result = json.loads(self.recognizer.PartialResult())
            return partial_result.get('partial', '')
    
    def translate_text(self, text, source_lang, target_lang):
        """Translate text using offline models"""
        if source_lang == target_lang:
            return text
        
        try:
            # Try IndicTrans2 first for Indian languages
            if self.indic_model and (source_lang in ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa'] 
                                   or target_lang in ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa']):
                return self.translate_with_indictrans(text, source_lang, target_lang)
            
            # Fall back to Argos Translate
            return argostranslate.translate.translate(text, source_lang, target_lang)
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def translate_with_indictrans(self, text, source_lang, target_lang):
        """Translate using IndicTrans2 models"""
        if source_lang == 'en' and target_lang in ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa']:
            # English to Indic
            inputs = self.indic_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.indic_model.generate(**inputs, max_length=512)
            return self.indic_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        elif source_lang in ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa'] and target_lang == 'en':
            # Indic to English
            inputs = self.en_indic_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.en_indic_model.generate(**inputs, max_length=512)
            return self.en_indic_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        else:
            # For Indic to Indic, pivot through English
            if source_lang != 'en':
                english_text = self.translate_with_indictrans(text, source_lang, 'en')
                return self.translate_with_indictrans(english_text, 'en', target_lang)
            return text
    
    def speak_translation(self):
        """Convert translated text to speech"""
        translated_text = self.target_text.get(1.0, tk.END).strip()
        
        if not translated_text:
            messagebox.showwarning("Warning", "No translation to speak!")
            return
        
        try:
            self.status_var.set("Speaking translation...")
            
            target_lang_code = self.lang_codes[self.target_lang.get()]
            
            # Set appropriate voice if available
            if target_lang_code in self.tts_voices:
                self.tts_engine.setProperty('voice', self.tts_voices[target_lang_code])
            
            # Speak the text
            self.tts_engine.say(translated_text)
            self.tts_engine.runAndWait()
            
            self.status_var.set("Speech completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"TTS failed: {str(e)}")
            self.status_var.set("Speech failed!")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

# Main execution
if __name__ == "__main__":
    app = OfflineSpeechTranslator()
    app.run()
