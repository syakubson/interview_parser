import whisper
from typing import Dict, Any, Optional
import os
from loguru import logger

class WhisperTranscriber:
    """
    Class for audio transcription using Whisper.
    Always uses models/whisper as the download and search directory.
    If the model is not found, it will be downloaded automatically.
    """
    def __init__(self, model_name: str = "base", device: str = "cpu") -> None:
        """
        Args:
            model_name (str): Whisper model name (e.g., "base", "small", "medium", "large").
            device (str): Device to run the model on ("cpu" or "cuda").
        """
        self.model_name = model_name
        self.device = device
        self.download_dir = os.path.join("models", "whisper")
        model_path = os.path.join(self.download_dir, model_name + ".pt")

        # Check if model not downloaded
        if not os.path.exists(model_path):
            logger.info(f"Whisper model '{model_name}' not found in '{self.download_dir}', starting download...")
            os.makedirs(self.download_dir, exist_ok=True)
        self.model = whisper.load_model(model_name, download_root=self.download_dir, device=self.device)

    def transcribe(self, audio_path: str, language: str = 'ru') -> Dict[str, Any]:
        """
        Transcribe an audio file.
        Args:
            audio_path (str): Path to the audio file.
            language (str): Audio language (default: 'ru').
        Returns:
            Dict[str, Any]: Transcription result with keys 'text' and 'segments'.
        """
        audio = whisper.load_audio(audio_path)
        result = self.model.transcribe(audio, language=language, word_timestamps=True, verbose=False, fp16=(self.device=="cuda"))
        return {
            'text': result['text'],
            'segments': result['segments']
        } 