from pyannote.audio import Pipeline, Audio
import os
from loguru import logger
from typing import Any

class DiarizationPipeline:
    """
    Class for speaker diarization using pyannote.audio.
    Loads model from models/pyannote or downloads if not present.
    """
    def __init__(self, token: str, model_name: str = 'pyannote/speaker-diarization-3.1', device: str = 'cpu') -> None:
        """
        Args:
            token (str): HuggingFace access token.
            model_name (str): Name of the pyannote model.
            device (str): Device to run the model on ("cpu" or "cuda").
        """
        self.token = token
        self.model_name = model_name
        self.device = device
        self.model_dir = os.path.join('models', 'pyannote')
        os.makedirs(self.model_dir, exist_ok=True)

        # pyannote uses huggingface cache, so we just set cache_dir
        if not os.path.exists(self.model_dir) or not os.listdir(self.model_dir):
            logger.info(f"Pyannote model '{model_name}' not found in '{self.model_dir}', starting download...")
        self.pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=token,
            cache_dir=self.model_dir,
        )
        if self.device == "cuda":
            self.pipeline.to(0)
        self.audio = Audio()

    def diarize(self, audio_path: str, num_speakers: int) -> Any:
        """
        Perform diarization on the given audio file.
        Args:
            audio_path (str): Path to the audio file.
            num_speakers (int): Number of speakers.
        Returns:
            Any: Diarization result (pyannote.core.Annotation)
        """
        waveform, sample_rate = self.audio(audio_path)
        diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=num_speakers)
        return diarization 