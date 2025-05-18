import os
import torch
from typing import List, Optional, Tuple
from code.audio_export import AudioProcessor

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

def list_prompts() -> List[str]:
    """Return list of prompt filenames in the prompts directory."""
    return [f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")]

def read_prompt(filename: str) -> str:
    """Read prompt text from file by filename."""
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def add_prompt_to_text(prompt: str, text: str) -> str:
    """Add prompt to the beginning of text."""
    if not text:
        return prompt
    return prompt + "\n" + text

def extract_audio_from_video(video_path: str) -> Tuple[Optional[str], str]:
    """Extract audio from video file and return audio path and status message."""
    if not video_path:
        return None, "❌ No video file uploaded."
    try:
        processor = AudioProcessor(video_path)
        audio_path = processor.get_audio()
        return audio_path, f"✅ Audio extracted"
    except Exception as e:
        return None, f"❌ Error: {e}"

def cut_audio(audio_path: str, start: float, end: float) -> Tuple[Optional[str], str]:
    """Cut audio by interval and return cut audio path and status message."""
    if not audio_path:
        return None, "❌ No audio file to cut."
    try:
        if start is None or end is None:
            return audio_path, "⚠️ Interval not set, using full audio."
        processor = AudioProcessor("")
        processor.audio_path = audio_path
        cut_path = processor.cut_audio(float(start), float(end))
        return cut_path, f"✅ Audio cut"
    except Exception as e:
        return None, f"❌ Error: {e}"

def get_device_name() -> str:
    """Return device type and name (CPU or CUDA + GPU name)"""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA ({name})"
    else:
        return "CPU" 

def get_device() -> str:
    """Return device type (CPU or CUDA)"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu" 