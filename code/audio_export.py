import ffmpeg
import tempfile
from typing import Optional, Tuple

class AudioProcessor:
    """
    Class for extracting and cutting audio from video files.

    Attributes:
        video_path (str): Path to the input video file.
        audio_path (Optional[str]): Path to the extracted or processed audio file.
    """
    def __init__(self, video_path: str) -> None:
        """
        Initialize AudioProcessor with the path to a video file.

        Args:
            video_path (str): Path to the video file.
        """
        self.video_path: str = video_path
        self.audio_path: Optional[str] = None

    def extract_audio(self) -> str:
        """
        Extract audio from the video file and save it as a temporary WAV file.

        Returns:
            str: Path to the extracted audio file.
        """
        tmp: tempfile._TemporaryFileWrapper = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_path: str = tmp.name
        tmp.close()
        (
            ffmpeg
            .input(self.video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        self.audio_path = audio_path
        return audio_path

    def cut_audio(self, start: float, end: float) -> str:
        """
        Cut a segment from the extracted audio between start and end times.

        Args:
            start (float): Start time in seconds.
            end (float): End time in seconds.

        Returns:
            str: Path to the cut audio file.
        """
        if self.audio_path is None:
            self.extract_audio()
        tmp: tempfile._TemporaryFileWrapper = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        cut_path: str = tmp.name
        tmp.close()
        (
            ffmpeg
            .input(self.audio_path, ss=start, to=end)
            .output(cut_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        self.audio_path = cut_path
        return cut_path

    def get_audio(self, interval: Optional[Tuple[float, float]] = None) -> Optional[str]:
        """
        Get the path to the audio file, optionally cutting it to a specific interval.

        Args:
            interval (Optional[Tuple[float, float]]): Tuple with start and end times in seconds.

        Returns:
            Optional[str]: Path to the audio file, or None if extraction failed.
        """
        self.extract_audio()
        if interval:
            self.cut_audio(interval[0], interval[1])
        return self.audio_path 