import click
from loguru import logger
from audio_export import AudioProcessor
from transcribe import WhisperTranscriber
import os
from output_utils import TranscriptSaver
from diarization import DiarizationPipeline
import torch

@click.command()
@click.option('--video', required=True, type=click.Path(exists=True), help='Path to video file')
@click.option('--interval', nargs=2, type=float, required=False, help='Time interval in seconds (start end)')
@click.option('--speakers', required=True, type=int, help='Number of speakers')
def main(video: str, interval: tuple = None, speakers: int = 1) -> None:
    """
    Main entry point for the CLI tool.
    Args:
        video (str): Path to video file.
        interval (tuple, optional): Time interval (start, end) in seconds.
        speakers (int): Number of speakers.
    """
    token = os.environ.get('HUGGINGFACE_TOKEN')
    if not token:
        raise RuntimeError('HUGGINGFACE_TOKEN environment variable not set')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info('Extracting audio from video...')
    processor = AudioProcessor(video)
    audio_path = processor.get_audio(interval)

    logger.info('Transcribing audio...')
    transcriber = WhisperTranscriber(device=device)
    transcribed = transcriber.transcribe(audio_path)

    logger.info('Running diarization...')
    diarizer = DiarizationPipeline(token, device=device)
    diarized = diarizer.diarize(audio_path, speakers)

    logger.info('Saving transcript...')
    saver = TranscriptSaver('out')
    saver.save_transcript(diarized, transcribed['segments'])
    
    logger.info('Done!')

if __name__ == "__main__":
    main()
