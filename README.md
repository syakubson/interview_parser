# Interview Parser

This project extracts audio from a video, transcribes it using OpenAI Whisper, and performs speaker diarization using pyannote.audio. The result is a merged transcript with speaker labels.

## Features
- Audio extraction from video files
- Transcription with [OpenAI Whisper](https://github.com/openai/whisper)
- Speaker diarization with [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- Output transcript with speaker labels

## Requirements
- Python 3.10
- ffmpeg (system package)

## Installation

### 1. Create and activate a conda environment
```bash
conda create -n interview_parser python=3.10
conda activate interview_parser
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Install pre-commit hooks
```bash
pre-commit install
```

## HuggingFace Token for pyannote
The diarization pipeline uses the model `pyannote/speaker-diarization-3.1`, which requires a HuggingFace access token.

1. Register or log in at https://huggingface.co
2. Go to your account settings → Access Tokens: https://huggingface.co/settings/tokens
3. Create a new token (read access is enough)
4. Set the token as an environment variable before running the script:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```

## Usage

```bash
python code/main.py --video path/to/video.mp4 --speakers 2
```

- `--video`: Path to the input video file
- `--speakers`: Number of speakers in the audio
- `--interval`: (Optional) Start and end time in seconds, e.g. `--interval 10 60`

## Output
The transcript will be saved in the `out` directory as `transcript.txt` with speaker labels and timestamps.

## Notes
- Whisper and pyannote models will be downloaded automatically on first run.
- CUDA will be used automatically if available for faster processing. 