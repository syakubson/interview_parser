#!/bin/bash

# Usage example:
# PYANNOTE_TOKEN=hf_xxx ./scripts/run_parser.sh path/to/video.mp4 0 60 2 ./result
# или для всего видео:
# PYANNOTE_TOKEN=hf_xxx ./scripts/run_parser.sh path/to/video.mp4 all all 2 ./result

export HUGGINGFACE_TOKEN=""

VIDEO=""
START="all"
END="all"
SPEAKERS=1

if [ "$START" = "all" ] || [ "$END" = "all" ]; then
  python3 code/main.py --video "$VIDEO" --speakers "$SPEAKERS"
else
  python3 code/main.py --video "$VIDEO" --interval "$START" "$END" --speakers "$SPEAKERS"
fi
