import os
from typing import Any, Dict, List


class TranscriptSaver:
    """
    Class for saving diarized transcript with speaker merging.
    """

    def __init__(self, output_dir: str) -> None:
        """
        Initialize the TranscriptSaver.
        Args:
            output_dir (str): Directory to save the transcript file.
        """
        self.output_dir: str = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path: str = os.path.join(self.output_dir, "transcript.txt")

    def merge_segments(self, diarized_text: Any, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge consecutive segments of the same speaker.
        Args:
            diarized_text (Any): Diarization result (pyannote.core.Annotation).
            segments (List[Dict[str, Any]]): List of transcription segments.
        Returns:
            List[Dict[str, Any]]: List of merged segments with speaker labels.
        """
        # Collect speaker intervals
        speaker_turns: List[Any] = []
        for turn in diarized_text.itertracks(yield_label=True):
            segment, _, speaker = turn
            speaker_turns.append((segment.start, segment.end, speaker))
        # Assign speaker to each text segment and merge consecutive segments of the same speaker
        merged: List[Dict[str, Any]] = []
        last_speaker: Any = None
        last_start: Any = None
        last_end: Any = None
        last_text: List[str] = []
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_text = seg["text"].strip()
            speaker = "Unknown"
            for s_start, s_end, s_label in speaker_turns:
                # If the segment is fully within the speaker interval
                if seg_start >= s_start and seg_end <= s_end:
                    speaker = s_label
                    break
            if speaker == last_speaker:
                last_end = seg_end
                last_text.append(seg_text)
            else:
                if last_speaker is not None:
                    merged.append(
                        {"start": last_start, "end": last_end, "speaker": last_speaker, "text": " ".join(last_text)}
                    )
                last_speaker = speaker
                last_start = seg_start
                last_end = seg_end
                last_text = [seg_text]
        if last_speaker is not None:
            merged.append({"start": last_start, "end": last_end, "speaker": last_speaker, "text": " ".join(last_text)})
        return merged

    def save_transcript(self, diarized_text: Any, segments: List[Dict[str, Any]]) -> None:
        """
        Save the merged transcript to a file.
        Args:
            diarized_text (Any): Diarization result (pyannote.core.Annotation).
            segments (List[Dict[str, Any]]): List of transcription segments.
        """
        merged_segments = self.merge_segments(diarized_text, segments)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for seg in merged_segments:
                f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] Speaker {seg['speaker']}: {seg['text']}\n")
