import os
import sys

import gradio as gr

# sys.path.append(os.path.join(os.path.dirname(__file__), "code"))
from code.audio_export import AudioProcessor
from code.diarization import DiarizationPipeline
from code.output_utils import TranscriptSaver
from code.transcribe import WhisperTranscriber


# Save transcript to file
def save_transcript(text, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return f"✅ Saved to {path}"
    except Exception as e:
        return f"❌ Error: {e}"


LANGUAGES = [
    ("Русский", "ru"),
    ("English", "en"),
]


def extract_audio_from_video(video_path):
    if not video_path:
        return None, "❌ No video file uploaded."
    # try:
    processor = AudioProcessor(video_path)
    audio_path = processor.get_audio()
    return audio_path, f"✅ Audio extracted: {audio_path}"
    # except Exception as e:
    #     return None, f"❌ Error: {e}"


def cut_audio(audio_path, start, end):
    if not audio_path:
        return None, "❌ No audio file to cut."
    try:
        if start is None or end is None:
            return audio_path, "⚠️ Interval not set, using full audio."
        processor = AudioProcessor("")
        processor.audio_path = audio_path
        cut_path = processor.cut_audio(float(start), float(end))
        return cut_path, f"✅ Audio cut: {cut_path}"
    except Exception as e:
        return None, f"❌ Error: {e}"


with gr.Blocks() as demo:
    gr.Markdown("# Interview Parser")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tab("Load & Cut"):
                video_file = gr.File(label="Upload video file", type="filepath", elem_classes=["compact-upload"])
                extract_btn = gr.Button("Extract audio", elem_classes=["compact-btn"])
                extract_status = gr.Markdown("")
                with gr.Row():
                    start_time = gr.Number(label="Cut start (s)", value=None, precision=2, scale=1)
                    end_time = gr.Number(label="Cut end (s)", value=None, precision=2, scale=1)
                cut_btn = gr.Button("Cut audio by interval", elem_classes=["compact-btn"])
                cut_status = gr.Markdown("")
                audio_path_state = gr.State()
            with gr.Tab("Extraction Settings"):
                with gr.Row():
                    num_speakers = gr.Number(label="Speakers", value=2, precision=0, scale=1)
                    whisper_model = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"], label="Model", value="base", scale=1)
                    language = gr.Dropdown(choices=[(code, name) for code, name in LANGUAGES], value="ru", label="Lang", scale=1)
                hf_token = gr.Textbox(label="HuggingFace token for Pyannote", type="password", elem_classes=["compact-token"])
                with gr.Row():
                    transcribe_btn = gr.Button("Transcribe", elem_classes=["compact-btn"])
                    stop_btn = gr.Button("Stop", variant="stop", elem_classes=["compact-btn"])
                transcribe_status = gr.Markdown("")
        with gr.Column(scale=3):
            audio_player = gr.Audio(label="Audio preview", interactive=False)
            transcribe_output = gr.Textbox(label="Transcript", lines=10)
            save_path = gr.Textbox(label="Save path (optional)")
            save_btn = gr.Button("Save transcript")
            save_status = gr.Markdown("")
    
    def on_extract(video):
        audio_path, status = extract_audio_from_video(video)
        return audio_path, audio_path, status
    
    extract_btn.click(
        on_extract,
        inputs=[video_file],
        outputs=[audio_path_state, audio_player, extract_status]
    )
    
    def on_cut(audio_path, start, end):
        cut_path, status = cut_audio(audio_path, start, end)
        return cut_path, cut_path, status
    
    cut_btn.click(
        on_cut,
        inputs=[audio_path_state, start_time, end_time],
        outputs=[audio_path_state, audio_player, cut_status],
    )
    
    def on_transcribe(audio_path, speakers, model, lang, token, progress=gr.Progress(track_tqdm=True)):
        try:
            if not audio_path:
                return "No audio file for transcription.", "❌ No audio file for transcription.", ""
            progress(0, desc="Loading model...")
            device = (
                "cuda"
                if (os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("NVIDIA_VISIBLE_DEVICES"))
                else "cpu"
            )
            transcriber = WhisperTranscriber(model_name=model, device=device)
            progress(0.2, desc="1. Getting text from audio...")
            result = transcriber.transcribe(audio_path, language=lang)
            progress(0.5, desc="2. Setting speakers...")
            diarizer = DiarizationPipeline(token, device=device)
            diarized = diarizer.diarize(audio_path, int(speakers))
            print(diarized)
            saver = TranscriptSaver("out")
            merged = saver.merge_segments(diarized, result["segments"])
            transcript_text = "\n".join(
                [
                    f"[{seg['start']:.2f} - {seg['end']:.2f}] Speaker {seg['speaker']}: {seg['text']}"
                    for seg in merged
                ]
            )
            progress(1, desc="Done!")
            return transcript_text, transcript_text, "✅ Transcription complete!"
        except Exception as e:
            return "", f"❌ Error: {e}", ""

    transcribe_event = transcribe_btn.click(
        on_transcribe,
        inputs=[audio_path_state, num_speakers, whisper_model, language, hf_token],
        outputs=[transcribe_output, transcribe_output, transcribe_status],
        queue=True,
    )
    stop_btn.click(None, cancels=[transcribe_event])
    
    def on_save(text, path):
        if not text or not path:
            return "Please provide both transcript and save path."
        return save_transcript(text, path)
    
    save_btn.click(on_save, inputs=[transcribe_output, save_path], outputs=save_status)

if __name__ == "__main__":
    demo.launch()
