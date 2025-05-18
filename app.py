import gradio as gr
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))
from code.transcribe import WhisperTranscriber
from code.diarization import DiarizationPipeline
from code.output_utils import TranscriptSaver
from code.audio_export import AudioProcessor

# Save transcript to file
def save_transcript(text, path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return f"✅ Saved to {path}"
    except Exception as e:
        return f"❌ Error: {e}"

LANGUAGES = [
    ("ru", "ru"),
    ("en", "en"),
]

def extract_audio_from_video(video_path):
    if not video_path:
        return None, "❌ No video file uploaded."
    try:
        processor = AudioProcessor(video_path)
        audio_path = processor.get_audio()
        return audio_path, f"✅ Audio extracted: {audio_path}"
    except Exception as e:
        return None, f"❌ Error: {e}"

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
    with gr.Tab("1. Video & Audio"):
        video_file = gr.File(label="Upload video file", type="filepath")
        extract_btn = gr.Button("Extract audio")
        audio_path_state = gr.State()
        audio_player = gr.Audio(label="Audio preview", interactive=False)
        extract_status = gr.Markdown("")
        
        start_time = gr.Number(label="Start time (seconds, optional)", value=None, precision=2)
        end_time = gr.Number(label="End time (seconds, optional)", value=None, precision=2)
        cut_btn = gr.Button("Cut audio by interval")
        cut_status = gr.Markdown("")
        
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
            outputs=[audio_path_state, audio_player, cut_status]
        )
    
    with gr.Tab("2. Transcriber Settings"):
        num_speakers = gr.Number(label="Number of speakers", value=2, precision=0)
        whisper_model = gr.Dropdown([
            "tiny", "base", "small", "medium", "large"
        ], label="Whisper model", value="base")
        language = gr.Dropdown(choices=[(code, name) for code, name in LANGUAGES], value="ru", label="Language")
        hf_token = gr.Textbox(label="HuggingFace token", type="password")
        transcribe_btn = gr.Button("Transcribe")
        stop_btn = gr.Button("Stop transcription", variant="stop")
        transcribe_output = gr.Textbox(label="Transcript", lines=10)
        transcribe_status = gr.Markdown("")
        audio_path_state2 = gr.State()
        
        def pass_audio_path(audio_path):
            return audio_path
        
        # Pass audio path from the first tab to the second tab
        audio_player.change(pass_audio_path, inputs=[audio_path_state], outputs=[audio_path_state2])
        
        def on_transcribe(audio_path, speakers, model, lang, token, progress=gr.Progress(track_tqdm=True)):
            try:
                if not audio_path:
                    return "No audio file for transcription.", "", "❌ No audio file for transcription."
                progress(0, desc="Loading model...")
                device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("NVIDIA_VISIBLE_DEVICES")) else "cpu"
                transcriber = WhisperTranscriber(model_name=model, device=device)
                progress(0.2, desc="Transcribing audio...")
                result = transcriber.transcribe(audio_path, language=lang)
                progress(0.5, desc="Running diarization...")
                diarizer = DiarizationPipeline(token, device=device)
                diarized = diarizer.diarize(audio_path, int(speakers))
                saver = TranscriptSaver("out")
                merged = saver.merge_segments(diarized, result["segments"])
                transcript_text = "\n".join([
                    f"[{seg['start']:.2f} - {seg['end']:.2f}] Speaker {seg['speaker']}: {seg['text']}" for seg in merged
                ])
                progress(1, desc="Done!")
                return transcript_text, transcript_text, "✅ Transcription & diarization complete!"
            except Exception as e:
                return "", "", f"❌ Error: {e}"

        transcribe_event = transcribe_btn.click(
            on_transcribe,
            inputs=[audio_path_state, num_speakers, whisper_model, language, hf_token],
            outputs=[transcribe_output, transcribe_output, transcribe_status],
            queue=True
        )
        stop_btn.click(None, cancels=[transcribe_event])
    
    with gr.Tab("3. Result & Save"):
        transcript = gr.Textbox(label="Transcript", lines=10)
        save_path = gr.Textbox(label="Save path (optional)")
        save_btn = gr.Button("Save transcript")
        save_status = gr.Markdown("")

        def on_save(text, path):
            if not text or not path:
                return "Please provide both transcript and save path."
            return save_transcript(text, path)

        save_btn.click(
            on_save,
            inputs=[transcript, save_path],
            outputs=save_status
        )

if __name__ == "__main__":
    demo.launch() 