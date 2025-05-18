import os
from code.audio_export import AudioProcessor
from code.diarization import DiarizationPipeline
from code.output_utils import TranscriptSaver
from code.transcribe import WhisperTranscriber

import gradio as gr
import torch

LANGUAGES = [
    ("Русский", "ru"),
    ("English", "en"),
]

PROMPTS_DIR = "prompts"
prompt_files = [f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")]


def extract_audio_from_video(video_path):
    if not video_path:
        return None, "❌ No video file uploaded."
    try:
        processor = AudioProcessor(video_path)
        audio_path = processor.get_audio()
        return audio_path, f"✅ Audio extracted"
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
        return cut_path, f"✅ Audio cut"
    except Exception as e:
        return None, f"❌ Error: {e}"


def get_device_status():
    """Return device type and name (CPU or CUDA + GPU name)"""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA ({name})"
    else:
        return "CPU"


with gr.Blocks() as demo:
    gr.Markdown("# Interview Parser")
    device_status = gr.State(get_device_status())
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tab("Load & Cut"):
                device_info = gr.Markdown(f"**Device:** {get_device_status()}")
                video_file = gr.File(label="Upload video file", type="filepath", elem_classes=["compact-upload"])
                extract_btn = gr.Button("Extract Audio", elem_classes=["compact-btn"])
                extract_status = gr.Markdown("")
                with gr.Row():
                    start_time = gr.Number(label="Cut Start (s)", value=0, precision=2, scale=1)
                    end_time = gr.Number(label="Cut End (s)", value=60, precision=2, scale=1)
                cut_btn = gr.Button("Cut Audio by Interval", elem_classes=["compact-btn"])
                cut_status = gr.Markdown("")
                audio_path_state = gr.State()
            with gr.Tab("Extraction Settings"):
                with gr.Row():
                    num_speakers = gr.Number(label="Speakers", value=2, precision=0, scale=1)
                    whisper_model = gr.Dropdown(
                        ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
                        label="Whisper model",
                        value="base",
                        scale=1,
                    )
                    language = gr.Dropdown(
                        choices=[(code, name) for code, name in LANGUAGES], value="ru", label="Language", scale=1
                    )
                transcribe_mode = gr.Radio(
                    ["Text only", "Text with speaker roles"],
                    value="Text with speaker roles",
                    label="Transcription mode",
                )
                hf_token = gr.Textbox(
                    label="HuggingFace token for Pyannote",
                    type="password",
                    elem_classes=["compact-token"],
                    visible=True,
                )
                with gr.Row():
                    transcribe_btn = gr.Button("Transcribe", elem_classes=["compact-btn"])
                transcribe_status = gr.Markdown("")
            with gr.Tab("Add prompts"):
                prompt_selector = gr.Dropdown(prompt_files, label="Select prompt")
                prompt_preview = gr.Textbox(label="Prompt preview", interactive=False, lines=6)
                add_prompt_btn = gr.Button("Add prompt to transcript")
                add_prompt_status = gr.Markdown("")
        with gr.Column(scale=3):
            audio_player = gr.Audio(label="Audio Preview", interactive=False, show_download_button=True, scale=3)
            transcribe_output = gr.Textbox(label="Transcript", lines=10, show_copy_button=True)

    def on_extract(video):
        audio_path, status = extract_audio_from_video(video)
        return audio_path, audio_path, status

    extract_btn.click(on_extract, inputs=[video_file], outputs=[audio_path_state, audio_player, extract_status])

    def on_cut(audio_path, start, end):
        cut_path, status = cut_audio(audio_path, start, end)
        return cut_path, cut_path, status

    cut_btn.click(
        on_cut,
        inputs=[audio_path_state, start_time, end_time],
        outputs=[audio_path_state, audio_player, cut_status],
    )

    def on_mode_change(mode):
        return gr.update(visible=(mode == "Text with speaker roles"))

    transcribe_mode.change(
        on_mode_change,
        inputs=[transcribe_mode],
        outputs=[hf_token],
    )

    def on_transcribe(audio_path, speakers, model, lang, mode, token):
        try:
            if not audio_path:
                yield "No audio file for transcription.", ""
                return
            yield "Loading model...", ""
            device = (
                "cuda"
                if (os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("NVIDIA_VISIBLE_DEVICES"))
                else "cpu"
            )
            transcriber = WhisperTranscriber(model_name=model, device=device)
            yield "Transcribing audio...", ""
            result = transcriber.transcribe(audio_path, language=lang)
            if mode == "Text only":
                yield result["text"], ""
                return
            yield "Diarizing speakers...", ""
            diarizer = DiarizationPipeline(token, device=device)
            diarized = diarizer.diarize(audio_path, int(speakers))
            print(diarized)
            saver = TranscriptSaver("out")
            merged = saver.merge_segments(diarized, result["segments"])
            transcript_text = "\n".join([f"Speaker {seg['speaker']}: {seg['text']}" for seg in merged])
            yield transcript_text, ""
        except Exception as e:
            yield "", f"❌ Error: {e}"

    transcribe_event = transcribe_btn.click(
        on_transcribe,
        inputs=[audio_path_state, num_speakers, whisper_model, language, transcribe_mode, hf_token],
        outputs=[transcribe_output, transcribe_status],
        queue=True,
    )

    def on_add_prompt(prompt_file, transcript):
        if not prompt_file:
            return transcript, "Please select a prompt."
        prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
            if not transcript:
                new_text = prompt_text
            else:
                new_text = prompt_text + "\n" + transcript
            return new_text, f"✅ Prompt added."
        except Exception as e:
            return transcript, f"❌ Error: {e}"

    def on_prompt_select(prompt_file):
        if not prompt_file:
            return ""
        prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""

    prompt_selector.change(
        on_prompt_select,
        inputs=[prompt_selector],
        outputs=[prompt_preview],
    )

    add_prompt_btn.click(
        on_add_prompt,
        inputs=[prompt_selector, transcribe_output],
        outputs=[transcribe_output, add_prompt_status],
    )

if __name__ == "__main__":
    demo.launch()
