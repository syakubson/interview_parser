from code.diarization import DiarizationPipeline
from code.gradio_utils import (
    add_prompt_to_text,
    cut_audio,
    extract_audio_from_video,
    get_device,
    get_device_name,
    list_prompts,
    read_prompt,
)
from code.output_utils import TranscriptSaver
from code.transcribe import WhisperTranscriber

import gradio as gr

LANGUAGES = [
    ("Русский", "ru"),
    ("English", "en"),
]

PROMPT_FILES = list_prompts()


def build_load_cut_tab():
    with gr.Tab("Load & Cut"):
        device_info = gr.Markdown(f"**Device:** {get_device_name()}")
        video_file = gr.File(label="Upload video file", type="filepath", elem_classes=["compact-upload"])
        extract_btn = gr.Button("Extract Audio", elem_classes=["compact-btn"])
        extract_status = gr.Markdown("")
        with gr.Row():
            start_time = gr.Number(label="Cut Start (s)", value=0, precision=2, scale=1)
            end_time = gr.Number(label="Cut End (s)", value=60, precision=2, scale=1)
        cut_btn = gr.Button("Cut Audio by Interval", elem_classes=["compact-btn"])
        cut_status = gr.Markdown("")
        audio_path_state = gr.State()
        return dict(
            device_info=device_info,
            video_file=video_file,
            extract_btn=extract_btn,
            extract_status=extract_status,
            start_time=start_time,
            end_time=end_time,
            cut_btn=cut_btn,
            cut_status=cut_status,
            audio_path_state=audio_path_state,
        )


def build_extraction_settings_tab():
    with gr.Tab("Extraction Settings"):
        with gr.Row():
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
        num_speakers = gr.Number(label="Speakers", value=2, precision=0, scale=1, visible=True)
        hf_token = gr.Textbox(
            label="HuggingFace token for Pyannote",
            type="password",
            elem_classes=["compact-token"],
            visible=True,
        )
        with gr.Row():
            transcribe_btn = gr.Button("Transcribe", elem_classes=["compact-btn"])
        transcribe_status = gr.Markdown("")
        return dict(
            num_speakers=num_speakers,
            whisper_model=whisper_model,
            language=language,
            transcribe_mode=transcribe_mode,
            hf_token=hf_token,
            transcribe_btn=transcribe_btn,
            transcribe_status=transcribe_status,
        )


def build_add_prompts_tab():
    with gr.Tab("Add prompts"):
        prompt_selector = gr.Dropdown(PROMPT_FILES, label="Select prompt")
        prompt_preview = gr.Textbox(label="Prompt preview", interactive=False, lines=6)
        add_prompt_btn = gr.Button("Add prompt to transcript")
        add_prompt_status = gr.Markdown("")
        return dict(
            prompt_selector=prompt_selector,
            prompt_preview=prompt_preview,
            add_prompt_btn=add_prompt_btn,
            add_prompt_status=add_prompt_status,
        )


def build_output_column():
    with gr.Column(scale=3):
        audio_player = gr.Audio(label="Audio Preview", interactive=False, show_download_button=True, scale=3)
        transcribe_output = gr.Textbox(label="Transcript", lines=10, show_copy_button=True)
        return dict(
            audio_player=audio_player,
            transcribe_output=transcribe_output,
        )


# --- Gradio UI Construction ---
with gr.Blocks() as demo:
    gr.Markdown("# Interview Parser")
    device_status = gr.State(get_device_name())
    with gr.Row():
        with gr.Column(scale=1):
            # Секции интерфейса
            load_cut = build_load_cut_tab()
            extraction = build_extraction_settings_tab()
            add_prompts = build_add_prompts_tab()
        output = build_output_column()

    def on_extract(video: str) -> tuple:
        # Extract audio from video and return audio path and status
        audio_path, status = extract_audio_from_video(video)
        return audio_path, audio_path, status

    load_cut["extract_btn"].click(
        on_extract,
        inputs=[load_cut["video_file"]],
        outputs=[load_cut["audio_path_state"], output["audio_player"], load_cut["extract_status"]],
    )

    def on_cut(audio_path: str, start: float, end: float) -> tuple:
        # Cut audio by interval and return cut audio path and status
        cut_path, status = cut_audio(audio_path, start, end)
        return cut_path, cut_path, status

    load_cut["cut_btn"].click(
        on_cut,
        inputs=[load_cut["audio_path_state"], load_cut["start_time"], load_cut["end_time"]],
        outputs=[load_cut["audio_path_state"], output["audio_player"], load_cut["cut_status"]],
    )

    def on_mode_change(mode: str) -> tuple:
        # Show or hide HuggingFace token and Speakers field depending on transcription mode
        visible = mode == "Text with speaker roles"
        return gr.update(visible=visible), gr.update(visible=visible)

    extraction["transcribe_mode"].change(
        on_mode_change,
        inputs=[extraction["transcribe_mode"]],
        outputs=[extraction["hf_token"], extraction["num_speakers"]],
    )

    def on_transcribe(
        audio_path: str,
        speakers: int,
        model: str,
        lang: str,
        mode: str,
        token: str,
    ) -> tuple:
        # Transcribe audio and (optionally) diarize speakers, yielding progress/status
        try:
            if not audio_path:
                yield "No audio file for transcription.", ""
                return
            yield "Loading model...", ""
            device = get_device()
            transcriber = WhisperTranscriber(model_name=model, device=device)
            yield "Transcribing audio...", ""
            result = transcriber.transcribe(audio_path, language=lang)
            if mode == "Text only":
                yield result["text"], ""
                return
            yield "Diarizing speakers...", ""
            diarizer = DiarizationPipeline(token, device=device)
            diarized = diarizer.diarize(audio_path, int(speakers))
            saver = TranscriptSaver("out")
            merged = saver.merge_segments(diarized, result["segments"])
            transcript_text = "\n".join([f"Speaker {seg['speaker']}: {seg['text']}" for seg in merged])
            yield transcript_text, ""
        except Exception as e:
            yield "", f"❌ Error: {e}"

    extraction["transcribe_btn"].click(
        on_transcribe,
        inputs=[
            load_cut["audio_path_state"],
            extraction["num_speakers"],
            extraction["whisper_model"],
            extraction["language"],
            extraction["transcribe_mode"],
            extraction["hf_token"],
        ],
        outputs=[output["transcribe_output"], extraction["transcribe_status"]],
        queue=True,
    )

    def on_add_prompt(prompt_file: str, transcript: str) -> tuple:
        # Add selected prompt to the beginning of the transcript
        if not prompt_file:
            return transcript, "Please select a prompt."
        try:
            prompt_text = read_prompt(prompt_file)
            new_text = add_prompt_to_text(prompt_text, transcript)
            return new_text, f"✅ Prompt added."
        except Exception as e:
            return transcript, f"❌ Error: {e}"

    def on_prompt_select(prompt_file: str) -> str:
        # Show prompt text in preview window when prompt is selected
        if not prompt_file:
            return ""
        try:
            return read_prompt(prompt_file)
        except Exception:
            return ""

    add_prompts["prompt_selector"].change(
        on_prompt_select,
        inputs=[add_prompts["prompt_selector"]],
        outputs=[add_prompts["prompt_preview"]],
    )

    add_prompts["add_prompt_btn"].click(
        on_add_prompt,
        inputs=[add_prompts["prompt_selector"], output["transcribe_output"]],
        outputs=[output["transcribe_output"], add_prompts["add_prompt_status"]],
    )

if __name__ == "__main__":
    demo.launch()
