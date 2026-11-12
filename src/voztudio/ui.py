from __future__ import annotations

from pathlib import Path

from voztudio.domain import GenerationRequest
from voztudio.generation import GenerationService

UI_NOTICE = """# Voztudio
Built with Fish Audio.

O idioma `pt` é um rótulo local. Este build do Fish Speech não recebe o idioma e não separa português do Brasil de português europeu. O sotaque sai do áudio de referência.

Style becomes one inline tag. `calm` is sent as `[calm]`. Tags you already wrote, such as `[whisper]`, are kept.

Recommended reference: 10–30 seconds of clear speech and the exact transcript. Accepted length is 10–90 seconds.
"""


def voice_choices(service: GenerationService) -> list[tuple[str, str]]:
    return [(record.id, record.id) for record in service.store.list()]


def register_reference(service: GenerationService, path: str | None, transcript: str | None, consent: bool):
    import gradio as gr

    if not path:
        return "Upload a reference recording.", gr.update()
    try:
        record = service.store.register(Path(path), transcript, consent=bool(consent))
    except ValueError as exc:
        return str(exc), gr.update()
    return (
        f"Registered voice {record.id}.",
        gr.update(choices=voice_choices(service), value=record.id),
    )


def synthesize(
    service: GenerationService,
    voice_id: str | None,
    language: str,
    text: str,
    instructions: str | None,
):
    if not voice_id:
        return None, "Choose a reference voice."
    try:
        result = service.generate(
            GenerationRequest(
                text=text,
                language=language,
                voice_id=voice_id,
                instructions=instructions or None,
            )
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        return None, str(exc)
    return str(result.audio_path), ""


def remove_voice(service: GenerationService, voice_id: str | None):
    import gradio as gr

    if not voice_id:
        return "Choose a reference voice.", gr.update()
    try:
        service.store.delete(voice_id)
    except KeyError:
        return "Voice not found.", gr.update(choices=voice_choices(service), value=None)
    return "Voice deleted.", gr.update(choices=voice_choices(service), value=None)


def create_ui(service: GenerationService):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install gradio to start the local UI") from exc

    def refresh():
        return gr.update(choices=voice_choices(service))

    with gr.Blocks(title="Voztudio") as demo:
        gr.Markdown(UI_NOTICE)
        audio = gr.Audio(sources=["upload"], type="filepath", label="Reference recording")
        transcript = gr.Textbox(label="Exact transcript of the reference")
        consent = gr.Checkbox(
            value=False,
            label="I have the rights to clone this voice, or the speaker consented",
        )
        register_button = gr.Button("Register voice")
        status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            voice = gr.Dropdown(choices=voice_choices(service), label="Reference voice")
            language = gr.Dropdown(
                choices=["en", "pt", "es", "hi", "zh", "fr"],
                value="en",
                label="Language label",
                info="Stored with the file. Not sent to Fish Speech. pt is not pt-BR.",
            )
        refresh_button = gr.Button("Refresh voices")
        delete_button = gr.Button("Delete voice")
        text = gr.Textbox(lines=8, label="Script")
        instructions = gr.Textbox(
            label="Style tag (optional). Plain text becomes one [tag].",
        )
        generate_button = gr.Button("Generate")
        output = gr.Audio(label="Generated audio", type="filepath")
        error = gr.Textbox(label="Generation", interactive=False)
        register_button.click(
            lambda path, words, agreed: register_reference(service, path, words, agreed),
            [audio, transcript, consent],
            [status, voice],
        )
        refresh_button.click(refresh, outputs=voice)
        delete_button.click(lambda voice_id: remove_voice(service, voice_id), [voice], [status, voice])
        generate_button.click(
            lambda voice_id, selected, script, style: synthesize(service, voice_id, selected, script, style),
            [voice, language, text, instructions],
            [output, error],
        )
    return demo
