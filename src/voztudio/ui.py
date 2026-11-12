from __future__ import annotations

from voztudio.generation import GenerationService


def create_ui(service: GenerationService):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install gradio to start the local UI") from exc

    def voices():
        return [(record.id, record.id) for record in service.store.list()]

    def generate(voice_id, language, text, instructions):
        from voztudio.domain import GenerationRequest

        result = service.generate(
            GenerationRequest(
                text=text,
                language=language,
                voice_id=voice_id,
                instructions=instructions or None,
            )
        )
        return str(result.audio_path)

    with gr.Blocks(title="Voztudio") as demo:
        gr.Markdown("# Voztudio\nLocal Fish Speech voice generation")
        with gr.Row():
            voice = gr.Dropdown(choices=voices(), label="Reference voice")
            language = gr.Dropdown(
                choices=["en", "pt", "es", "hi", "zh", "fr"],
                value="en",
                label="Language",
            )
        text = gr.Textbox(lines=8, label="Script")
        instructions = gr.Textbox(label="Style/instructions (optional)")
        generate_button = gr.Button("Generate")
        output = gr.Audio(label="Generated audio", type="filepath")
        generate_button.click(generate, [voice, language, text, instructions], output)
    return demo
