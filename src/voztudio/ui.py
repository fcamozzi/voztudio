from __future__ import annotations

from pathlib import Path

from voztudio.domain import GenerationRequest
from voztudio.generation import GenerationService
from voztudio.stylemap import QWEN_STYLE_WARNING

UI_NOTICE = """# Voztudio

O idioma `pt` é um rótulo local. Nenhum backend separa português do Brasil de português europeu. O sotaque sai do áudio de referência e o idioma falado segue o texto.

A referência aceita continua de 10 a 90 segundos. Na síntese, o áudio é cortado no início da janela do modelo: 15 s no Qwen3-TTS, 20 s no VoxCPM2 e 30 s no Fish Speech. A transcrição tem de cobrir só esses primeiros segundos. Se o texto colado for mais longo, ou estiver vazio, o trecho cortado é transcrito automaticamente.

Qwen3-TTS e VoxCPM2 usam Apache-2.0. Os pesos `fishaudio/s2-pro` são só para pesquisa e uso não comercial.
"""

BACKEND_CHOICES = ["qwen3", "qwen3-fast", "voxcpm2", "fish"]


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
    pronunciations: str | None = None,
):
    if not voice_id:
        return None, "Choose a reference voice.", ""
    try:
        result = service.generate(
            GenerationRequest(
                text=text,
                language=language,
                voice_id=voice_id,
                instructions=instructions or None,
                pronunciations=pronunciations or None,
            )
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        return None, str(exc), ""
    return str(result.audio_path), "\n".join(result.warnings), result.reference_transcript or ""


def remove_voice(service: GenerationService, voice_id: str | None):
    import gradio as gr

    if not voice_id:
        return "Choose a reference voice.", gr.update()
    try:
        service.store.delete(voice_id)
    except KeyError:
        return "Voice not found.", gr.update(choices=voice_choices(service), value=None)
    return "Voice deleted.", gr.update(choices=voice_choices(service), value=None)


def backend_banner(service: GenerationService) -> str:
    engine = service.engine
    name = getattr(engine, "name", None) or "custom"
    runtime = getattr(engine, "runtime", "")
    model_id = getattr(engine, "model_id", "")
    lines = [
        f"Backend em execução: `{name}`"
        + (f" ({runtime})" if runtime else "")
        + (f", modelo `{model_id}`." if model_id else "."),
        "Para trocar, reinicie com `VOZTUDIO_BACKEND=qwen3`, `qwen3-fast`, `voxcpm2` ou `fish`. "
        "Este processo não carrega outro modelo.",
    ]
    if name in {"qwen3", "qwen3-fast"} or not getattr(engine, "supports_style", True):
        lines.append(QWEN_STYLE_WARNING)
    elif name == "voxcpm2":
        lines.append(
            "Tags como `[whisper]` viram uma instrução entre parênteses, por exemplo `(sussurrando)`. "
            "A tabela está no README."
        )
    else:
        lines.append(
            "No Fish Speech, um estilo vira uma tag. `calm` é enviado como `[calm]`. "
            "Tags que você já escreveu, como `[whisper]`, são mantidas."
        )
    check = getattr(engine, "health", None)
    if callable(check):
        ready, detail = check()
        if not ready and detail:
            lines.append(detail)
    return "\n\n".join(lines)


def style_label(service: GenerationService) -> str:
    name = getattr(service.engine, "name", "")
    if name in {"qwen3", "qwen3-fast"} or not getattr(service.engine, "supports_style", True):
        return "Estilo (ignorado). O clone Base do Qwen3-TTS remove tags como [whisper]."
    if name == "voxcpm2":
        return "Estilo. [whisper] vira (sussurrando). Texto solto vira a instrução entre parênteses."
    return "Style tag (optional). Plain text becomes one [tag]."


def backend_switch_message(choice: str | None, running: str) -> str:
    selected = choice or running
    if selected == running:
        return f"Este processo está em {running}."
    return (
        f"Este processo está em {running}. "
        f"Pare o servidor e rode VOZTUDIO_BACKEND={selected} scripts/start.sh para usar {selected}."
    )


def create_ui(service: GenerationService):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install gradio to start the local UI") from exc

    def refresh():
        return gr.update(choices=voice_choices(service))

    running = getattr(service.engine, "name", None) or "custom"
    with gr.Blocks(title="Voztudio") as demo:
        gr.Markdown(UI_NOTICE)
        gr.Markdown(backend_banner(service))
        backend = gr.Dropdown(
            choices=BACKEND_CHOICES,
            value=running if running in BACKEND_CHOICES else None,
            label="Backend",
            info="Mostra o backend deste processo. Trocar aqui não recarrega o modelo.",
        )
        audio = gr.Audio(sources=["upload"], type="filepath", label="Reference recording")
        window = getattr(service.engine, "reference_max_seconds", None)
        window_label = (
            f"Transcrição opcional só dos primeiros {float(window):g} segundos"
            if window
            else "Transcrição opcional da referência"
        )
        transcript = gr.Textbox(
            label=window_label,
            info=(
                "Tem de descrever só o começo que o modelo recebe "
                "(15 s no Qwen3, 20 s no VoxCPM2, 30 s no Fish). "
                "Um texto vazio ou longo demais é substituído pela transcrição automática desse trecho."
            ),
            lines=3,
        )
        consent = gr.Checkbox(
            value=False,
            label="I have the rights to clone this voice, or the speaker consented",
        )
        register_button = gr.Button("Register voice")
        status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            voice = gr.Dropdown(choices=voice_choices(service), label="Reference voice")
            language = gr.Dropdown(
                choices=["pt", "en", "es", "hi", "zh", "fr"],
                value="pt",
                label="Language label",
                info="Padrão pt. O rótulo é gravado com o arquivo e enviado aos modelos que aceitam idioma. pt não separa pt-BR de pt-PT. Qwen3-TTS não lista hindi.",
            )
        refresh_button = gr.Button("Refresh voices")
        delete_button = gr.Button("Delete voice")
        text = gr.Textbox(lines=8, label="Script")
        instructions = gr.Textbox(label=style_label(service))
        pronunciations = gr.Textbox(
            lines=4,
            label="Pronúncia (opcional)",
            info="Uma linha por palavra: FastHunting=Fést Rânting. Também vale pronunciations.json.",
        )
        generate_button = gr.Button("Generate")
        output = gr.Audio(label="Generated audio", type="filepath")
        used_transcript = gr.Textbox(
            label="Transcrição dos primeiros segundos usada na clonagem",
            interactive=False,
        )
        error = gr.Textbox(label="Generation", interactive=False)
        backend.change(
            lambda choice: backend_switch_message(choice, running),
            [backend],
            [status],
        )
        register_button.click(
            lambda path, words, agreed: register_reference(service, path, words, agreed),
            [audio, transcript, consent],
            [status, voice],
        )
        refresh_button.click(refresh, outputs=voice)
        delete_button.click(lambda voice_id: remove_voice(service, voice_id), [voice], [status, voice])
        generate_button.click(
            lambda voice_id, selected, script, style, spoken: synthesize(
                service, voice_id, selected, script, style, spoken
            ),
            [voice, language, text, instructions, pronunciations],
            [output, error, used_transcript],
        )
    return demo
