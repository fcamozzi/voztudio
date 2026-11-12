from __future__ import annotations

from dataclasses import replace

import uvicorn

from voztudio.api import create_app
from voztudio.config import ensure_bind_allowed, get_settings, normalize_host
from voztudio.ui import create_ui


def main() -> None:
    settings = get_settings()
    host = normalize_host(settings.host)
    ensure_bind_allowed(host, settings.allow_public_bind)
    if host != settings.host:
        settings = replace(settings, host=host)
    app = create_app(settings)
    try:
        import gradio as gr

        gr.mount_gradio_app(app, create_ui(app.state.service), path="/ui")
    except RuntimeError:
        pass
    uvicorn.run(app, host=host, port=settings.port)


if __name__ == "__main__":
    main()
