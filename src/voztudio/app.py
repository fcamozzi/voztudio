from __future__ import annotations

import uvicorn

from voztudio.api import create_app
from voztudio.config import get_settings
from voztudio.ui import create_ui


def main() -> None:
    settings = get_settings()
    app = create_app(settings)
    try:
        import gradio as gr

        gr.mount_gradio_app(app, create_ui(app.state.service), path="/ui")
    except RuntimeError:
        pass
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
