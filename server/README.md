# VibeVoice Server

Run a FastAPI server alongside the current app.

## Endpoints
- GET /health
- GET /models
- GET /templates/{kind}
- POST /templates
- GET /templates/{kind}/{name}
- PUT /templates/{kind}/{name}
- DELETE /templates/{kind}/{name}
- POST /audio/edit
- WS  /ws/stream

## Environment
- VIBEVOICE_MODELS: e.g.
```
microsoft/VibeVoice-1.5B|VibeVoice 1.5B;DevParker/VibeVoice7b-low-vram (4-bit)|VibeVoice 7B 4-bit
```

## Start
Install deps (see project root instructions), then run:
```
uvicorn server.app.main:app --host 0.0.0.0 --port 8081 --reload
```

