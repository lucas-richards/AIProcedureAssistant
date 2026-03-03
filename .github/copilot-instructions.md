# Copilot Instructions

## Project startup behavior

When the user says **"start app"** (or equivalent), always run the backend using the project virtual environment at `venv/`.

Use this exact command sequence from the workspace root:

```bash
source venv/bin/activate && cd backend && uvicorn app:app --host 0.0.0.0 --port 8000
```

## Notes

- Do not use `.venv`; use `venv`.
- Do not run `python app.py` to start the server.
- If `uvicorn` is missing, report that dependency is not installed in the active `venv`.
