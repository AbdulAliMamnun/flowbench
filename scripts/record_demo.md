# Recording the 2-minute demo

Target: a short GIF or screen recording that a recruiter can watch without sound.

## Preparation

1. Run the full pipeline once so `artifacts/default/` holds a checkpoint and a report:
   ```bash
   make inspect prepare train evaluate
   ```
2. Terminal A: `make serve` (FastAPI on http://127.0.0.1:8000).
3. Terminal B: `make ui` (Streamlit on http://localhost:8501).
4. Resize the browser to 1280×800; hide bookmarks bar.

## Script (about 120 seconds)

| t     | Action                                                                 |
| ----- | ---------------------------------------------------------------------- |
| 0:00  | Show the README benchmark table for 5 seconds.                         |
| 0:05  | Switch to the Streamlit tab. Pick a held-out simulation from the list. |
| 0:15  | Point at the four panels: input, reference, prediction, error map.     |
| 0:35  | Read the per-sample metrics under the panels.                          |
| 0:45  | Pick a second simulation from the high-vorticity slice.                |
| 1:05  | Switch to a terminal: `curl` the `/health` and `/version` endpoints.   |
| 1:20  | `curl` `/predict` with a malformed body; show the structured 422.      |
| 1:40  | Back to the UI; end on the error map.                                  |

## Export

- macOS: `Cmd+Shift+5` → record selected portion → save to `docs/assets/demo.mov`.
- Convert: `ffmpeg -i docs/assets/demo.mov -vf "fps=10,scale=960:-1" docs/assets/demo.gif`.
- Keep the GIF under 5 MB; link it from the README's first section.
