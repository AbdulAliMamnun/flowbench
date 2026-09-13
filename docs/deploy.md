# Deploying the viewer to Streamlit Community Cloud

The hosted viewer runs the same `src/flowbench/ui/app.py` as `flowbench ui`, but on a
CPU-only host without `data/` or `artifacts/`. It therefore falls back to the committed
`demo/` bundle (trained checkpoint, persistence checkpoint, 100 seeded held-out pairs,
hash manifest) and shows a banner saying so. Local behaviour with full data is unchanged.

## What is committed for the demo

| Path                                  | Purpose                                           |
| ------------------------------------- | ------------------------------------------------- |
| `demo/MANIFEST.json`                  | Provenance, subset IDs, slice threshold, SHA-256s |
| `demo/test_subset.pt`                 | `x`, `y`, `instance_ids` for 100 test pairs        |
| `demo/checkpoints/cnn/`               | The trained checkpoint directory, unchanged       |
| `demo/checkpoints/persistence/`       | Zero-parameter baseline with the same statistics  |
| `demo/metrics.json`                   | Copy of the full-run report for the table         |
| `requirements.txt`                    | Minimal CPU install for the cloud builder         |

Regenerate after retraining:

```bash
make train evaluate        # or the full quickstart
make export-demo           # rewrites demo/ and fails if it exceeds demo.max_bytes (2 MB)
make requirements          # only needed when uv.lock changed
git add demo requirements.txt && git commit -m "chore(demo): refresh demo bundle"
```

`tests/integration/test_demo_bundle.py` checks that the bundle verifies against its
manifest and predicts exactly like the full checkpoint on matching samples.

## Steps on Streamlit Community Cloud

1. Push `main` to GitHub (`AbdulAliMamnun/flowbench`) with `demo/` and
   `requirements.txt` committed.
2. Sign in at https://share.streamlit.io with the GitHub account that owns the repo and
   click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository**: `AbdulAliMamnun/flowbench`
   - **Branch**: `main`
   - **Main file path**: `src/flowbench/ui/app.py`
   - **App URL**: pick a slug, e.g. `flowbench`
4. Open **Advanced settings** and set **Python version** to `3.12`. No secrets are needed.
5. Click **Deploy**. The builder installs `requirements.txt`: it pulls the CPU torch
   wheel from the PyTorch index via the `--extra-index-url` line, then `-e .` installs
   `flowbench` and its declared dependencies. Expect 5 to 10 minutes on the first build.
6. When the app starts, the banner should read "Running on the 100-sample demo subset".
   The sidebar caption ends with `mode: demo` and the device shows `cpu`.

## Notes and limits

- The app is launched without `--config`, so `configs/default.yaml` is used; the
  repository root is the working directory, so `demo/` resolves correctly.
- `run.device: auto` resolves to `cpu` whenever MPS is unavailable; nothing else needs
  changing for the hosted tier.
- The bundle is verified against its manifest on every cold start; a hash mismatch stops
  the app with an error rather than serving altered weights.
- Community Cloud gives about 1 GB of RAM. The CNN has 19 105 parameters and the subset
  is under 1 MB, so memory is dominated by importing torch; that fits.
- The benchmark table on the hosted page is a copy of the full-run `metrics.json`; the
  per-sample metrics are computed live on the host.
- After a `git push` to `main`, Community Cloud redeploys automatically.
