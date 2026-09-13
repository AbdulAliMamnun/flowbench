---
name: Bug report
about: Something in the pipeline, API or UI does not behave as documented
title: "bug: "
labels: bug
assignees: ""
---

## What happened

<!-- One or two sentences. -->

## What you expected

## Steps to reproduce

```bash
# exact commands, including the config file
uv sync --frozen
uv run flowbench <step> --config configs/<file>.yaml
```

## Environment

- FlowBench version (`uv run flowbench --version`):
- OS / chip:
- Device used (`mps` / `cpu`):
- Python (`uv run python --version`):

## Attachments

- [ ] The config file used
- [ ] `data/manifest.json` (for data issues)
- [ ] The relevant log lines (JSON, from stderr)
