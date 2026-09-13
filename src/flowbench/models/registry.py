"""Model registry: config name → constructor.

The FNO extension point lives here: a future ``fno`` entry would register a
constructor with the same ``(ModelConfig) -> Predictor`` signature and nothing else in
the pipeline would change.
"""

from __future__ import annotations
