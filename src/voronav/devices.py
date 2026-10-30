"""Device selection shared by every PyTorch path (JFA and SAM 2).

Set VORONAV_DISABLE_MPS=1 to treat Apple's Metal (MPS) backend as absent.
CI needs this: GitHub's macOS runners are virtual machines whose virtualized
GPU reports MPS as available but fails even tiny allocations ("MPS backend
out of memory" at ~50 KiB), so the MPS tests run on real Apple hardware only.
"""

from __future__ import annotations

import os

DISABLE_MPS_ENV = "VORONAV_DISABLE_MPS"


def mps_enabled() -> bool:
    if os.environ.get(DISABLE_MPS_ENV, "").strip() not in ("", "0"):
        return False
    import torch

    return torch.backends.mps.is_available()
