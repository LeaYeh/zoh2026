"""Pre-download foundation model weights to local HuggingFace cache.

Run before the hackathon to avoid slow downloads on competition day.

Usage:
    uv run python scripts/download_models.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = [
    "amazon/chronos-t5-small",
    "amazon/chronos-t5-base",
]


def download(repo_id: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {repo_id}")
    print(f"{'='*60}")
    t0 = time.time()
    local_dir = snapshot_download(
        repo_id=repo_id,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    elapsed = time.time() - t0
    size_mb = sum(f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file()) / 1e6
    print(f"  ✓ cached → {local_dir}")
    print(f"    {size_mb:.0f} MB  |  {elapsed:.1f}s")


def verify(repo_id: str) -> bool:
    """Quick load test — catches corrupt downloads."""
    import torch
    from chronos import ChronosPipeline

    print(f"  verifying {repo_id} ...", end=" ", flush=True)
    try:
        pipe = ChronosPipeline.from_pretrained(repo_id, dtype=torch.float32, device_map="cpu")
        ctx = torch.randn(1, 64)
        out = pipe.predict(ctx, 24, num_samples=4)
        assert out.shape == (1, 4, 24), f"unexpected shape {out.shape}"
        print("ok")
        return True
    except Exception as exc:
        print(f"FAIL — {exc}")
        return False


def main() -> None:
    print("Downloading time-series foundation models")
    print("(weights are cached in ~/.cache/huggingface)\n")

    for model in MODELS:
        download(model)

    print("\n\nVerifying loaded inference …")
    failures = [m for m in MODELS if not verify(m)]

    if failures:
        print(f"\n[FAIL] {len(failures)} model(s) failed verification: {failures}")
        sys.exit(1)
    else:
        print(f"\n[OK] All {len(MODELS)} models ready.")


if __name__ == "__main__":
    main()
