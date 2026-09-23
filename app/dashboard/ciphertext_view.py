"""Pure helpers for showing a ciphertext as bytes.

Separate from `Walkthrough.py` because that file is a Streamlit *main script*:
importing it executes the page, so anything defined inside it cannot be tested
without a Streamlit runtime. These two functions carry the only logic on that
page worth asserting against, so they live where a test can reach them.

New file rather than an addition to `common.py`, which `main` owns - this branch
stays additive so it cannot conflict and can be abandoned without cleanup.
"""

from __future__ import annotations

import numpy as np


def hex_dump(data: bytes, *, start: int = 0, width: int = 16) -> str:
    """Classic offset / hex / printable-ASCII dump of a slice of bytes.

    `start` is the offset the slice was *taken from*, not an index into it, so a
    window read from the middle of a ciphertext reports where it truly sits in
    the file. Labelling the payload window as offset 0 would leave a reader
    comparing it against another ciphertext's header without knowing it.
    """
    lines: list[str] = []
    for offset in range(0, len(data), width):
        chunk = data[offset : offset + width]
        # Padded so a short final row keeps its ASCII column aligned with the
        # rows above rather than sliding left underneath the hex.
        hexed = " ".join(f"{b:02x}" for b in chunk).ljust(width * 3 - 1)
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{start + offset:08x}  {hexed}  |{text}|")
    return "\n".join(lines)


def byte_entropy(histogram: np.ndarray) -> float:
    """Shannon entropy in bits per byte, from a 256-bin count of byte values.

    Intended to be computed over a whole ciphertext and nowhere else. A 256-byte
    window holds 256 draws across 256 bins, so even perfectly uniform bytes score
    only about 7.2 there; reporting a per-window figure would invite a comparison
    that the sample size decides rather than the cryptography.
    """
    counts = np.asarray(histogram)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())
