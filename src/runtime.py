"""Work out how much room this process actually has, and pick a profile to fit.

The dashboard ships two profiles. The local one uses a mini-batch, which needs
Galois rotation keys and measured 1901 MB; the cloud one uses a batch of one,
which needs no rotation keys and measured 105 MB for the same ring dimension and
scale. Running the first inside a 1 GB hosting tier gets the process killed
part-way through a demonstration, which looks like a broken project rather than
an exhausted container.

So the profile is chosen from the memory the process can actually see, not from
a guess about where it is running. On Linux containers that means the cgroup
limit: `psutil.virtual_memory()` reports the *host's* memory, which on a hosted
tier is wildly larger than what the container is allowed to use, and trusting it
would pick exactly the profile that gets killed.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Below this, use the low-memory profile. The batched profile needs ~1.9 GB for
# its rotation keys alone, so anything under roughly 1.5 GB cannot run it.
LOW_MEMORY_THRESHOLD_MB = 1500

# cgroup v2 then v1. A container's real limit lives here; /proc/meminfo and
# psutil both report the host.
_CGROUP_LIMIT_FILES = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)

# Containers with no limit set report a sentinel close to 2**63; anything above
# this is "unlimited", not a real allowance.
_UNLIMITED_ABOVE_BYTES = 1 << 62


def cgroup_memory_limit_mb() -> float | None:
    """The container's memory allowance, or `None` when there is no limit set."""
    for path in _CGROUP_LIMIT_FILES:
        try:
            raw = Path(path).read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= _UNLIMITED_ABOVE_BYTES:
            return None
        return value / 1e6
    return None


def available_memory_mb() -> float | None:
    """Memory this process may use, preferring the container limit.

    `None` when it cannot be determined - reported as unknown rather than
    guessed, so a caller can decide what to do about not knowing.
    """
    limit = cgroup_memory_limit_mb()
    if limit is not None:
        return limit
    try:
        import psutil

        return psutil.virtual_memory().total / 1e6
    except Exception:  # noqa: BLE001 - absence is reported as unknown
        return None


def default_config_name() -> str:
    """Which shipped configuration to open the dashboard with.

    An explicit `FHE_TRAINNET_CONFIG` always wins, so a deployment can pin a
    profile regardless of what the detection concludes.
    """
    override = os.environ.get("FHE_TRAINNET_CONFIG", "").strip()
    if override:
        return override

    memory = available_memory_mb()
    if memory is not None and memory < LOW_MEMORY_THRESHOLD_MB:
        return "cloud"
    return "demo"


def default_config_path() -> Path:
    return ROOT / "configs" / f"{default_config_name()}.yaml"


def profile_note() -> str | None:
    """A line for the interface when the low-memory profile was chosen for you.

    Returns `None` when the full profile is in use, so nothing is said when there
    is nothing to explain.
    """
    if default_config_name() != "cloud":
        return None
    memory = available_memory_mb()
    seen = f"{memory:.0f} MB" if memory is not None else "an unknown amount"
    return (
        f"Running the **low-memory profile**: this process can see {seen} of memory, "
        f"below the {LOW_MEMORY_THRESHOLD_MB} MB the batched profile needs for its "
        "Galois rotation keys (measured at 1901 MB). The single-slot profile runs the "
        "same algorithm at the same ring dimension and scale for about 105 MB, "
        "processing one sample per step instead of a batch. Everything shown is real; "
        "only the throughput is smaller."
    )
