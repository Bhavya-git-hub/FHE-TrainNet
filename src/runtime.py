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

# Below this, use the low-memory profile.
#
# The number is the batched profile's *measured peak resident size*, not its key
# size, plus room to be wrong. A demo run recorded 2196 MB peak RSS - the 1901 MB
# of Galois rotation keys is most of it, but not all of it, and an earlier
# version of this file used the key figure alone and set the bar at 1500 MB. That
# let a container reporting 2 GB select a profile that needs 2.2 GB, which is a
# kill part-way through a demonstration rather than an error anyone can read.
LOW_MEMORY_THRESHOLD_MB = 3000

# Measured peak RSS of the batched profile, for the message in `profile_note`.
BATCHED_PROFILE_PEAK_MB = 2196

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


# Containers generally advertise themselves through `/.dockerenv` or their own
# cgroup names.
_CONTAINER_MARKERS = ("docker", "kubepods", "containerd", "lxc")

# Streamlit Community Cloud checks the repository out under `/mount/src`; the
# traceback from a failed deployment shows the path.
_STREAMLIT_CLOUD_MOUNT = Path("/mount/src")


def on_streamlit_community_cloud() -> bool:
    """Whether this is the free Streamlit tier, whose allowance we know."""
    return _STREAMLIT_CLOUD_MOUNT.is_dir()


def looks_containerised() -> bool:
    """Whether this process is running somewhere its memory is not its own."""
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False
    return any(marker in cgroup for marker in _CONTAINER_MARKERS)


def looks_hosted() -> bool:
    """Either of the above. Kept as one name because callers ask one question."""
    return on_streamlit_community_cloud() or looks_containerised()


def default_config_name() -> str:
    """Which shipped configuration to open the dashboard with.

    An explicit `FHE_TRAINNET_CONFIG` always wins, so a deployment can pin a
    profile regardless of what the detection concludes.
    """
    override = os.environ.get("FHE_TRAINNET_CONFIG", "").strip()
    if override:
        return override

    # Community Cloud first, and ahead of any measurement, because on that tier
    # no measurement available to this process is trustworthy. The enforced
    # allowance is roughly 1 GB, but the cgroup file - when it is readable at all
    # - can report the far larger figure of whatever the container runs inside.
    # An earlier version of this function read the cgroup first and was killed
    # anyway: it believed a number that was not the limit being enforced.
    #
    # A deployment that really does have the memory says so with
    # FHE_TRAINNET_CONFIG, which is checked above and wins.
    if on_streamlit_community_cloud():
        return "cloud"

    limit = cgroup_memory_limit_mb()
    if limit is not None:
        return "cloud" if limit < LOW_MEMORY_THRESHOLD_MB else "demo"

    # Some other container, with no limit to read. That is not evidence of a
    # generous allowance - it is the absence of evidence, and `psutil` will
    # happily answer with the host's memory, which is not ours to use. Guessing
    # high kills the process; guessing low costs throughput and says so on
    # screen. The asymmetry decides it.
    if looks_containerised():
        return "cloud"

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
    limit = cgroup_memory_limit_mb()
    if limit is not None:
        seen = f"a {limit:.0f} MB container limit"
    elif looks_hosted():
        seen = "no container memory limit it is allowed to read"
    else:
        memory = available_memory_mb()
        seen = f"{memory:.0f} MB" if memory is not None else "an unknown amount"
    return (
        f"Running the **low-memory profile**: this process sees {seen}, "
        f"against the {LOW_MEMORY_THRESHOLD_MB} MB needed for the batched profile "
        f"(measured peak {BATCHED_PROFILE_PEAK_MB} MB, mostly Galois rotation keys "
        "at 1901 MB). The single-slot profile runs the "
        "same algorithm at the same ring dimension and scale for about 105 MB, "
        "processing one sample per step instead of a batch. Everything shown is real; "
        "only the throughput is smaller."
    )


def profile_diagnostics() -> dict[str, str]:
    """Every observation the profile decision was made from, and the decision.

    This exists because a deployed process cannot be interrogated. When the
    hosted app kept being killed there was no way to tell whether the detection
    had picked the wrong profile or the right profile was still too large, and
    the difference decides what to fix. Reporting the inputs alongside the
    conclusion turns that from a guess into a reading.
    """
    override = os.environ.get("FHE_TRAINNET_CONFIG", "").strip()
    limit = cgroup_memory_limit_mb()
    try:
        import psutil

        psutil_mb = f"{psutil.virtual_memory().total / 1e6:.0f} MB"
    except Exception:  # noqa: BLE001 - absence is reported as unknown
        psutil_mb = "unavailable"
    return {
        "Selected profile": default_config_name(),
        "FHE_TRAINNET_CONFIG": override or "not set",
        "Streamlit Community Cloud": str(on_streamlit_community_cloud()),
        "Containerised": str(looks_containerised()),
        "cgroup memory limit": f"{limit:.0f} MB" if limit is not None else "not readable",
        "psutil total (the host's, not ours)": psutil_mb,
        "Batched profile needs": f"{LOW_MEMORY_THRESHOLD_MB} MB "
                                 f"(measured peak {BATCHED_PROFILE_PEAK_MB} MB)",
    }
