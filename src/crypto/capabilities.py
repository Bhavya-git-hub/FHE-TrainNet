"""What the installed FHE library can actually do, established by calling it.

Master Prompt 2 and Technical Design 11 both forbid asserting a cryptographic
capability we have not demonstrated. Every field below is therefore filled by
`probe()`, which invokes the real API and records whether it succeeded. Nothing
here is a literal copied from documentation: if TenSEAL is upgraded and gains a
`scale()` accessor or loses `polyval`, the probe result changes and the README,
the dashboard and `docs/LIBRARY_CAPABILITIES.md` change with it.
"""

from __future__ import annotations

import math
import platform
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable


class ProbeResult(str, Enum):
    """Outcome of exercising one API.

    `UNSUPPORTED` and `ERROR` are deliberately distinct. The first means the
    library has no such operation; the second means it has one and it failed,
    which is a defect we want to see rather than a limitation we accept.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


@dataclass(frozen=True)
class Probe:
    """One attempted operation and what came back."""

    name: str
    result: ProbeResult
    detail: str = ""
    evidence: str = ""

    @property
    def ok(self) -> bool:
        return self.result is ProbeResult.SUPPORTED


@dataclass
class Capabilities:
    """The probed capability set of a crypto backend.

    `refresh_mechanism` is the field the rest of the system keys off. It is the
    single source of truth for whether the dashboard may use the word
    "bootstrapping" (see `src.crypto.backend.RefreshKind`).
    """

    backend: str
    library: str
    library_version: str
    scheme: str
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    probes: list[Probe] = field(default_factory=list)
    refresh_mechanism: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def add(self, probe: Probe) -> None:
        self.probes.append(probe)

    def get(self, name: str) -> Probe | None:
        return next((p for p in self.probes if p.name == name), None)

    def supports(self, name: str) -> bool:
        probe = self.get(name)
        return probe is not None and probe.ok

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["probes"] = [
            {"name": p.name, "result": p.result.value, "detail": p.detail, "evidence": p.evidence}
            for p in self.probes
        ]
        return data


def attempt(name: str, fn: Callable[[], Any], *, evidence_fmt: str = "{!r}") -> Probe:
    """Call `fn` and classify the outcome.

    A missing attribute or a `NotImplementedError` means the library does not
    offer the operation; anything else that raises is recorded as an error with
    its message, because a silent `except: pass` here is precisely the
    "swallow the reason" failure the specification warns against.
    """
    try:
        value = fn()
    except (AttributeError, TypeError, NotImplementedError) as exc:
        return Probe(name, ProbeResult.UNSUPPORTED, f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - reason is preserved in `detail` below
        return Probe(name, ProbeResult.ERROR, f"{type(exc).__name__}: {exc}")
    return Probe(name, ProbeResult.SUPPORTED, evidence=evidence_fmt.format(value))


def precision_bits(actual: float, expected: float) -> float | None:
    """Bits of agreement between a decrypted value and the exact one.

    Returns `None` when the two are bit-identical, because "infinitely precise"
    is not a number we may put on a chart. Missing stays missing rather than
    becoming a convenient large constant.
    """
    if expected == 0:
        return None
    error = abs(actual - expected) / abs(expected)
    if error == 0:
        return None
    return -math.log2(error)
