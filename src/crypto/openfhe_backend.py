"""Host side of the OpenFHE backend: detect, build, run, and never pretend.

This module's most important behaviour is what it does when it cannot work. If
Docker is not installed, not running, or the image will not build, it reports
`available=False` with the reason. It has no fallback path. A silent downgrade to
the TenSEAL backend would mean the dashboard showing "ACTUAL CKKS BOOTSTRAPPING"
over results produced by a client-aided refresh, which is the single most
damaging thing this project could do.

The container is invoked once per experiment and returns a JSON document. It is
not a per-operation bridge: shipping megabyte ciphertexts across a pipe for every
multiplication would measure the pipe rather than the cryptography.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.crypto.backend import RefreshKind

ROOT = Path(__file__).resolve().parents[2]
IMAGE_TAG = "fhe-trainnet-openfhe:1.5.1"
DOCKERFILE_DIR = ROOT / "docker" / "openfhe"
SCRIPT_IN_CONTAINER = "docker/openfhe/openfhe_experiment.py"


@dataclass
class BackendStatus:
    """Whether this backend can run, and if not, precisely why."""

    available: bool
    reason: str = ""
    docker_version: str | None = None
    image_present: bool = False
    openfhe_version: str | None = None
    supports_native_bootstrap: bool = False
    probe: dict[str, Any] = field(default_factory=dict)

    @property
    def refresh_kind(self) -> RefreshKind:
        """Never `NATIVE_BOOTSTRAP` unless the probe actually found it."""
        if self.available and self.supports_native_bootstrap:
            return RefreshKind.NATIVE_BOOTSTRAP
        return RefreshKind.UNSUPPORTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "openfhe_docker",
            "available": self.available,
            "reason": self.reason,
            "docker_version": self.docker_version,
            "image_present": self.image_present,
            "openfhe_version": self.openfhe_version,
            "supports_native_bootstrap": self.supports_native_bootstrap,
            "refresh_kind": self.refresh_kind.value,
            "refresh_kind_label": self.refresh_kind.label,
            "probe": self.probe,
        }


def _run(args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Run a docker command and capture its output as UTF-8.

    The encoding is explicit because `text=True` alone decodes with the platform's
    preferred codec - cp1252 on Windows - and Docker's build output is UTF-8 with
    progress glyphs in it. That combination raises
    `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81`, which killed a
    build that had in fact succeeded.
    """
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, check=False,
        encoding="utf-8", errors="replace",
    )


def docker_status(timeout: float = 30.0) -> BackendStatus:
    """Is there a Docker daemon we can actually reach?"""
    if shutil.which("docker") is None:
        return BackendStatus(
            available=False,
            reason="Docker is not installed or not on PATH. The OpenFHE wheel is published "
                   "for Linux and macOS only, so this backend needs a container on Windows.",
        )
    try:
        proc = _run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=timeout)
    except subprocess.TimeoutExpired:
        return BackendStatus(
            available=False,
            reason=f"`docker info` did not respond within {timeout:.0f}s. Docker Desktop may "
                   "be starting; try again shortly.",
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        return BackendStatus(
            available=False,
            reason="The Docker daemon is not reachable. Start Docker Desktop and try again. "
                   f"Docker said: {detail[0] if detail else 'no detail'}",
        )
    version = proc.stdout.strip()

    images = _run(["docker", "images", "-q", IMAGE_TAG], timeout=timeout)
    present = bool(images.stdout.strip())
    return BackendStatus(
        available=True,
        reason="" if present else f"Image {IMAGE_TAG} not built yet; run build_image().",
        docker_version=version,
        image_present=present,
    )


def build_image(*, timeout: float = 1800.0, progress: Any = None) -> BackendStatus:
    """Build the OpenFHE image. Slow the first time; cached afterwards."""
    status = docker_status()
    if not status.available:
        return status
    if status.image_present:
        return probe(status)

    if progress:
        progress(f"Building {IMAGE_TAG} (first build downloads Ubuntu and OpenFHE)...")
    started = time.perf_counter()
    try:
        proc = _run(
            ["docker", "build", "-t", IMAGE_TAG, str(DOCKERFILE_DIR)], timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return BackendStatus(
            available=False,
            docker_version=status.docker_version,
            reason=f"Image build exceeded {timeout / 60:.0f} minutes.",
        )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        return BackendStatus(
            available=False,
            docker_version=status.docker_version,
            reason="Image build failed: " + " | ".join(tail),
        )
    if progress:
        progress(f"Built in {time.perf_counter() - started:.0f}s")
    return probe(docker_status())


def probe(status: BackendStatus | None = None, *, timeout: float = 300.0) -> BackendStatus:
    """Ask the container what its OpenFHE build supports."""
    status = status or docker_status()
    if not status.available:
        return status
    if not status.image_present:
        status.reason = f"Image {IMAGE_TAG} is not built. Run `build_image()` first."
        return status

    try:
        proc = _run(
            [
                "docker", "run", "--rm",
                "-v", f"{ROOT}:/work",
                IMAGE_TAG, SCRIPT_IN_CONTAINER, "--probe",
            ],
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        status.available = False
        status.reason = f"The container did not respond within {timeout:.0f}s."
        return status

    if proc.returncode != 0:
        status.available = False
        status.reason = "Probe failed inside the container: " + (
            (proc.stderr or proc.stdout).strip().splitlines() or ["no detail"]
        )[-1]
        return status

    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        status.available = False
        status.reason = "The container returned output that was not JSON."
        return status

    status.probe = report
    status.openfhe_version = report.get("version")
    status.supports_native_bootstrap = bool(report.get("supports_native_bootstrap"))

    if not report.get("available"):
        # Distinguishing "the library did not load" from "the library lacks a
        # feature" matters. Reporting a failed *import* as a missing EvalBootstrap
        # sent the diagnosis in exactly the wrong direction once already: the real
        # cause was a missing libgomp1 in the image, and nothing about the OpenFHE
        # build at all.
        status.reason = (
            "OpenFHE did not load inside the container, so no capability could be "
            f"determined: {report.get('error', 'no detail reported')}. "
            "This is an image problem, not a statement about what OpenFHE supports."
        )
    elif not status.supports_native_bootstrap:
        status.reason = (
            "The container runs and OpenFHE imports, but this build does not expose "
            "EvalBootstrap. Reported as unsupported rather than substituting a "
            "client-aided refresh."
        )
    else:
        status.reason = ""
    return status


def run_comparison(config: dict[str, Any], *, timeout: float = 3600.0) -> dict[str, Any]:
    """Run the policy comparison with real bootstrapping, inside the container.

    Returns `{"ok": False, "reason": ...}` when it cannot, and never falls back.
    """
    status = probe()
    if not status.available or not status.supports_native_bootstrap:
        return {
            "ok": False,
            "reason": status.reason or "the OpenFHE backend is unavailable",
            "status": status.to_dict(),
        }

    try:
        proc = _run(
            [
                "docker", "run", "--rm",
                "-v", f"{ROOT}:/work",
                IMAGE_TAG, SCRIPT_IN_CONTAINER,
                "--config-json", json.dumps(config),
            ],
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "reason": f"The experiment exceeded {timeout / 60:.0f} minutes inside the container.",
            "status": status.to_dict(),
        }

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        combined = (proc.stdout or "") + (proc.stderr or "")
        # A container the kernel killed for memory reports itself as a truncated
        # stream, not as an error. Saying "output was not JSON" would send the
        # reader looking for a parsing bug instead of at the memory limit, so the
        # signature is named explicitly.
        if "unexpected EOF" in combined or "OOMKilled" in combined or proc.returncode == 137:
            return {
                "ok": False,
                "reason": (
                    "The container was killed part-way through, which at these parameters "
                    "is almost certainly the memory limit: bootstrapping key generation "
                    "at a large ring dimension needs several GB. Check `docker info` for "
                    "the VM's total memory, then either raise it in Docker Desktop's "
                    "settings or lower --ring-dim, --level-budget and --scaling-mod-size. "
                    f"Container said: {combined.strip()[:300]}"
                ),
                "status": status.to_dict(),
            }
        return {
            "ok": False,
            "reason": "The container returned output that was not JSON: " + combined[:400],
            "status": status.to_dict(),
        }
    payload["status"] = status.to_dict()
    return payload
