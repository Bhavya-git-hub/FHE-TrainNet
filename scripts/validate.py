"""Check the project against its own claims, and print what failed.

    python scripts/validate.py            # everything except the long experiments
    python scripts/validate.py --full     # also runs demo, control and report

This is the Master Prompt section 25 checklist as a program. It is deliberately
suspicious of the documentation: several checks read the README and the generated
capability file and fail if they disagree with each other, because the failure
this project most needs to catch is a claim that has quietly stopped being true.

Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preflight import require_dependencies  # noqa: E402

require_dependencies(script="scripts/validate.py")

PY = sys.executable

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def check(name: str, fn: Callable[[], tuple[str, str]]) -> None:
    print(f"  {name:<52}", end="", flush=True)
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a broken check is a failed check
        status, detail = FAIL, f"{type(exc).__name__}: {exc}"
    print(status + (f"  {detail}" if detail else ""))
    results.append((name, status, detail))


def run(args: list[str], timeout: float = 1800.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False
    )


# -- environment ----------------------------------------------------------


def check_imports() -> tuple[str, str]:
    import tenseal

    return PASS, f"TenSEAL {tenseal.__version__}"


def check_datasets() -> tuple[str, str]:
    from src.data.loader import available_datasets, load_dataset

    names = available_datasets()
    if not names:
        return FAIL, "no datasets in datasets/ - run scripts/build_datasets.py"
    for name in names:
        load_dataset(name)
    return PASS, f"{len(names)} dataset(s) load and validate"


def check_configs() -> tuple[str, str]:
    from src.experiments.config import ExperimentConfig

    paths = sorted((ROOT / "configs").glob("*.yaml"))
    for path in paths:
        ExperimentConfig.load(path)
    return PASS, f"{len(paths)} config(s) valid"


# -- cryptography ---------------------------------------------------------


def check_capability_file() -> tuple[str, str]:
    path = ROOT / "docs" / "capabilities.json"
    if not path.exists():
        return FAIL, "missing - run scripts/probe_library.py"
    caps = json.loads(path.read_text(encoding="utf-8"))
    errored = [p["name"] for p in caps["probes"] if p["result"] == "error"]
    if errored:
        return FAIL, f"probe(s) errored: {errored}"
    return PASS, f"{len(caps['probes'])} probes, none errored"


def check_no_bootstrapping_claim() -> tuple[str, str]:
    """The capability file and the code must agree that SEAL cannot bootstrap."""
    from src.crypto.backend import RefreshKind

    caps = json.loads((ROOT / "docs" / "capabilities.json").read_text(encoding="utf-8"))
    probe = next(p for p in caps["probes"] if p["name"] == "native_bootstrap")
    if probe["result"] == "supported":
        return FAIL, "the probe found bootstrapping but the code reports client-aided"
    if caps["refresh_mechanism"] != RefreshKind.CLIENT_AIDED.value:
        return FAIL, f"refresh_mechanism is {caps['refresh_mechanism']}"
    if RefreshKind.CLIENT_AIDED.is_bootstrapping:
        return FAIL, "CLIENT_AIDED reports itself as bootstrapping"
    return PASS, "probe, enum and label agree: no CKKS bootstrapping"


def check_trust_boundary() -> tuple[str, str]:
    from src.crypto.backend import CKKSParams
    from src.crypto.tenseal_backend import DataOwnerZone

    owner = DataOwnerZone(
        CKKSParams(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), scale_bits=40),
        generate_galois=False,
    )
    enc = owner.encrypt([1.0, 2.0])
    try:
        enc.raw.decrypt()
    except Exception:  # noqa: BLE001 - the refusal is the pass condition
        return PASS, "compute zone cannot decrypt"
    return FAIL, "THE COMPUTE ZONE DECRYPTED ITS INPUT - the privacy claim is false"


def check_bad_parameters_refused() -> tuple[str, str]:
    from src.crypto.backend import CKKSParams, ParameterError

    bad = [
        dict(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 40, 60)),
        dict(poly_modulus_degree=8192, coeff_mod_bit_sizes=(50, 30, 30, 30, 50), scale_bits=40),
        dict(poly_modulus_degree=8192, coeff_mod_bit_sizes=(40, 21, 21, 21, 40), scale_bits=21),
    ]
    for params in bad:
        try:
            CKKSParams(**params)
        except ParameterError:
            continue
        return FAIL, f"accepted an unsound configuration: {params}"
    return PASS, "unsound CKKS parameters refused with reasons"


# -- honesty --------------------------------------------------------------


def check_no_secrets_on_disk() -> tuple[str, str]:
    import re

    suspicious = re.compile(r"secret_key|private_key|BEGIN [A-Z ]*KEY", re.IGNORECASE)
    hits = []
    for folder in ("results", "reports", "docs"):
        base = ROOT / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".csv", ".yaml", ".md", ".txt"):
                if suspicious.search(path.read_text(encoding="utf-8", errors="ignore")):
                    hits.append(str(path.relative_to(ROOT)))
    if hits:
        return FAIL, f"possible key material in {hits[:3]}"
    return PASS, "no key material in results, reports or docs"


def check_missing_stays_missing() -> tuple[str, str]:
    """A plaintext run must record refresh metrics as null, never zero."""
    from src.experiments.registry import list_runs

    for summary in list_runs():
        if not summary.payload:
            continue
        for run_data in summary.payload["runs"]:
            if run_data["mode"] != "plaintext":
                continue
            metrics = run_data["metrics"]
            for key in ("refreshes", "refresh_kind"):
                if metrics.get(key) is not None:
                    return FAIL, f"{summary.run_id}: plaintext {key} is {metrics[key]!r}, not null"
        return PASS, f"checked {summary.run_id}"
    return SKIP, "no recorded run with a plaintext mode"


def check_readme_matches_capabilities() -> tuple[str, str]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    if "does not implement ckks bootstrapping" not in readme:
        return FAIL, "README does not state that SEAL cannot bootstrap"
    if "no ckks noise budget" not in readme:
        return FAIL, "README does not state that no noise budget is exposed"
    return PASS, "README states both limitations"


# -- the suite ------------------------------------------------------------


def check_tests() -> tuple[str, str]:
    started = time.perf_counter()
    proc = run([PY, "-m", "pytest", "-q", "--no-header"], timeout=2400)
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:]
    summary = tail[0] if tail else "no output"
    if proc.returncode != 0:
        return FAIL, summary
    if "skipped" in summary:
        return FAIL, f"tests skipped - every test should run: {summary}"
    return PASS, f"{summary} in {time.perf_counter() - started:.0f}s"


def check_demo_experiment() -> tuple[str, str]:
    proc = run(
        [PY, "scripts/run_experiment.py", "--config", "configs/demo.yaml",
         "--name", "validate", "--quiet"],
        timeout=2400,
    )
    if proc.returncode != 0:
        return FAIL, (proc.stdout + proc.stderr).strip().splitlines()[-1]
    verdict = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().startswith("*")]
    return PASS, verdict[0][2:80] if verdict else "completed"


def check_no_refresh_control_fails() -> tuple[str, str]:
    """The control is supposed to FAIL. A pass here means the limit is not real."""
    proc = run(
        [PY, "scripts/run_experiment.py", "--config", "configs/no_refresh_control.yaml",
         "--set", "epochs=2", "--quiet"],
        timeout=2400,
    )
    if "exhausted the ciphertext's computation capacity" not in proc.stdout:
        return FAIL, "the no-refresh control did NOT exhaust capacity as expected"
    return PASS, "capacity exhaustion reproduced and reported"


def check_report() -> tuple[str, str]:
    proc = run([PY, "scripts/make_report.py"], timeout=900)
    if proc.returncode != 0:
        return FAIL, (proc.stdout + proc.stderr).strip().splitlines()[-1]
    line = next((ln.strip() for ln in proc.stdout.splitlines() if "figures drawn" in ln), "")
    return PASS, line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also run the long experiments")
    args = parser.parse_args()

    print("FHE-TrainNet validation\n")
    print("Environment")
    check("dependencies import", check_imports)
    check("bundled datasets load", check_datasets)
    check("shipped configs valid", check_configs)

    print("\nCryptography")
    check("capability probe recorded, none errored", check_capability_file)
    check("trust boundary enforced", check_trust_boundary)
    check("unsound CKKS parameters refused", check_bad_parameters_refused)

    print("\nHonesty")
    check("no bootstrapping is claimed anywhere", check_no_bootstrapping_claim)
    check("no key material written to disk", check_no_secrets_on_disk)
    check("unmeasured quantities stay null", check_missing_stays_missing)
    check("README agrees with the probe", check_readme_matches_capabilities)

    print("\nTest suite")
    check("pytest, no failures and no skips", check_tests)

    if args.full:
        print("\nExperiments (slow)")
        check("demo experiment runs end to end", check_demo_experiment)
        check("no-refresh control fails as designed", check_no_refresh_control_fails)
        check("report and figures generate", check_report)
    else:
        print("\nExperiments: skipped (pass --full to run them)")

    failed = [r for r in results if r[1] == FAIL]
    skipped = [r for r in results if r[1] == SKIP]
    print(f"\n{len(results) - len(failed) - len(skipped)} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped")
    for name, _, detail in failed:
        print(f"  FAILED  {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
