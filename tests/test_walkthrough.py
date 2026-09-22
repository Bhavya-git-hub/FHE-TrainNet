"""Tests for the encryption/decryption walkthrough entry script.

Deliberately a separate file. The parametrised import tests in
`test_honesty_and_dashboard.py` glob `pages/*.py` and so do not cover a new
*main* script, but extending them would make this branch modify a file `main`
also owns. Keeping every change on this branch in new files means it cannot
conflict with `main` and can be deleted without cleanup, which is worth more here
than avoiding a little duplication.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.data.loader import load_dataset, prepare
from src.experiments.config import ExperimentConfig, Mode
from src.experiments.runner import run_single

ROOT = Path(__file__).resolve().parents[1]
WALKTHROUGH = ROOT / "app" / "dashboard" / "Walkthrough.py"


def test_the_walkthrough_puts_the_repo_root_on_sys_path_first() -> None:
    """A main script must add the repository root before importing from it.

    Streamlit inserts only the main script's own directory into `sys.path`
    (`streamlit/runtime/scriptrunner/exec_code.py`), never the repository root.
    On a development machine the root arrives anyway via the working directory,
    so omitting this works locally and fails on every hosted deployment - which
    is exactly how `Home.py` shipped broken once.

    The order is asserted as well as the presence: a bootstrap placed below the
    first repository import is dead code, because that import has already raised.
    """
    lines = WALKTHROUGH.read_text(encoding="utf-8").splitlines()
    bootstrap = next(
        (i for i, ln in enumerate(lines) if "sys.path.insert(0, str(" in ln), None
    )
    first_repo_import = next(
        (
            i
            for i, ln in enumerate(lines)
            if ln.startswith(("from app.", "from src.", "import app", "import src"))
        ),
        None,
    )
    assert bootstrap is not None, "Walkthrough.py never puts the repo root on sys.path"
    assert first_repo_import is not None
    assert bootstrap < first_repo_import, (
        f"the bootstrap is at line {bootstrap + 1}, below the first repository import "
        f"at line {first_repo_import + 1} - the import raises first"
    )


def test_the_walkthrough_imports_with_only_its_own_directory_on_sys_path() -> None:
    """Prove the bootstrap works, not merely that its text is present."""
    probe = textwrap.dedent(
        """
        import os, sys, runpy
        sys.path.insert(0, os.path.join(os.getcwd(), "app", "dashboard"))
        sys.path = [p for p in sys.path if p not in ("", ".", os.getcwd())]
        try:
            runpy.run_path("app/dashboard/Walkthrough.py", run_name="__main__")
        except ModuleNotFoundError as exc:
            print("MISSING", exc)
            raise SystemExit(1)
        except Exception:
            pass          # no Streamlit runtime in bare mode; the imports resolved
        print("RESOLVED")
        """
    )
    # OpenBLAS reserves per-thread scratch buffers on import. Spawned at the end
    # of a suite that has had CKKS contexts open, it could not allocate them and
    # died before reaching the import under test - a failure of the host reported
    # as a failure of the code.
    environment = dict(os.environ)
    environment.update(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, errors="replace", cwd=ROOT, env=environment,
    )
    combined = result.stdout + result.stderr
    if "RESOLVED" not in result.stdout and "Memory allocation" in combined:
        pytest.skip(
            "the machine could not spare the memory to start a probe subprocess, so "
            "the Walkthrough.py import path was NOT verified in this run "
            f"({result.stderr.strip()[:120]})"
        )
    assert "RESOLVED" in result.stdout, result.stdout + result.stderr[-500:]


def test_both_encrypted_policies_return_a_decryptable_model(smoke_config) -> None:
    """Section 4 of the page decrypts `metrics["model"]` for each policy.

    If either run ever stops recording it, the page has nothing to decrypt and
    its entire point disappears - so the contract is asserted here rather than
    discovered in front of an audience.
    """
    dataset = load_dataset(smoke_config.dataset)
    split = prepare(
        dataset,
        test_fraction=smoke_config.test_fraction,
        seed=smoke_config.split_seed,
        n_samples=smoke_config.n_samples,
        feature_range=smoke_config.feature_range,
    )

    for mode in (Mode.FHE_BASELINE, Mode.FHE_ADAPTIVE):
        run = run_single(smoke_config, mode, split)
        assert run.status == "SUCCEEDED", f"{mode}: {run.reason}"

        model = run.metrics.get("model")
        assert model is not None, f"{mode} recorded no final model to decrypt"
        assert "weights" in model and "bias" in model, f"{mode} model is missing fields"
        assert len(model["weights"]) == split.x_train.shape[1]
        assert all(isinstance(w, float) for w in model["weights"])
        assert isinstance(model["bias"], float)

        # The page also shows these next to the weights; absent keys would render
        # as "not measured" rather than crash, but their absence would be a
        # regression worth catching here.
        assert run.metrics.get("encrypted_ops_total", 0) > 0
        assert run.decisions, f"{mode} recorded no controller decisions"
