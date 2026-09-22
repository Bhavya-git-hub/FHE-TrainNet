"""Tests for the claims the project must never quietly break.

Three failures would be invisible in ordinary use and fatal to the project's
credibility:

* a client-aided refresh described as bootstrapping,
* a quantity that was never measured rendered as `0`,
* secret key material reaching a results file or the screen.

None of those produces an exception or a wrong-looking chart. They produce a
confident, plausible, false artifact. So they are tested directly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.visualization.figures import (
    accuracy_comparison,
    capacity_over_operations,
    ciphertext_size_over_operations,
    decision_timeline,
    loss_curves,
    precision_over_operations,
    refresh_comparison,
    scalability,
    timing_comparison,
)
from src.crypto.backend import RefreshKind
from src.experiments.config import ExperimentConfig, Mode
from src.experiments.runner import run_experiment
from src.noise.monitor import METRIC_DISCLAIMER, METRIC_NAME

ROOT = Path(__file__).resolve().parents[1]


# -- never call it bootstrapping -----------------------------------------


def test_client_aided_refresh_is_never_labelled_bootstrapping() -> None:
    kind = RefreshKind.CLIENT_AIDED
    assert kind.is_bootstrapping is False
    assert "not CKKS bootstrapping" in kind.label
    assert RefreshKind.NATIVE_BOOTSTRAP.is_bootstrapping is True
    assert RefreshKind.UNSUPPORTED.is_bootstrapping is False


def test_no_source_file_claims_seal_supports_bootstrapping() -> None:
    """Guards against a future edit quietly promoting the claim."""
    offenders: list[str] = []
    pattern = re.compile(
        r"(seal|tenseal)[^.\n]{0,60}(supports?|performs?|provides?)[^.\n]{0,20}bootstrap",
        re.IGNORECASE,
    )
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "app").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            window = text[max(0, match.start() - 90) : match.end() + 40].lower()
            if "not" in window or "no " in window or "does not" in window:
                continue
            offenders.append(f"{path.name}: {match.group(0)}")
    assert not offenders, f"Found claims that SEAL supports bootstrapping: {offenders}"


def test_the_metric_is_not_called_noise_percentage() -> None:
    assert "noise" not in METRIC_NAME.lower().replace("noise/capacity", "")
    assert "Level" in METRIC_NAME
    assert "no CKKS noise budget" in METRIC_DISCLAIMER


def test_no_source_file_invents_a_noise_percentage() -> None:
    pattern = re.compile(r"noise[_ ]?(percent|pct|%|budget)", re.IGNORECASE)
    offenders: list[str] = []
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "app").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            window = text[max(0, match.start() - 160) : match.end() + 160].lower()
            # Mentions are allowed only where we say the quantity does not exist,
            # or in the probe that demonstrates the accessor is absent.
            if any(
                phrase in window
                for phrase in ("no ckks noise", "exposes no", "unsupported", "never", "not ",
                               "does not", "would be", "forbid", "accessor")
            ):
                continue
            offenders.append(f"{path.name}:{match.group(0)}")
    assert not offenders, f"Possible invented noise quantity: {offenders}"


# -- missing stays missing ------------------------------------------------


def test_fmt_renders_missing_as_words_not_zero() -> None:
    from app.dashboard.common import fmt

    assert fmt(None) == "not measured"
    assert fmt(None, ".1f", "-") == "-"
    assert fmt(0.0) == "0.000"
    assert fmt(0.5) == "0.500"


def test_results_file_keeps_unmeasured_quantities_null(tmp_path: Path) -> None:
    config = ExperimentConfig.load(ROOT / "configs" / "fast_smoke.yaml")
    config.modes = (Mode.PLAINTEXT,)
    payload = run_experiment(config, results_dir=tmp_path)
    raw = (tmp_path / payload["run_id"] / "results.json").read_text(encoding="utf-8")
    stored = json.loads(raw)
    metrics = stored["runs"][0]["metrics"]
    assert metrics["refreshes"] is None
    assert metrics["refresh_kind"] is None
    assert metrics["min_levels_remaining"] is None


def test_resource_summary_is_unavailable_not_zero() -> None:
    from src.evaluation.resources import ResourceSampler

    sampler = ResourceSampler()
    summary = sampler.summary()  # never started
    assert summary.available is False
    assert summary.peak_rss_mb is None
    assert summary.reason


# -- no secrets escape ----------------------------------------------------


def test_results_directory_contains_no_key_material(tmp_path: Path) -> None:
    config = ExperimentConfig.load(ROOT / "configs" / "fast_smoke.yaml")
    payload = run_experiment(config, results_dir=tmp_path)
    out = tmp_path / payload["run_id"]
    suspicious = re.compile(r"secret_key|private_key|BEGIN [A-Z ]*KEY", re.IGNORECASE)
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert not suspicious.search(text), f"{path.name} may contain key material"


def test_gitignore_excludes_key_material() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("keys/", "*.key", "*.secret"):
        assert pattern in text


def test_creating_a_key_zone_writes_nothing_to_disk(tmp_path: Path, monkeypatch) -> None:
    """Behavioural, not textual: a secret key that is never written cannot be committed."""
    from src.crypto.backend import CKKSParams
    from src.crypto.tenseal_backend import DataOwnerZone

    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    zone = DataOwnerZone(
        CKKSParams(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), scale_bits=40),
        generate_galois=False,
    )
    zone.encrypt([1.0, 2.0])
    zone.secret_key_fingerprint()
    assert set(tmp_path.rglob("*")) == before, "key generation must not touch the filesystem"


def test_secret_key_serialisation_happens_only_for_the_fingerprint_hash() -> None:
    """The one place the secret key is serialised feeds a hash, and nothing else."""
    source = (ROOT / "src" / "crypto" / "tenseal_backend.py").read_text(encoding="utf-8")
    assert source.count("save_secret_key=True") == 1
    line = next(l for l in source.splitlines() if "save_secret_key=True" in l)
    assert "sha256" in line and "hexdigest" in line


# -- figures --------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_payload(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("figs")
    config = ExperimentConfig.load(ROOT / "configs" / "fast_smoke.yaml")
    config.modes = (Mode.PLAINTEXT, Mode.FHE_BASELINE, Mode.FHE_ADAPTIVE)
    return run_experiment(config, results_dir=tmp)


def test_every_figure_builds_from_real_run_data(sample_payload) -> None:
    by_mode = sample_payload["comparison"]["by_mode"]
    encrypted = [r for r in sample_payload["runs"] if r["mode"] != "plaintext"][0]

    for figure in (
        accuracy_comparison(by_mode),
        timing_comparison(by_mode),
        refresh_comparison(by_mode),
        capacity_over_operations([], []),
        decision_timeline([]),
    ):
        assert figure is not None
        assert figure.to_dict() is not None


def test_figures_say_why_they_are_empty_rather_than_showing_a_blank() -> None:
    figure = precision_over_operations([])
    text = json.dumps(figure.to_dict())
    assert "No canary precision" in text


def test_capacity_figure_labels_its_provenance() -> None:
    series = [
        {"index": 0, "levels_remaining": 4, "state": "safe", "serialized_bytes": 100,
         "precision_bits": 13.0},
        {"index": 1, "levels_remaining": 0, "state": "exhausted", "serialized_bytes": 50,
         "precision_bits": 11.0},
    ]
    text = json.dumps(capacity_over_operations(series, []).to_dict())
    assert "derived" in text.lower()
    assert "no CKKS noise budget" in text


def test_precision_figure_keeps_gaps_for_unmeasured_points() -> None:
    series = [
        {"index": 0, "precision_bits": 13.0},
        {"index": 1, "precision_bits": None},
        {"index": 2, "precision_bits": 11.0},
    ]
    figure = precision_over_operations(series)
    y = figure.data[0].y
    assert y[1] is None, "an unmeasured point must be a gap, not an interpolated value"
    assert figure.data[0].connectgaps is False


def test_size_figure_handles_a_run_with_no_size_measurements() -> None:
    text = json.dumps(ciphertext_size_over_operations([{"index": 0, "serialized_bytes": None}]).to_dict())
    assert "not measured" in text.lower() or "was not measured" in text.lower()


def test_scalability_figure_skips_missing_points() -> None:
    points = [{"size": 10, "train_seconds": 1.0}, {"size": 20, "train_seconds": None}]
    figure = scalability(points, "size", "train_seconds", "Samples", "Seconds", "t")
    assert len(figure.data[0].x) == 1


def test_loss_curves_handles_modes_without_loss() -> None:
    assert loss_curves({"plaintext": [{"epoch": 0, "train_loss": None}]}) is not None


# -- dashboard pages ------------------------------------------------------


def test_every_dashboard_page_compiles() -> None:
    """Catches syntax and import-time errors without launching Streamlit."""
    import py_compile

    pages = list((ROOT / "app" / "dashboard").rglob("*.py"))
    assert len(pages) >= 8, "expected the full set of dashboard pages"
    for path in pages:
        py_compile.compile(str(path), doraise=True)


def test_scripts_compile() -> None:
    import py_compile

    for path in (ROOT / "scripts").glob("*.py"):
        py_compile.compile(str(path), doraise=True)


# -- the OpenFHE backend must never over-claim ----------------------------


def test_unavailable_openfhe_backend_never_claims_bootstrapping() -> None:
    """The most damaging possible bug: a dashboard saying "ACTUAL CKKS
    BOOTSTRAPPING" over results produced by a client-aided refresh."""
    from src.crypto.openfhe_backend import BackendStatus

    for status in (
        BackendStatus(available=False, reason="docker down"),
        BackendStatus(available=True, image_present=False),
        BackendStatus(available=True, image_present=True, supports_native_bootstrap=False),
    ):
        assert status.refresh_kind is RefreshKind.UNSUPPORTED
        assert status.refresh_kind.is_bootstrapping is False


def test_openfhe_claims_bootstrapping_only_when_probed_successfully() -> None:
    from src.crypto.openfhe_backend import BackendStatus

    status = BackendStatus(
        available=True, image_present=True, supports_native_bootstrap=True,
        openfhe_version="1.5.1",
    )
    assert status.refresh_kind is RefreshKind.NATIVE_BOOTSTRAP
    assert status.refresh_kind.is_bootstrapping is True


def test_openfhe_backend_has_no_fallback_to_tenseal() -> None:
    """A silent downgrade would be indistinguishable from success in the UI.

    The module cannot fall back to the client-aided refresh if it cannot reach the
    code that performs one, so the check is simply that it never imports it.
    """
    source = (ROOT / "src" / "crypto" / "openfhe_backend.py").read_text(encoding="utf-8")
    assert "from src.crypto.tenseal_backend import" not in source
    assert "import src.crypto.tenseal_backend" not in source
    assert "DataOwnerZone" not in source


def test_openfhe_status_reports_a_reason_when_unavailable() -> None:
    from src.crypto.openfhe_backend import docker_status

    status = docker_status(timeout=20.0)
    if not status.available:
        assert status.reason, "an unavailable backend must say why"
        assert status.refresh_kind.is_bootstrapping is False


def test_container_script_refuses_without_evalbootstrap() -> None:
    """The in-container script must not run the comparison if it cannot bootstrap."""
    source = (ROOT / "docker" / "openfhe" / "openfhe_experiment.py").read_text(encoding="utf-8")
    assert 'if not probe.get("supports_native_bootstrap")' in source
    assert "does not expose EvalBootstrap" in source


# -- the environment trap -------------------------------------------------


def test_preflight_detects_a_missing_dependency() -> None:
    """Running a script without the venv must explain itself, not raise ImportError.

    This happened: `python scripts/validate.py` on a system interpreter reported
    several project checks as FAILED with `ModuleNotFoundError: No module named
    'tenseal'`, which reads as "the project is broken" when the shell was wrong.
    """
    import subprocess
    import sys as _sys

    from src.preflight import REQUIRED, venv_python

    assert "tenseal" in REQUIRED

    # Drive the check with an interpreter that cannot see the dependencies by
    # blanking the path it would find them on.
    proc = subprocess.run(
        [_sys.executable, "-c",
         "import sys; sys.path=[p for p in sys.path if 'site-packages' not in p];"
         f"sys.path.insert(0, r'{ROOT}');"
         "from src.preflight import require_dependencies;"
         "require_dependencies(script='probe')"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    message = proc.stderr
    assert "cannot run" in message
    assert "tenseal" in message
    assert "environment problem" in message
    assert str(venv_python()) in message


def test_every_script_checks_its_interpreter_first() -> None:
    """A script that skips the check reintroduces the confusing failure."""
    offenders = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "require_dependencies" not in text:
            offenders.append(path.name)
    assert not offenders, f"scripts missing the interpreter preflight: {offenders}"


def test_commands_doc_holds_no_pasted_run_output() -> None:
    """Documentation must not accumulate captured console output.

    `docs/COMMANDS.md` once had two validation runs redirected into it, leaving
    a reference document asserting that the project failed.
    """
    text = (ROOT / "docs" / "COMMANDS.md").read_text(encoding="utf-8")
    assert "FHE-TrainNet validation" not in text
    assert "passed, 0 failed" not in text
    assert text.count("ModuleNotFoundError") <= 1  # one deliberate mention


def test_stat_tolerates_a_metric_the_run_never_recorded() -> None:
    """A results file from before a metric existed must not take a page down.

    This is not hypothetical: `make_report.py` crashed with `KeyError:
    'peak_rss_above_baseline_mb'` on a benchmark recorded twenty minutes before
    that metric was added.
    """
    from app.dashboard.common import stat

    stats = {"train_seconds": {"mean": 12.5, "min": 12.0, "max": 13.0, "n": 3}}
    assert stat(stats, "train_seconds", ".1f") == "12.5"
    assert stat(stats, "peak_rss_above_baseline_mb") == "not recorded"
    assert stat({"refreshes": {"mean": None}}, "refreshes") == "not measured"


def test_dashboard_reads_comparison_stats_defensively() -> None:
    """Direct stats["key"]["mean"] indexing is the shape of that crash."""
    import re

    offenders = []
    pattern = re.compile(r'\b(s|stats)\["[a-z_]+"\]\["mean"\]')
    for path in sorted((ROOT / "app" / "dashboard").rglob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if pattern.search(line):
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, f"fragile stat access that breaks on older runs: {offenders}"


# -- the OpenFHE container script ----------------------------------------


def test_container_script_preserves_its_probe_value() -> None:
    """Squaring the probe must not destroy it.

    The level-consuming operation is `EvalMult(ct, ct)`. That only leaves the
    value intact for a probe of 1.0; an earlier version squared 0.25, which
    underflows toward zero within a few steps and makes the decrypted
    verification at the end meaningless.
    """
    source = (ROOT / "docker" / "openfhe" / "openfhe_experiment.py").read_text(encoding="utf-8")
    assert "values = [1.0] * min(8, num_slots)" in source
    # Only code matters here - the old value is still named in the comment that
    # explains why it was wrong, and that comment is worth keeping.
    code = [l for l in source.splitlines() if not l.lstrip().startswith("#")]
    assert not [l for l in code if "0.25" in l]


def test_container_script_starts_from_a_fresh_ciphertext() -> None:
    """Encoding at the bottom of the chain would leave nothing to spend.

    OpenFHE's own bootstrapping examples encode at `depth - 1` to show a bootstrap
    lifting an exhausted ciphertext. This experiment needs the opposite: a fresh
    ciphertext the policy can spend down.
    """
    source = (ROOT / "docker" / "openfhe" / "openfhe_experiment.py").read_text(encoding="utf-8")
    assert "MakeCKKSPackedPlaintext(values, 1, 0, None, num_slots)" in source
    assert 'total_depth"] - 1' not in source


def test_container_script_does_not_double_rescale() -> None:
    """FLEXIBLEAUTO rescales inside EvalMult; a manual Rescale spends a second level."""
    source = (ROOT / "docker" / "openfhe" / "openfhe_experiment.py").read_text(encoding="utf-8")
    assert "cc.Rescale(" not in source


def test_container_script_reserves_depth_for_bootstrapping() -> None:
    """Bootstrapping needs its own levels on top of the computation's."""
    source = (ROOT / "docker" / "openfhe" / "openfhe_experiment.py").read_text(encoding="utf-8")
    assert "GetBootstrapDepth" in source
    assert "usable_levels + bootstrap_depth" in source


def test_dockerfile_installs_the_openmp_runtime() -> None:
    """OpenFHE links against libgomp; without it `import openfhe` fails outright."""
    dockerfile = (ROOT / "docker" / "openfhe" / "Dockerfile").read_text(encoding="utf-8")
    assert "libgomp1" in dockerfile


def test_probe_separates_a_failed_import_from_a_missing_feature() -> None:
    """Misattributing one as the other sends the diagnosis the wrong way.

    It did: a missing `libgomp1` in the image made `import openfhe` fail, and the
    host reported "this OpenFHE build does not expose EvalBootstrap" - a statement
    about the library, when the fault was in the image.
    """
    import src.crypto.openfhe_backend as ob

    captured = {}

    class _Proc:
        returncode = 0
        stderr = ""

        def __init__(self, payload):
            self.stdout = payload

    # Import failed inside the container.
    def fake_import_failure(args, timeout=60.0):
        return _Proc(json.dumps({
            "library": "openfhe", "probes": {}, "available": False,
            "error": "ImportError: libgomp.so.1: cannot open shared object file",
        }))

    base = ob.BackendStatus(available=True, image_present=True, docker_version="29.7.2")
    original = ob._run
    try:
        ob._run = fake_import_failure
        status = ob.probe(base)
    finally:
        ob._run = original

    assert status.supports_native_bootstrap is False
    assert status.refresh_kind.is_bootstrapping is False
    assert "did not load" in status.reason
    assert "libgomp" in status.reason
    assert "image problem" in status.reason
    # It must NOT claim anything about what OpenFHE supports.
    assert "does not expose EvalBootstrap" not in status.reason


def test_docker_output_is_decoded_as_utf8() -> None:
    """Docker build output is UTF-8; cp1252 decoding killed a successful build."""
    source = (ROOT / "src" / "crypto" / "openfhe_backend.py").read_text(encoding="utf-8")
    assert 'encoding="utf-8"' in source
    assert 'errors="replace"' in source


# --- the dashboard must be importable the way Streamlit actually imports it ----

ENTRY_SCRIPTS = [ROOT / "app" / "dashboard" / "Home.py"] + sorted(
    (ROOT / "app" / "dashboard" / "pages").glob("*.py")
)


@pytest.mark.parametrize("script", ENTRY_SCRIPTS, ids=lambda p: p.name)
def test_entry_scripts_put_the_repo_root_on_sys_path_first(script: Path) -> None:
    """Every Streamlit entry script must add the repository root before using it.

    Streamlit inserts only the main script's own directory into `sys.path`
    (`streamlit/runtime/scriptrunner/exec_code.py`), never the repository root.
    Every page under `pages/` did this from the start. `Home.py` - the file
    Streamlit is actually pointed at - did not, and on a development machine the
    root arrives anyway via the working directory, so the omission was invisible
    through the entire build. On Streamlit Cloud it does not arrive: the app
    died with ModuleNotFoundError before rendering a line.

    Parametrising over all eleven rather than testing `Home.py` alone is the
    point. Ten files were right and one was wrong, and nothing distinguished
    them; the next page added will be checked the same way.

    Asserting the order matters as much as the presence. A bootstrap that sits
    below the first `app.`/`src.` import is dead code - the import above it has
    already raised.
    """
    lines = script.read_text(encoding="utf-8").splitlines()
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
    assert bootstrap is not None, (
        f"{script.name} imports the repository but never puts its root on sys.path"
    )
    assert first_repo_import is not None
    assert bootstrap < first_repo_import, (
        f"{script.name} adds the repo root at line {bootstrap + 1}, below its first "
        f"repository import at line {first_repo_import + 1} - the import raises first"
    )


def test_home_imports_with_only_the_script_dir_on_sys_path() -> None:
    """Prove the bootstrap works, rather than only that its text is present.

    The static test above checks ordering; this one reproduces Streamlit Cloud's
    actual import environment and runs the real file.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent(
        """
        import os, sys, runpy
        sys.path.insert(0, os.path.join(os.getcwd(), "app", "dashboard"))
        sys.path = [p for p in sys.path if p not in ("", ".", os.getcwd())]
        try:
            runpy.run_path("app/dashboard/Home.py", run_name="__main__")
        except ModuleNotFoundError as exc:
            print("MISSING", exc)
            raise SystemExit(1)
        except Exception:
            pass          # no Streamlit runtime in bare mode; imports resolved
        print("RESOLVED")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, errors="replace", cwd=ROOT,
    )
    assert "RESOLVED" in result.stdout, result.stdout + result.stderr[-500:]


# --- profile selection must never choose a profile the host cannot run --------


@pytest.mark.parametrize(
    ("cgroup_mb", "hosted", "psutil_mb", "expected"),
    [
        # A hosted tier that publishes no readable limit. This is the case that
        # killed the deployed app: the old code fell through to psutil, was told
        # the *host's* memory, and picked the batched profile.
        (None, True, 16000.0, "cloud"),
        # A container that does publish a limit, but one below the batched
        # profile's measured 2196 MB peak. The old threshold of 1500 MB called
        # this enough.
        (2700.0, True, 16000.0, "cloud"),
        # Genuinely roomy container: the batched profile fits.
        (8000.0, True, 16000.0, "demo"),
        # An ordinary laptop. No cgroup, not hosted, psutil is telling the truth.
        (None, False, 16000.0, "demo"),
        # A small machine, honestly reported.
        (None, False, 900.0, "cloud"),
    ],
)
def test_the_profile_is_never_one_the_host_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
    cgroup_mb: float | None,
    hosted: bool,
    psutil_mb: float,
    expected: str,
) -> None:
    """Guessing high kills the process; guessing low only costs throughput.

    The batched profile was measured at 2196 MB peak RSS. Anything that selects
    it with less than that available is not a degraded demonstration, it is a
    container kill part-way through one - which reads as a broken project rather
    than an exhausted host, and is exactly what the deployed app did.
    """
    from src import runtime

    monkeypatch.delenv("FHE_TRAINNET_CONFIG", raising=False)
    monkeypatch.setattr(runtime, "cgroup_memory_limit_mb", lambda: cgroup_mb)
    monkeypatch.setattr(runtime, "looks_hosted", lambda: hosted)
    monkeypatch.setattr(runtime, "available_memory_mb", lambda: cgroup_mb or psutil_mb)

    assert runtime.default_config_name() == expected


def test_the_threshold_clears_the_measured_peak() -> None:
    """The bar has to be above what the profile actually needs, not below it.

    It was set from the rotation-key figure (1901 MB) and rounded down to 1500,
    while the profile's real peak is 2196 MB. A limit between those two numbers
    selected a profile that could not run.
    """
    from src import runtime

    assert runtime.LOW_MEMORY_THRESHOLD_MB > runtime.BATCHED_PROFILE_PEAK_MB


def test_an_explicit_override_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment that knows better than the detection must be able to say so."""
    from src import runtime

    monkeypatch.setenv("FHE_TRAINNET_CONFIG", "benchmark")
    monkeypatch.setattr(runtime, "looks_hosted", lambda: True)
    assert runtime.default_config_name() == "benchmark"
