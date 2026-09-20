"""Check the interpreter before a script does any work.

Every entry point in `scripts/` needs TenSEAL, and the commonest way to not have
it is the most ordinary mistake there is: running `python scripts/...` with the
system interpreter because the virtual environment was not activated. Python's
own report of that is `ModuleNotFoundError: No module named 'tenseal'`, which
says nothing about the actual problem or its fix.

Worse, it is *misleading*. `scripts/validate.py` run that way reports several
project checks as FAILED, which reads as "the project is broken" when the project
is fine and the shell is not. That happened, and this module exists because of it.

The check is cheap, runs before anything else, and names the exact command to fix
the situation on the platform it is running on.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Import name -> the name you would install it under, where they differ.
REQUIRED: dict[str, str] = {
    "tenseal": "tenseal",
    "numpy": "numpy",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
}


def venv_python() -> Path:
    """Where this project's interpreter lives on this platform."""
    if os.name == "nt":
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def running_in_project_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == venv_python().resolve()
    except OSError:
        return False


def missing_dependencies() -> list[str]:
    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def require_dependencies(*, script: str = "this script", exit_code: int = 2) -> None:
    """Exit with an explanation if the interpreter cannot run this project.

    Deliberately exits rather than raising: the reader is at a shell, and a
    traceback would bury the one line that matters.
    """
    missing = missing_dependencies()
    if not missing:
        return

    interpreter = venv_python()
    lines = [
        "",
        f"{script} cannot run: this Python is missing {', '.join(sorted(missing))}.",
        "",
        f"  interpreter in use : {sys.executable}",
        f"  project interpreter: {interpreter}",
        "",
    ]

    if interpreter.exists() and not running_in_project_venv():
        # The overwhelmingly common case: the venv exists and was not activated.
        activate = (
            r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
        )
        lines += [
            "The project's virtual environment exists but is not active. Activate it:",
            "",
            f"    cd {ROOT}",
            f"    {activate}",
            "",
            "or run the script with that interpreter directly:",
            "",
            f"    {interpreter} {' '.join(sys.argv)}",
        ]
    elif not interpreter.exists():
        lines += [
            "The project's virtual environment has not been created yet:",
            "",
            f"    cd {ROOT}",
            "    python -m venv .venv",
            r"    .venv\Scripts\activate" if os.name == "nt" else "    source .venv/bin/activate",
            "    pip install -r requirements.txt",
        ]
    else:
        lines += [
            "The virtual environment is active but incomplete. Reinstall:",
            "",
            "    pip install -r requirements.txt",
        ]

    lines += [
        "",
        "Nothing is wrong with the project - this is an environment problem.",
        "",
    ]
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(exit_code)
