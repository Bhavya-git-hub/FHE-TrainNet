"""Build, probe and run the OpenFHE container - the backend that really bootstraps.

    python scripts/openfhe_backend.py --probe
    python scripts/openfhe_backend.py --build
    python scripts/openfhe_backend.py --run

If anything is unavailable, this prints why and exits non-zero. It never falls
back to the TenSEAL client-aided refresh, because reporting that as bootstrapping
would be the one lie this project must not tell.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preflight import require_dependencies  # noqa: E402

require_dependencies(script="scripts/openfhe_backend.py")

from src.crypto.openfhe_backend import (  # noqa: E402
    IMAGE_TAG,
    build_image,
    docker_status,
    probe,
    run_comparison,
)

RESULTS = ROOT / "results" / "openfhe"


def show(status) -> None:
    print(f"  docker available      : {status.available}")
    print(f"  docker version        : {status.docker_version or '-'}")
    print(f"  image {IMAGE_TAG:<22}: {'present' if status.image_present else 'not built'}")
    print(f"  openfhe version       : {status.openfhe_version or '-'}")
    print(f"  EvalBootstrap         : {status.supports_native_bootstrap}")
    print(f"  refresh kind          : {status.refresh_kind.label}")
    if status.reason:
        print(f"  reason                : {status.reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="build the image (slow first time)")
    parser.add_argument("--probe", action="store_true", help="report what the container supports")
    parser.add_argument("--run", action="store_true", help="run the policy comparison")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--depth-per-step", type=int, default=5)
    parser.add_argument("--usable-levels", type=int, default=10)
    parser.add_argument("--ring-dim", type=int, default=1 << 16)
    parser.add_argument("--baseline-interval", type=int, default=1)
    parser.add_argument(
        "--level-budget", type=int, nargs=2, default=[2, 2],
        help="bootstrapping encode/decode level budget; smaller reserves less depth "
             "and far less memory, at the cost of a slower EvalBootstrap",
    )
    parser.add_argument(
        "--scaling-mod-size", type=int, default=45,
        help="bits per level; fewer bits keeps the total modulus inside what a "
             "smaller ring allows at the 128-bit security level",
    )
    args = parser.parse_args()

    if not (args.build or args.probe or args.run):
        args.probe = True

    if args.build:
        print(f"Building {IMAGE_TAG} ...")
        status = build_image(progress=print)
        show(status)
        if not status.available:
            return 1

    if args.probe or args.run:
        print("\nOpenFHE backend status:")
        status = probe()
        show(status)
        if not (status.available and status.supports_native_bootstrap):
            print(
                "\nThe OpenFHE backend is unavailable. This is reported, not worked around: "
                "the TenSEAL backend's client-aided refresh is NOT substituted and NOT "
                "described as bootstrapping.\n"
                "Phase A (the TenSEAL path) is complete and self-sufficient without this."
            )
            return 1

    if args.run:
        config = {
            "steps": args.steps,
            "depth_per_step": args.depth_per_step,
            "usable_levels": args.usable_levels,
            "ring_dim": args.ring_dim,
            "baseline_interval": args.baseline_interval,
            "level_budget": list(args.level_budget),
            "scaling_mod_size": args.scaling_mod_size,
            "num_slots": 8,
        }
        print(f"\nRunning the comparison in the container with REAL EvalBootstrap ...")
        print(f"  {config}")
        started = time.perf_counter()
        payload = run_comparison(config)
        elapsed = time.perf_counter() - started

        if not payload.get("ok"):
            print(f"\nFAILED after {elapsed:.0f}s: {payload.get('reason')}")
            return 1

        ctx = payload["context"]
        print(f"\nContext: ring {ctx['ring_dim']}, {ctx['usable_levels']} usable levels "
              f"(+{ctx['bootstrap_depth']} reserved for bootstrapping), "
              f"setup {ctx['setup_seconds']:.0f}s")
        print(f"\n{'policy':<16} {'status':<10} {'bootstraps':>11} {'total s':>9} {'mean boot s':>12}")
        for name, r in payload["results"].items():
            mean = r["bootstrap_seconds_mean"]
            print(f"  {name:<14} {r['status']:<10} {r['bootstraps']:>11} "
                  f"{r['seconds']:>9.1f} {('-' if mean is None else f'{mean:.2f}'):>12}")
            if r["reason"]:
                print(f"      reason: {r['reason']}")

        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"openfhe-{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nWritten to {path}")
        print("\nEvery refresh above is an ACTUAL CKKS bootstrapping call (OpenFHE "
              "EvalBootstrap), not a client-aided refresh.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
