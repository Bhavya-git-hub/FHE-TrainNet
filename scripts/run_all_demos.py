"""The nine demonstrations, runnable without the dashboard.

    python scripts/run_all_demos.py            # all of them
    python scripts/run_all_demos.py --demo 5   # just one
    python scripts/run_all_demos.py --list

Each demo answers one of the questions in Master Prompt section 15, and each one
computes its answer here and now. Where the honest answer is "this library cannot
do that", the demo says so and shows the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preflight import require_dependencies  # noqa: E402

require_dependencies(script="scripts/run_all_demos.py")

import numpy as np  # noqa: E402

from src.crypto.backend import RefreshKind  # noqa: E402
from src.crypto.capabilities import precision_bits  # noqa: E402
from src.crypto.tenseal_backend import get_owner_zone  # noqa: E402
from src.data.loader import load_dataset, prepare  # noqa: E402
from src.experiments.config import ExperimentConfig, Mode  # noqa: E402
from src.experiments.runner import run_single  # noqa: E402

OUT = ROOT / "results" / "demos"


def header(n: int, question: str) -> None:
    print()
    print("=" * 78)
    print(f"DEMO {n}: {question}")
    print("=" * 78)


def load(config: ExperimentConfig):
    dataset = load_dataset(config.dataset)
    return dataset, prepare(
        dataset, test_fraction=config.test_fraction, seed=config.split_seed,
        n_samples=config.n_samples, feature_range=config.feature_range,
    )


def demo1(config: ExperimentConfig) -> dict[str, Any]:
    header(1, "How does CKKS protect the data?")
    _, split = load(config)
    owner = get_owner_zone(config.ckks_params())
    backend = owner.backend()
    record = split.x_train[0]

    print(f"\nThe owner's record (readable here, in the owner's own process):")
    for name, value in zip(split.feature_names, record):
        print(f"    {name:<24} {value: .6f}")

    enc = owner.encrypt(record.tolist())
    blob = enc.raw.serialize()
    print(f"\nWhat the training engine receives instead: {len(blob):,} bytes of ciphertext")
    print(f"    first 40 bytes: {blob[:40].hex()}")

    try:
        enc.raw.decrypt()
        verdict = "FAILED - the compute zone decrypted the record"
    except Exception as exc:  # noqa: BLE001 - the refusal is the demonstration
        verdict = f"blocked ({type(exc).__name__})"
    print(f"\nCompute zone attempting to decrypt it: {verdict}")
    print("    The compute zone holds a public-only context. It has no secret key to use.")

    back = owner.decrypt(enc)
    print(f"\nThe owner decrypts: {np.round(back[: len(record)], 6).tolist()}")
    print(f"    agreement: {precision_bits(back[0], record[0]):.1f} bits")
    return {"ciphertext_bytes": len(blob), "compute_zone_decrypt": verdict}


def demo2(config: ExperimentConfig) -> dict[str, Any]:
    header(2, "Can we compute on encrypted data?")
    owner = get_owner_zone(config.ckks_params())
    be = owner.backend()
    x, y = owner.encrypt([1.5] * 4), owner.encrypt([2.0] * 4)
    rows = []
    for label, fn, exact in [
        ("x + y", lambda: be.add(x, y), 3.5),
        ("x - y", lambda: be.sub(x, y), -0.5),
        ("x * y", lambda: be.mul(x, y), 3.0),
        ("x * y + x", lambda: be.add(be.mul(x, y), x), 4.5),
        ("x^2", lambda: be.square(x), 2.25),
    ]:
        t0 = time.perf_counter()
        res = fn()
        ms = (time.perf_counter() - t0) * 1000
        got = owner.decrypt(res)[0]
        levels = config.ckks_params().max_depth - res.depth
        rows.append({"op": label, "got": got, "exact": exact, "ms": ms, "levels_left": levels})
        print(f"  {label:<12} = {got: .8f}  (exact {exact: .6f}, "
              f"{precision_bits(got, exact):.1f} bits, {ms:6.1f} ms, {levels} level(s) left)")
    print("\n  Operands were never decrypted. Only the results above were, to verify them.")
    return {"operations": rows}


def demo3(config: ExperimentConfig) -> dict[str, Any]:
    header(3, "Can we train an ML model while keeping the data encrypted?")
    _, split = load(config)
    plain = run_single(config, Mode.PLAINTEXT, split)
    enc = run_single(config, Mode.FHE_ADAPTIVE, split)
    print(f"\n  plaintext : accuracy {plain.metrics['final_test_accuracy']:.4f} in "
          f"{plain.metrics['train_seconds']:.3f}s")
    print(f"  encrypted : accuracy {enc.metrics['final_test_accuracy']:.4f} in "
          f"{enc.metrics['train_seconds']:.1f}s  ({enc.status})")
    print(f"  {enc.metrics['encrypted_ops_total']} homomorphic operations; the weights were "
          "ciphertexts throughout and were decrypted only at the end.")
    return {"plaintext": plain.metrics, "encrypted": enc.metrics}


def demo4(config: ExperimentConfig) -> dict[str, Any]:
    header(4, "How does ciphertext noise/capacity change during training?")
    _, split = load(config)
    run = run_single(config, Mode.FHE_ADAPTIVE, split)
    series = run.capacity_series
    print(f"\n  {len(series)} readings. Level is DERIVED; size and precision are MEASURED.")
    print(f"  {'op':>4} {'levels':>7} {'bytes (measured)':>18} {'precision bits':>15}  state")
    for r in series[:12]:
        # Built outside the f-string: nesting the same quote character inside one
        # needs Python 3.12, and this script should run on 3.11 too.
        size = f"{r['serialized_bytes']:,}" if r["serialized_bytes"] else "not measured"
        precision = f"{r['precision_bits']:.1f}" if r["precision_bits"] else "-"
        print(
            f"  {r['index']:>4} {r['levels_remaining']:>7} {size:>18} "
            f"{precision:>15}  {r['state']}"
        )
    cap = run.metrics["capacity"]
    print(f"\n  consistency failures (derived vs measured): {cap['consistency_failures']}")
    print(f"  precision {cap['precision_bits_first']:.1f} -> {cap['precision_bits_last']:.1f} bits")
    return {"capacity": cap}


def demo5(config: ExperimentConfig) -> dict[str, Any]:
    header(5, "When does the adaptive controller decide to bootstrap?")
    _, split = load(config)
    run = run_single(config, Mode.FHE_ADAPTIVE, split)
    print()
    for d in run.decisions:
        mark = ">>" if d["decision"] == "refresh" else "  "
        print(f"  {mark} step {d['iteration']:>3}  {d['decision'].upper():<8} "
              f"levels {d['levels_remaining']}/{d['max_depth']} need {d['levels_needed']}")
        print(f"        {d['reason']}")
    print(f"\n  Primitive: {RefreshKind(run.metrics['refresh_kind']).label}")
    return {"decisions": run.decisions}


def demo6(config: ExperimentConfig) -> dict[str, Any]:
    header(6, "What happens if the threshold changes?")
    _, split = load(config)
    rows = []
    for margin in (0, 1, 2, 3):
        data = config.to_dict()
        data.pop("derived", None)
        data["adaptive_safety_margin"] = margin
        cfg = ExperimentConfig.from_dict(data)
        run = run_single(cfg, Mode.FHE_ADAPTIVE, split)
        rows.append({
            "safety_margin": margin, "status": run.status,
            "refreshes": run.metrics["refreshes"],
            "seconds": run.metrics["train_seconds"],
            "accuracy": run.metrics["final_test_accuracy"],
        })
        print(f"  margin {margin}: {run.status:<10} refreshes={run.metrics['refreshes']:<3} "
              f"time={run.metrics['train_seconds']:6.1f}s "
              f"accuracy={run.metrics['final_test_accuracy']}")
    print("\n  A larger margin refreshes earlier and more often: the threshold is real and its "
          "effect is visible.")
    return {"threshold_sweep": rows}


def demo7(config: ExperimentConfig) -> dict[str, Any]:
    header(7, "How does adaptive behaviour compare with a baseline policy?")
    _, split = load(config)
    rows = []
    per_step = config.model_config(1).depth_per_step(config.batch_size)
    depth = config.ckks_params().max_depth
    for interval in range(1, (depth // per_step) + 3):
        data = config.to_dict()
        data.pop("derived", None)
        data["baseline_interval"] = interval
        cfg = ExperimentConfig.from_dict(data)
        run = run_single(cfg, Mode.FHE_BASELINE, split)
        rows.append({"interval": interval, "status": run.status,
                     "refreshes": run.metrics["refreshes"],
                     "seconds": run.metrics["train_seconds"]})
        print(f"  fixed every {interval} step(s): {run.status:<10} "
              f"refreshes={run.metrics['refreshes']:<3} time={run.metrics['train_seconds']:6.1f}s")
    adaptive = run_single(config, Mode.FHE_ADAPTIVE, split)
    print(f"  adaptive (no tuning):    {adaptive.status:<10} "
          f"refreshes={adaptive.metrics['refreshes']:<3} "
          f"time={adaptive.metrics['train_seconds']:6.1f}s")
    print("\n  The fixed policy has to be tuned: too large an interval fails outright, too small "
          "wastes refreshes. The adaptive policy was given no interval at all.")
    return {"baseline_sweep": rows, "adaptive": adaptive.metrics}


def demo8(config: ExperimentConfig) -> dict[str, Any]:
    header(8, "How does workload size affect encrypted training?")
    dataset, _ = load(config)
    rows = []
    for n in (20, 40, 80):
        if n > dataset.n_samples:
            continue
        data = config.to_dict()
        data.pop("derived", None)
        data["n_samples"] = n
        data["batch_size"] = min(config.batch_size, max(8, n // 3))
        cfg = ExperimentConfig.from_dict(data)
        _, split = load(cfg)
        run = run_single(cfg, Mode.FHE_ADAPTIVE, split)
        rows.append({"samples": n, "seconds": run.metrics["train_seconds"],
                     "steps": run.metrics["steps"], "refreshes": run.metrics["refreshes"],
                     "ops": run.metrics["encrypted_ops_total"],
                     "peak_rss_mb": run.resources.get("peak_rss_mb")})
        print(f"  {n:>4} samples: {run.metrics['train_seconds']:6.1f}s  "
              f"{run.metrics['steps']} steps  {run.metrics['refreshes']} refreshes  "
              f"{run.metrics['encrypted_ops_total']} ops  "
              f"peak RSS {run.resources.get('peak_rss_mb', 0):.0f} MB")
    return {"scalability": rows}


def demo9(config: ExperimentConfig) -> dict[str, Any]:
    header(9, "Can the authorized user recover the final result?")
    _, split = load(config)
    run = run_single(config, Mode.FHE_ADAPTIVE, split)
    model = run.metrics.get("model")
    if not model:
        print("  The run did not complete, so there is no model to decrypt.")
        return {"model": None}
    print("\n  The model existed only as ciphertext during training. Decrypted by the owner:")
    for name, w in zip(split.feature_names, model["weights"]):
        print(f"    {name:<24} {w: .6f}")
    print(f"    {'bias':<24} {model['bias']: .6f}")
    print(f"\n  Test accuracy of the encrypted-trained model: "
          f"{run.metrics['final_test_accuracy']:.4f}")
    print("  An unauthorized party holding the same ciphertexts and no secret key recovers "
          "nothing.")
    return {"model": model, "accuracy": run.metrics["final_test_accuracy"]}


DEMOS: dict[int, tuple[str, Callable[[ExperimentConfig], dict[str, Any]]]] = {
    1: ("How does CKKS protect the data?", demo1),
    2: ("Can we compute on encrypted data?", demo2),
    3: ("Can we train an ML model while keeping the data encrypted?", demo3),
    4: ("How does ciphertext noise/capacity change during training?", demo4),
    5: ("When does the adaptive controller decide to bootstrap?", demo5),
    6: ("What happens if the threshold changes?", demo6),
    7: ("How does adaptive behaviour compare with a baseline policy?", demo7),
    8: ("How does workload size affect encrypted training?", demo8),
    9: ("Can the authorized user recover the final result?", demo9),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "demo.yaml"))
    parser.add_argument("--demo", type=int, action="append", choices=sorted(DEMOS))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for n, (question, _) in DEMOS.items():
            print(f"  {n}. {question}")
        return 0

    config = ExperimentConfig.load(args.config)
    chosen = sorted(set(args.demo)) if args.demo else sorted(DEMOS)
    OUT.mkdir(parents=True, exist_ok=True)

    collected: dict[str, Any] = {}
    for n in chosen:
        question, fn = DEMOS[n]
        try:
            collected[f"demo{n}"] = {"question": question, "result": fn(config)}
        except Exception as exc:  # noqa: BLE001 - a failed demo is reported, not hidden
            print(f"\n  DEMO {n} FAILED: {type(exc).__name__}: {exc}")
            collected[f"demo{n}"] = {"question": question, "error": f"{type(exc).__name__}: {exc}"}

    path = OUT / f"demos-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(collected, indent=2, default=str), encoding="utf-8")
    print(f"\n\nDemo output written to {path}")
    return 0 if all("error" not in v for v in collected.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
