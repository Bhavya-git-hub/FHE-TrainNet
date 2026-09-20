"""Run the refresh-policy comparison inside the container, using real bootstrapping.

This script executes in the OpenFHE image. It exists so the project can answer
"what happens with *actual* CKKS bootstrapping?" with a measurement rather than a
projection.

Two design choices matter:

* It imports `src.bootstrapping.policies` from the mounted repository, so the
  controller deciding when to bootstrap here is byte-for-byte the same code the
  TenSEAL backend uses. Only the primitive differs - `EvalBootstrap` instead of a
  client-aided refresh - which is exactly the comparison worth making.
* It runs the whole experiment in-process and prints one JSON object at the end,
  rather than exposing per-operation calls over a bridge. Shipping megabyte
  ciphertexts across a pipe for every multiplication would measure the pipe.

Everything it cannot do, it reports. If `EvalBootstrap` is unavailable or the API
differs from what is expected, the output says so and the host reports it as
unavailable rather than substituting the client-aided path and calling it
bootstrapping.

    python3 docker/openfhe/openfhe_experiment.py --config-json '{...}'
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, "/work")


def probe_openfhe() -> dict[str, Any]:
    """Establish what this OpenFHE build actually provides, by calling it."""
    report: dict[str, Any] = {"library": "openfhe", "probes": {}}
    try:
        import openfhe
    except Exception as exc:  # noqa: BLE001 - reported to the host verbatim
        report["available"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["available"] = True
    report["version"] = getattr(openfhe, "__version__", "unknown")
    for name in (
        "CCParamsCKKSRNS", "GenCryptoContext", "FHECKKSRNS",
        "UNIFORM_TERNARY", "PKE", "KEYSWITCH", "LEVELEDSHE", "ADVANCEDSHE", "FHE",
    ):
        report["probes"][name] = hasattr(openfhe, name)

    cc_methods = ["EvalBootstrap", "EvalBootstrapSetup", "EvalBootstrapKeyGen"]
    try:
        params = openfhe.CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(3)
        params.SetScalingModSize(50)
        cc = openfhe.GenCryptoContext(params)
        for name in cc_methods:
            report["probes"][f"context.{name}"] = hasattr(cc, name)
    except Exception as exc:  # noqa: BLE001
        report["probes"]["context_creation_error"] = f"{type(exc).__name__}: {exc}"

    report["supports_native_bootstrap"] = all(
        report["probes"].get(f"context.{n}", False) for n in cc_methods
    )
    return report


# OpenFHE caps the correction factor; values above this are rejected outright.
MAX_CORRECTION_FACTOR = 27


def build_context(cfg: dict[str, Any]) -> dict[str, Any]:
    """Create a CKKS context configured for bootstrapping.

    Bootstrapping is not free to set up: it needs its own depth budget on top of
    the levels the computation actually uses, and `GetBootstrapDepth` is what says
    how much. Getting this wrong is the usual reason an OpenFHE bootstrapping
    example fails, so the numbers are computed rather than guessed.
    """
    import openfhe

    level_budget = list(cfg.get("level_budget", [4, 4]))
    usable_levels = int(cfg.get("usable_levels", 10))
    ring_dim = int(cfg.get("ring_dim", 1 << 16))
    scaling_mod = int(cfg.get("scaling_mod_size", 59))

    secret_dist = openfhe.UNIFORM_TERNARY
    bootstrap_depth = openfhe.FHECKKSRNS.GetBootstrapDepth(level_budget, secret_dist)
    depth = usable_levels + bootstrap_depth

    params = openfhe.CCParamsCKKSRNS()
    params.SetSecretKeyDist(secret_dist)
    params.SetRingDim(ring_dim)
    params.SetScalingModSize(scaling_mod)
    params.SetFirstModSize(60)
    params.SetMultiplicativeDepth(depth)
    try:
        params.SetScalingTechnique(openfhe.FLEXIBLEAUTO)
    except Exception:  # noqa: BLE001 - older builds may not expose it; not fatal
        pass

    started = time.perf_counter()
    try:
        cc = openfhe.GenCryptoContext(params)
    except Exception as exc:  # noqa: BLE001 - re-raised with what to change
        message = str(exc)
        if "ring dimension" in message or "HE standards" in message:
            # OpenFHE enforces the 128-bit security tables and refuses parameters
            # that break them. That is the library doing its job, so the fix is to
            # ask for less depth or a bigger ring - never to lower the security
            # level, which would make every number produced here meaningless.
            raise RuntimeError(
                "OpenFHE refused these parameters for security reasons: "
                + message
                + f" | Bootstrapping reserves {bootstrap_depth} level(s) on top of the "
                f"{usable_levels} requested, giving a multiplicative depth of {depth}, "
                f"which needs a larger ring than {ring_dim}."
                " | Either lower --usable-levels (and --depth-per-step with it, to keep"
                " the same steps-between-bootstraps ratio), or raise --ring-dim to the"
                " value OpenFHE names above. Do not lower the security level to fit."
            ) from exc
        raise
    for feature in (openfhe.PKE, openfhe.KEYSWITCH, openfhe.LEVELEDSHE,
                    openfhe.ADVANCEDSHE, openfhe.FHE):
        cc.Enable(feature)

    num_slots = int(cfg.get("num_slots", ring_dim // 2))

    # The correction factor bounds the ciphertext depth EvalBootstrap will accept:
    # OpenFHE refuses with "Degree [D] must be less than or equal to the correction
    # factor [C]" when the total multiplicative depth exceeds it. Left at 0 it is
    # auto-derived and came out at 11 against a depth of 20, so it is set
    # explicitly here. It cannot simply be raised without limit - OpenFHE caps it -
    # so the request is clamped and the value actually used is reported.
    correction_factor = int(cfg.get("correction_factor", 0))
    if correction_factor == 0:
        correction_factor = min(MAX_CORRECTION_FACTOR, depth + 1)
    correction_factor = max(0, min(MAX_CORRECTION_FACTOR, correction_factor))

    try:
        cc.EvalBootstrapSetup(level_budget, [0, 0], num_slots, correction_factor)
    except Exception as exc:  # noqa: BLE001 - fall back to the auto value, and say so
        setup_note = (
            f"EvalBootstrapSetup rejected correction_factor={correction_factor} "
            f"({type(exc).__name__}: {exc}); retried with the library default."
        )
        cc.EvalBootstrapSetup(level_budget, [0, 0], num_slots)
        correction_factor = 0
    else:
        setup_note = None
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapKeyGen(keys.secretKey, num_slots)
    setup_seconds = time.perf_counter() - started

    return {
        "cc": cc,
        "keys": keys,
        "num_slots": num_slots,
        "usable_levels": usable_levels,
        "bootstrap_depth": bootstrap_depth,
        "total_depth": depth,
        "setup_seconds": setup_seconds,
        "ring_dim": ring_dim,
        "correction_factor": correction_factor,
        "setup_note": setup_note,
    }


def run_policy_comparison(cfg: dict[str, Any]) -> dict[str, Any]:
    """Consume levels under each policy, bootstrapping for real when told to.

    The workload is a chain of multiplications that costs `depth_per_step` levels
    per step - the same shape the encrypted trainer's step has, and the quantity
    the controller reasons about. What is being measured here is the policy and
    the cost of the primitive, not a second training implementation.
    """
    import numpy as np
    import openfhe

    from src.bootstrapping.policies import (
        AdaptivePolicy,
        BaselinePolicy,
        Decision,
        DecisionLog,
        DecisionRecord,
    )
    from src.noise.monitor import CapacityReading, CapacityState

    ctx = build_context(cfg)
    cc, keys = ctx["cc"], ctx["keys"]
    num_slots = ctx["num_slots"]
    usable = ctx["usable_levels"]
    depth_per_step = int(cfg.get("depth_per_step", 5))
    steps = int(cfg.get("steps", 12))

    # The probe value is 1.0 and the level-consuming operation is squaring, because
    # 1.0 squared is 1.0: the value is preserved exactly while each multiplication
    # still spends a level. An earlier version squared 0.25, which underflows to
    # noise within a few steps and makes the decrypted check meaningless.
    values = [1.0] * min(8, num_slots)
    results: dict[str, Any] = {}

    def level_of(ciphertext) -> int | None:
        """OpenFHE's own level, where the build exposes it.

        This is the real difference between the two backends: OpenFHE reports the
        level, so here it is MEASURED. TenSEAL exposes no such accessor, so there
        it has to be derived and cross-checked against ciphertext size.
        """
        for name in ("GetLevel", "GetTowersLeft"):
            getter = getattr(ciphertext, name, None)
            if getter is not None:
                try:
                    return int(getter())
                except Exception:  # noqa: BLE001 - absence is reported as None
                    return None
        return None

    for policy_name, policy in (
        ("fhe_baseline", BaselinePolicy(interval=int(cfg.get("baseline_interval", 1)))),
        ("fhe_adaptive", AdaptivePolicy(safety_margin=int(cfg.get("safety_margin", 0)))),
    ):
        log = DecisionLog()
        # Encoded at level 0 - a FRESH ciphertext with the whole chain ahead of it.
        # The OpenFHE bootstrapping examples encode at the BOTTOM of the chain to
        # show a bootstrap lifting it; here the point is the opposite, to spend the
        # chain down and let the controller decide when to lift it.
        ptxt = cc.MakeCKKSPackedPlaintext(values, 1, 0, None, num_slots)
        ct = cc.Encrypt(keys.publicKey, ptxt)
        consumed = 0
        bootstrap_seconds = 0.0
        measured_levels: list[int | None] = []
        started = time.perf_counter()
        status, reason = "SUCCEEDED", None

        try:
            for step in range(steps):
                reading = CapacityReading(
                    index=step, label=f"step{step}", iteration=step, epoch=0,
                    levels_consumed=consumed, levels_remaining=usable - consumed,
                    max_depth=usable, serialized_bytes=None,
                    measured_levels=level_of(ct), precision_bits=None,
                    state=CapacityState.SAFE,
                )
                decision, why = policy.decide(reading, depth_per_step, step)
                seconds = None
                if decision is Decision.REFRESH:
                    t0 = time.perf_counter()
                    ct = cc.EvalBootstrap(ct)   # actual CKKS bootstrapping
                    seconds = time.perf_counter() - t0
                    bootstrap_seconds += seconds
                    consumed = 0
                log.add(
                    DecisionRecord(
                        index=0, iteration=step, epoch=0, decision=decision, reason=why,
                        policy=policy.name, levels_remaining=usable - consumed,
                        levels_needed=depth_per_step, max_depth=usable,
                        threshold=policy.threshold_text(),
                        refresh_kind="native_bootstrap", refresh_seconds=seconds,
                    )
                )
                for _ in range(depth_per_step):
                    # Squaring 1.0 leaves 1.0 and costs one level. No manual
                    # Rescale: FLEXIBLEAUTO rescales inside EvalMult, and calling
                    # it again would spend a second level per multiplication.
                    ct = cc.EvalMult(ct, ct)
                    consumed += 1
                measured_levels.append(level_of(ct))
        except Exception as exc:  # noqa: BLE001 - recorded as a failed run, with its reason
            status = "FAILED"
            reason = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - started
        decrypted = None
        precision_bits = None
        try:
            out = cc.Decrypt(ct, keys.secretKey)
            out.SetLength(len(values))
            decrypted = [float(v) for v in out.GetRealPackedValue()][:4]
            if decrypted:
                error = abs(decrypted[0] - 1.0)
                precision_bits = None if error == 0 else -math.log2(error)
        except Exception as exc:  # noqa: BLE001
            reason = f"{reason + '; ' if reason else ''}final decryption failed: {exc}"

        summary = log.summary()
        results[policy_name] = {
            "status": status,
            "reason": reason,
            "seconds": elapsed,
            "bootstraps": summary["refreshes"],
            "continues": summary["continues"],
            "bootstrap_seconds_total": bootstrap_seconds,
            "bootstrap_seconds_mean": summary["refresh_seconds_mean"],
            "decisions": log.rows(),
            "decrypted_head": decrypted,
            "precision_bits": precision_bits,
            "measured_levels": measured_levels,
            "refresh_kind": "native_bootstrap",
        }

    return {
        "context": {
            "ring_dim": ctx["ring_dim"],
            "usable_levels": usable,
            "bootstrap_depth": ctx["bootstrap_depth"],
            "total_depth": ctx["total_depth"],
            "num_slots": num_slots,
            "setup_seconds": ctx["setup_seconds"],
            "correction_factor": ctx["correction_factor"],
            "setup_note": ctx["setup_note"],
        },
        "depth_per_step": depth_per_step,
        "steps": steps,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="report capabilities and exit")
    parser.add_argument("--config-json", default="{}")
    args = parser.parse_args()

    if args.probe:
        print(json.dumps(probe_openfhe(), indent=2))
        return 0

    cfg = json.loads(args.config_json)
    probe = probe_openfhe()
    if not probe.get("supports_native_bootstrap"):
        print(json.dumps({
            "ok": False,
            "reason": "this OpenFHE build does not expose EvalBootstrap",
            "probe": probe,
        }))
        return 1

    try:
        payload = run_policy_comparison(cfg)
        print(json.dumps({"ok": True, "probe": probe, **payload}, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001 - failure is reported, never silently downgraded
        print(json.dumps({
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "probe": probe,
        }))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
