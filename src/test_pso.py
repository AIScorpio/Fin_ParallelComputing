"""
PSO & LSMC American Option Pricing - Multi-case test runner.
Run with: python test_pso.py
"""

import os
import sys
import numpy as np
from pathlib import Path

# Path setup — also chdir so relative kernel paths in longstaff.py resolve correctly
project_root = Path(__file__).parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
# sys.path.insert(0, str(project_root / "src"))

from models.mc import hybridMonteCarlo
import models.benchmarks as bm
from models.longstaff import LSMC_OpenCL
from models.pso import (
    PSO_Numpy,
    PSO_OpenCL_hybrid,
    PSO_OpenCL_scalar,
    PSO_OpenCL_scalar_fusion,
    PSO_OpenCL_vec,
    PSO_OpenCL_vec_fusion,
)

# ── Test cases ─────────────────────────────────────────────────────────────────
# S&P 500 index options, Sep 2008 (secid=108105, T=30 days for all cases)
# Columns: name, S0(close), K, r(_3_MO/100), sigma, T(days/365),
#          opttype, impl_premium, nPath, nPeriod, nFish
#
# Market regimes:
#   02SEP08 – pre-Lehman,  S0=1277.58, r=1.72%  (low vol, mild skew)
#   15SEP08 – Lehman day,  S0=1192.70, r=1.02%  (vol spike)
#   29SEP08 – TARP reject, S0=1106.42, r=0.94%  (max stress, steep skew)
#
# Moneyness guide (abs delta):  ≈10 → deep OTM | ≈25 → OTM | ≈50 → ATM
#                                ≈75 → ITM      | ≈90 → deep ITM
#
# (* marked) impl_premium computed from Black-Scholes with vol consistent
# with the surrounding market data; all others are real market premiums.
#
# Pairs are grouped: each moneyness bucket has one Put and one Call.

# Hyperparameters
N_PATH = 10000  # Simulation paths - tunable
N_NUMPY_ITER = (
    5  # PSO_Numpy iterations — limited for speed; used for timing baseline only
)

TEST_CASES = [
    # ── ATM pair — pre-Lehman (02SEP08) ──────────────────────────────────────
    (
        "ATM Put   (d=-50, 02SEP08)",
        1277.58,
        1280.749,
        0.0172,
        0.213411,
        30 / 365,
        "P",
        32.4512,
        N_PATH,
        200,
        256,
    ),
    (
        "ATM Call  (d=+50, 02SEP08)",
        1277.58,
        1280.343,
        0.0172,
        0.209699,
        30 / 365,
        "C",
        29.5502,
        N_PATH,
        200,
        256,
    ),
    # ── ATM pair — TARP rejection (29SEP08, high-vol ATM) ────────────────────
    (
        "ATM Put   (d=-50, 29SEP08)",
        1106.42,
        1106.000,
        0.0094,
        0.275000,
        30 / 365,
        "P",
        34.0100,
        N_PATH,
        200,
        256,
    ),  # *
    (
        "ATM Call  (d=+50, 29SEP08)",
        1106.42,
        1106.000,
        0.0094,
        0.275000,
        30 / 365,
        "C",
        35.2800,
        N_PATH,
        200,
        256,
    ),  # *
    # ── OTM pair — Lehman collapse (15SEP08) ─────────────────────────────────
    (
        "OTM Put   (d=-25, 15SEP08)",
        1192.70,
        1126.930,
        0.0102,
        0.317198,
        30 / 365,
        "P",
        16.9545,
        N_PATH,
        200,
        256,
    ),
    (
        "OTM Call  (d=+25, 15SEP08)",
        1192.70,
        1265.262,
        0.0102,
        0.286691,
        30 / 365,
        "C",
        14.0483,
        N_PATH,
        200,
        256,
    ),
    # ── ITM pair — Lehman collapse (15SEP08, vol spiking) ────────────────────
    (
        "ITM Put   (d=-75, 15SEP08)",
        1192.70,
        1267.000,
        0.0102,
        0.290000,
        30 / 365,
        "P",
        87.3300,
        N_PATH,
        200,
        256,
    ),  # *
    (
        "ITM Call  (d=+75, 15SEP08)",
        1192.70,
        1125.000,
        0.0102,
        0.330000,
        30 / 365,
        "C",
        86.4800,
        N_PATH,
        200,
        256,
    ),  # *
    # ── ITM pair — TARP rejection (29SEP08) ──────────────────────────────────
    (
        "ITM Put   (d=-75, 29SEP08)",
        1106.42,
        1174.136,
        0.0094,
        0.282437,
        30 / 365,
        "P",
        78.9833,
        N_PATH,
        200,
        256,
    ),
    (
        "ITM Call  (d=+75, 29SEP08)",
        1106.42,
        1040.000,
        0.0094,
        0.350000,
        30 / 365,
        "C",
        84.7200,
        N_PATH,
        200,
        256,
    ),  # *
    # ── Very-OTM pair — TARP rejection (29SEP08, steep skew) ─────────────────
    (
        "vOTM Put  (d=-15, 29SEP08)",
        1106.42,
        987.000,
        0.0094,
        0.410000,
        30 / 365,
        "P",
        10.7400,
        N_PATH,
        200,
        256,
    ),  # *
    (
        "vOTM Call (d=+15, 29SEP08)",
        1106.42,
        1260.221,
        0.0094,
        0.411432,
        30 / 365,
        "C",
        9.6337,
        N_PATH,
        200,
        256,
    ),
    # ── Deep-OTM pair — pre-Lehman (02SEP08) ─────────────────────────────────
    (
        "dOTM Put  (d=-10, 02SEP08)",
        1277.58,
        1155.000,
        0.0172,
        0.270000,
        30 / 365,
        "P",
        4.0100,
        N_PATH,
        200,
        256,
    ),  # *
    (
        "dOTM Call (d=+10, 02SEP08)",
        1277.58,
        1365.000,
        0.0172,
        0.175000,
        30 / 365,
        "C",
        2.9400,
        N_PATH,
        200,
        256,
    ),  # *
    # ── Deep-ITM pair — TARP rejection (29SEP08) ─────────────────────────────
    (
        "dITM Put  (d=-90, 29SEP08)",
        1106.42,
        1217.000,
        0.0094,
        0.250000,
        30 / 365,
        "P",
        113.4100,
        N_PATH,
        150,
        256,
    ),  # *
    (
        "dITM Call (d=+90, 29SEP08)",
        1106.42,
        966.000,
        0.0094,
        0.390000,
        30 / 365,
        "C",
        147.1600,
        N_PATH,
        150,
        256,
    ),  # *
    # ── Synthetic reference cases (reproduced from initial test report) ────────
    # impl_premium ≈ binomial price from that run (used as market proxy for vs-Mkt column)
    # ATM Put (d=-50) was the failing case in the old codebase — verifies the bug is fixed
    (
        "OTM Put   (S0=100,K=110,T=1y)",
        100.0,
        110.0,
        0.03,
        0.30,
        1.0,
        "P",
        16.51,
        20000,
        200,
        256,
    ),
    (
        "ATM Put   (S0=22.7,K=22.7,T=60d)",
        22.7389,
        22.7389,
        0.0102,
        0.502026,
        60 / 365,
        "P",
        1.82,
        10000,
        250,
        256,
    ),
    (
        "ITM Put   (S0=100,K=105,T=1y)",
        100.0,
        105.0,
        0.03,
        0.20,
        1.0,
        "P",
        9.48,
        20000,
        200,
        256,
    ),
    (
        "OTM Call  (S0=100,K=105,T=1y)",
        100.0,
        105.0,
        0.05,
        0.25,
        1.0,
        "C",
        10.01,
        20000,
        200,
        256,
    ),
]

TOLERANCE = 0.15  # allow up to 15% relative error vs binomial (MC variance at these path counts)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _rel_err(price, ref):
    return abs(price - ref) / ref if ref > 1e-8 else abs(price - ref)


def _pass(price, ref):
    return "PASS" if _rel_err(price, ref) <= TOLERANCE else "FAIL"


def run_case(name, S0, K, r, sigma, T, opttype, impl_premium, nPath, nPeriod, nFish):
    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(
        f"  S0={S0:.4f}  K={K:.4f}  r={r:.4f}  sigma={sigma:.4f}  "
        f"T={T:.4f}y  nPath={nPath}  nPeriod={nPeriod}  nFish={nFish}"
    )
    print(f"  Market price (impl_premium): {impl_premium:.4f}")
    print(f"{'=' * 72}")

    # ── Reference ─────────────────────────────────────────────────────────────
    binomial, t_bin = bm.binomialAmericanOption(S0, K, r, sigma, nPeriod, T, opttype)

    # ── Monte Carlo setup ─────────────────────────────────────────────────────
    mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)

    lsmc_results = []
    pso_results = []
    cpu_results = []  # NumPy CPU baseline — timing only (N_NUMPY_ITER iters)

    # ── PSO CPU NumPy (timing baseline) ───────────────────────────────────────
    pso_np = PSO_Numpy(mc, nFish, iterMax=N_NUMPY_ITER)
    p_np, t_np, _, _, _ = pso_np.solvePsoAmerOption_np()
    cpu_results.append(("PSO CPU numpy(*)", float(p_np), t_np))

    # ── LSMC GPU ──────────────────────────────────────────────────────────────
    lsmc_cl = LSMC_OpenCL(mc, preCalc="optimized")
    p_lsmc_cl, t_lsmc_cl = lsmc_cl.longstaff_schwartz_itm_path_fast_hybrid()
    lsmc_results.append(("LSMC GPU", float(p_lsmc_cl), t_lsmc_cl))

    # ── PSO GPU ───────────────────────────────────────────────────────────────
    # pso_hy = PSO_OpenCL_hybrid(mc, nFish)
    # p_hy, t_hy, _, _, _ = pso_hy.solvePsoAmerOption_cl()
    # pso_results.append(("PSO GPU hybrid",     float(p_hy), t_hy))

    pso_sc = PSO_OpenCL_scalar(mc, nFish)
    p_sc, t_sc, _, _, _ = pso_sc.solvePsoAmerOption_cl()
    iter_info = f"({len(pso_sc.BestCosts)}/{pso_sc.iterMax})"
    pso_results.append((f"PSO GPU scalar {iter_info}", float(p_sc), t_sc))

    pso_sf = PSO_OpenCL_scalar_fusion(mc, nFish)
    p_sf, t_sf, _, _ = pso_sf.solvePsoAmerOption_cl()
    iter_info = f"({len(pso_sf.BestCosts)}/{pso_sf.iterMax})"
    pso_results.append((f"PSO GPU sc_fusion {iter_info}", float(p_sf), t_sf))

    pso_vc = PSO_OpenCL_vec(mc, nFish, vec_size=4)
    p_vc, t_vc, _, _, _ = pso_vc.solvePsoAmerOption_cl()
    iter_info = f"({len(pso_vc.BestCosts)}/{pso_vc.iterMax})"
    pso_results.append((f"PSO GPU vec4 {iter_info}", float(p_vc), t_vc))

    pso_vc = PSO_OpenCL_vec(mc, nFish, vec_size=8)
    p_vc, t_vc, _, _, _ = pso_vc.solvePsoAmerOption_cl()
    iter_info = f"({len(pso_vc.BestCosts)}/{pso_vc.iterMax})"
    pso_results.append((f"PSO GPU vec8 {iter_info}", float(p_vc), t_vc))

    pso_vc = PSO_OpenCL_vec(mc, nFish, vec_size=16)
    p_vc, t_vc, _, _, _ = pso_vc.solvePsoAmerOption_cl()
    iter_info = f"({len(pso_vc.BestCosts)}/{pso_vc.iterMax})"
    pso_results.append((f"PSO GPU vec16 {iter_info}", float(p_vc), t_vc))

    pso_vf = PSO_OpenCL_vec_fusion(mc, nFish, vec_size=4)
    p_vf, t_vf, _, _ = pso_vf.solvePsoAmerOption_cl()
    iter_info = f"({len(pso_vf.BestCosts)}/{pso_vf.iterMax})"
    pso_results.append((f"PSO GPU vec4_fusion {iter_info}", float(p_vf), t_vf))

    mc.cleanUp()

    # ── Print table ───────────────────────────────────────────────────────────
    W = 66
    print(
        f"\n  {'Method':<24} {'Price':>9}  {'Time':>10}  {'vs Bin':>7}  {'vs Mkt':>7}  Status"
    )
    print(f"  {'-' * W}")
    print(f"  {'Binomial (ref)':<24} {binomial:>9.4f}  {t_bin:>9.1f} ms")
    print(f"  {'Market (impl_prem)':<24} {impl_premium:>9.4f}")
    print(f"  {'-' * W}")
    for group_label, group in [
        ("── LSMC GPU", lsmc_results),
        ("── PSO GPU", pso_results),
    ]:
        print(f"  {group_label}")
        for label, price, elapsed_ms in group:
            err_bin = _rel_err(price, binomial)
            err_mkt = _rel_err(price, impl_premium)
            status = _pass(price, binomial)
            print(
                f"  {label:<24} {price:>9.4f}  {elapsed_ms:>9.1f} ms  {err_bin:>6.2%}  {err_mkt:>6.2%}  {status}"
            )
    # CPU NumPy — timing only, price not fully converged at N_NUMPY_ITER iters
    print(f"  ── PSO CPU baseline ({N_NUMPY_ITER} iters)")
    for label, price, elapsed_ms in cpu_results:
        err_bin = _rel_err(price, binomial)
        print(
            f"  {label:<24} {price:>9.4f}  {elapsed_ms:>9.1f} ms  {err_bin:>6.2%}  {'(timing only)':>7}"
        )
    # Print GPU scalar vs CPU ratio
    sc_t = next(
        (t for lbl, _, t in pso_results if "scalar" in lbl and "fusion" not in lbl),
        None,
    )
    vc_t = next((t for lbl, _, t in pso_results if "vec4" in lbl), None)
    np_t = cpu_results[0][2] if cpu_results else None
    if sc_t and np_t:
        ratio_full = (np_t / N_NUMPY_ITER * 30) / sc_t  # extrapolate numpy to 30 iters
        print(f"  [speedup] GPU scalar vs NumPy (extrapolated): {ratio_full:.1f}x")
    if vc_t and np_t:
        ratio_full = (np_t / N_NUMPY_ITER * 30) / vc_t
        print(f"  [speedup] GPU vec    vs NumPy (extrapolated): {ratio_full:.1f}x")

    all_results = lsmc_results + pso_results + cpu_results
    return all_results, float(binomial), float(impl_premium)


def write_report(summary, report_path):
    from datetime import datetime

    lines = []
    lines.append("# PSO & LSMC American Option Pricing — Test Report")
    lines.append(f"\n**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(
        f"**Tolerance:** {TOLERANCE * 100:.0f}% relative error vs Binomial tree\n"
    )

    overall = all(ok for _, _, _, _, _, ok in summary)
    lines.append(
        f"## Overall result: {'✅ ALL PASSED' if overall else '❌ SOME FAILED'}\n"
    )

    # ── Per-case sections ──────────────────────────────────────────────────
    for name, (
        S0,
        K,
        r,
        sigma,
        T,
        opttype,
        _,
        nPath,
        nPeriod,
        nFish,
    ), binomial, mkt, results, ok in summary:
        status_icon = "✅" if ok else "❌"
        lines.append(f"---\n")
        lines.append(f"## {status_icon} {name}\n")
        lines.append(f"| Parameter | Value |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| S0 (close) | {S0} |")
        lines.append(f"| K (impl_strike) | {K} |")
        lines.append(f"| r (_3_MO/100) | {r} |")
        lines.append(f"| σ (impl_vol) | {sigma} |")
        lines.append(f"| T  | {T:.4f} yr |")
        lines.append(f"| Type | {opttype} |")
        lines.append(f"| Market price (impl_premium) | {mkt:.4f} |")
        lines.append(f"| nPath | {nPath} |")
        lines.append(f"| nPeriod | {nPeriod} |")
        lines.append(f"| nFish | {nFish} |")
        lines.append("")
        lines.append(
            f"| Method | Price | Time (ms) | vs Binomial | vs Market | Status |"
        )
        lines.append(
            f"|--------|------:|----------:|------------:|----------:|--------|"
        )
        lines.append(
            f"| **Binomial (ref)** | **{binomial:.4f}** | — | — | {_rel_err(binomial, mkt):.2%} | — |"
        )
        lines.append(
            f"| *Market (impl_premium)* | *{mkt:.4f}* | — | {_rel_err(mkt, binomial):.2%} | — | — |"
        )

        prev_group = None
        for label, price, elapsed_ms in results:
            if "LSMC" in label:
                group = "LSMC GPU"
            elif "CPU" in label:
                group = f"PSO CPU (timing only, {N_NUMPY_ITER} iters)"
            else:
                group = "PSO GPU"
            if group != prev_group:
                lines.append(f"| *{group}* | | | | | |")
                prev_group = group
            err_bin = _rel_err(price, binomial)
            err_mkt = _rel_err(price, mkt)
            if "CPU" in label:
                icon = "—"  # not evaluated: timing-only run
            else:
                icon = "✅" if _pass(price, binomial) == "PASS" else "❌"
            lines.append(
                f"| {label} | {price:.4f} | {elapsed_ms:.1f} | {err_bin:.2%} | {err_mkt:.2%} | {icon} |"
            )

        # 添加 Speedup 信息（标注具体变体，最高的用橙色加粗）
        lsmc_time = next((t for lbl, _, t in results if "LSMC" in lbl), None)

        # 找出最快的 scalar 变体（包含 scalar 和 sc_fusion）
        scalar_variants = [
            (lbl, t)
            for lbl, _, t in results
            if ("scalar" in lbl.lower() or "sc_fusion" in lbl.lower())
            and "vec" not in lbl.lower()
        ]

        # 找出最快的 vec 变体
        vec_variants = [(lbl, t) for lbl, _, t in results if "vec" in lbl.lower()]

        if lsmc_time:
            speedups = []

            # 计算 scalar speedup
            if scalar_variants:
                fastest_sc = min(scalar_variants, key=lambda x: x[1])
                sc_name = fastest_sc[0].replace("PSO GPU ", "").split()[0]
                sc_speedup = lsmc_time / fastest_sc[1]
                speedups.append(("scalar", sc_name, fastest_sc[1], sc_speedup))

            # 计算 vec speedup
            if vec_variants:
                fastest_vc = min(vec_variants, key=lambda x: x[1])
                vc_name = fastest_vc[0].replace("PSO GPU ", "").split()[0]
                vc_speedup = lsmc_time / fastest_vc[1]
                speedups.append(("vec", vc_name, fastest_vc[1], vc_speedup))

            # 找出最高的 speedup
            if speedups:
                max_speedup = max(speedups, key=lambda x: x[3])

                for variant_type, variant_name, time_val, speedup_val in speedups:
                    is_max = speedup_val == max_speedup[3]
                    if is_max:
                        # 最高的用 ⭐ 标记并加粗
                        lines.append(
                            f"| *Speedup: {variant_name} vs LSMC GPU* | — | **{speedup_val:.2f}x** | — | — | ⭐ |"
                        )
                    else:
                        lines.append(
                            f"| *Speedup: {variant_name} vs LSMC GPU* | — | {speedup_val:.2f}x | — | — | — |"
                        )

        lines.append("")

    # ── Summary table ──────────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Summary\n")
    lines.append(
        "| Test Case | Binomial | Market | Avg LSMC GPU | Avg PSO GPU | PSO Speedup vs LSMC | Result |"
    )
    lines.append(
        "|-----------|--------:|-------:|-------------:|------------:|--------------------:|--------|"
    )
    for name, _, binomial, mkt, results, ok in summary:
        icon = "✅" if ok else "❌"
        lsmc_p = [p for lbl, p, _ in results if "LSMC" in lbl]
        pso_p = [p for lbl, p, _ in results if "PSO" in lbl and "CPU" not in lbl]
        avg_lsmc = f"{np.mean(lsmc_p):.4f}" if lsmc_p else "—"
        avg_pso = f"{np.mean(pso_p):.4f}" if pso_p else "—"
        # 计算 PSO vs LSMC 的最高 speedup
        lsmc_t = next((t for lbl, _, t in results if "LSMC" in lbl), None)
        pso_times = [t for lbl, _, t in results if "PSO" in lbl and "CPU" not in lbl]
        if lsmc_t and pso_times:
            min_pso_t = min(pso_times)
            max_speedup = lsmc_t / min_pso_t
            speedup_str = f"{max_speedup:.2f}x"
        else:
            speedup_str = "—"
        lines.append(
            f"| {name} | {binomial:.4f} | {mkt:.4f} | {avg_lsmc} | {avg_pso} | {speedup_str} | {icon} |"
        )

    lines.append(f"\n*Generated by `test_pso.py`*\n")

    report_path.write_text("\n".join(lines))
    print(f"\n  Report written to: {report_path}")


def main():
    from models.utils import checkOpenCL

    checkOpenCL()

    all_pass = True
    summary = []  # (name, params_tuple, binomial, mkt, results, case_pass)

    for case in TEST_CASES:
        name = case[0]
        params = case[
            1:
        ]  # (S0, K, r, sigma, T, opttype, impl_premium, nPath, nPeriod, nFish)
        results, binomial, mkt = run_case(name, *params)
        case_pass = all(
            _pass(p, binomial) == "PASS" for lbl, p, _ in results if "CPU" not in lbl
        )
        all_pass = all_pass and case_pass
        summary.append((name, params, binomial, mkt, results, case_pass))

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    for name, _, binomial, mkt, results, ok in summary:
        status = "PASS" if ok else "FAIL"
        lsmc_prices = [p for lbl, p, _ in results if "LSMC" in lbl]
        pso_prices = [p for lbl, p, _ in results if "PSO" in lbl and "CPU" not in lbl]
        avg_lsmc = np.mean(lsmc_prices) if lsmc_prices else float("nan")
        avg_pso = np.mean(pso_prices) if pso_prices else float("nan")
        print(
            f"  [{status}]  {name:<40}  binomial={binomial:.4f}  mkt={mkt:.4f}"
            f"  lsmc={avg_lsmc:.4f}  pso={avg_pso:.4f}"
        )

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'=' * 72}\n")

    # ── Markdown report ───────────────────────────────────────────────────────
    write_report(summary, project_root / "test_report.md")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
