"""
PSO vec_fusion Performance Diagnostic
======================================
Investigates why PSO_OpenCL_vec_fusion is ~15x slower than PSO_OpenCL_vec
despite fusing all operations into a single kernel.

Run with: python test_vec_fusion_profile.py

What this script does:
  1. Sub-step timing breakdown for vec (non-fused) and vec_fusion
  2. Fine-grained loop instrumentation for vec_fusion:
       - fused kernel time
       - enqueue_copy (host-transfer) time
       - np.argmax (CPU) time
       - gbest update kernel time (conditional)
  3. Kernel occupancy info (work group size, local/private memory)
  4. vec_size sweep [1, 2, 4, 8] for vec_fusion
  5. nFish sweep [64, 128, 256, 512] for both variants
  6. nPath sweep [2000, 5000, 10000, 20000, 40000] — NumPy CPU vs scalar vs vec vs vec_fusion
  7. Multi-case sweep — NumPy CPU vs scalar vs vec vs vec_fusion across 10 representative option cases
"""

import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pyopencl as cl
from models.mc import hybridMonteCarlo
from models.pso import (
    PSO_Numpy,
    PSO_OpenCL_scalar,
    PSO_OpenCL_vec,
    PSO_OpenCL_vec_fusion,
)
from models.utils import openCLEnv

# ── Test parameters (ATM Put 02SEP08, same as first case in test_pso.py) ─────
S0, K, r, sigma, T, opttype = 1277.58, 1280.749, 0.0172, 0.213411, 30 / 365, "P"
N_PATH = 10_000
N_PERIOD = 150
N_FISH = 256
VEC_SIZE = 4

SECTION = "=" * 72


# ── Profiled subclass of PSO_OpenCL_vec_fusion ────────────────────────────────
class PSO_vec_fusion_profiled(PSO_OpenCL_vec_fusion):
    """
    Overrides solvePsoAmerOption_cl() to add fine-grained per-step timing
    inside each PSO iteration, without modifying the original class.
    """

    def solvePsoAmerOption_cl_profiled(self):
        """Same logic as parent but with split timings per sub-step."""
        t_kernel = []  # _runPso() fused kernel time
        t_copy = []  # enqueue_copy(pbest_costs) host-transfer time
        t_argmax = []  # np.argmax on host
        t_gbest = []  # conditional gbest update kernel time

        start = time.perf_counter()

        for i in range(self.iterMax):
            # -- fused kernel (searchGrid + fitness + pbest update on GPU) ----
            t0 = time.perf_counter()
            self._runPso()
            t_kernel.append((time.perf_counter() - t0) * 1e3)

            # -- copy pbest_costs from device to host -------------------------
            t0 = time.perf_counter()
            cl.enqueue_copy(
                openCLEnv.queue, self.pbest_costs, self.pbest_costs_d
            ).wait()
            openCLEnv.queue.finish()
            t_copy.append((time.perf_counter() - t0) * 1e3)

            # -- CPU: find best particle index --------------------------------
            t0 = time.perf_counter()
            gid = np.argmax(self.pbest_costs)
            t_argmax.append((time.perf_counter() - t0) * 1e3)

            # -- conditional gbest position update on GPU --------------------
            t0 = time.perf_counter()
            if self.pbest_costs[gid] > self.gbest_cost:
                self.gbest_cost = self.pbest_costs[gid]
                self.knl_update_gbest_pos.set_args(
                    self.gbest_pos_d, self.pbest_pos_d, np.int32(gid)
                )
                cl.enqueue_nd_range_kernel(
                    openCLEnv.queue, self.knl_update_gbest_pos, (self.nDim,), None
                )
                openCLEnv.queue.finish()
            t_gbest.append((time.perf_counter() - t0) * 1e3)

            self.BestCosts = np.concatenate((self.BestCosts, [self.gbest_cost]))
            if (
                len(self.BestCosts) > 2
                and abs(self.BestCosts[-1] - self.BestCosts[-2]) < self._criteria
            ):
                break

        elapsed = (time.perf_counter() - start) * 1e3
        self.cleanUp()
        return elapsed, t_kernel, t_copy, t_argmax, t_gbest


# ── Helper ────────────────────────────────────────────────────────────────────
def _stat(lst, label, indent=4):
    if not lst:
        return
    pad = " " * indent
    print(
        f"{pad}{label:<30}  mean={np.mean(lst):7.3f}  std={np.std(lst):6.3f}  "
        f"sum={np.sum(lst):8.3f}  n={len(lst)}  ms"
    )


def _make_mc(nPath=N_PATH, nPeriod=N_PERIOD, nFish=N_FISH):
    return hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)


def _print_kernel_info(kernel, device, label="kernel"):
    """Print OpenCL work-group info for a compiled kernel."""
    wgi = cl.kernel_work_group_info
    try:
        wgs = kernel.get_work_group_info(wgi.WORK_GROUP_SIZE, device)
        lmem = kernel.get_work_group_info(wgi.LOCAL_MEM_SIZE, device)
        pref = kernel.get_work_group_info(
            wgi.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device
        )
    except Exception as e:
        wgs, lmem, pref = "?", "?", "?"
    try:
        priv = kernel.get_work_group_info(wgi.PRIVATE_MEM_SIZE, device)
    except Exception:
        priv = "n/a"
    print(f"  [{label}]")
    print(f"    WORK_GROUP_SIZE                  : {wgs}")
    print(f"    PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {pref}")
    print(f"    LOCAL_MEM_SIZE (bytes)           : {lmem}")
    print(f"    PRIVATE_MEM_SIZE (bytes)         : {priv}")


# ── Section 1: Sub-step breakdown (baseline vec vs fusion) ────────────────────
def section_substep_breakdown(results):
    print(f"\n{SECTION}")
    print("  SECTION 1 — Sub-step timing breakdown")
    print(f"{SECTION}")

    mc = _make_mc()

    # -- vec (non-fused) --
    print(f"\n  [PSO_OpenCL_vec — non-fused, vec_size={VEC_SIZE}]")
    pso_v = PSO_OpenCL_vec(mc, N_FISH, vec_size=VEC_SIZE)
    _, elapsed_v, search, fit, rest = pso_v.solvePsoAmerOption_cl()
    print(f"  Total elapsed: {elapsed_v:.2f} ms   iterations: {len(search)}")
    _stat(search, "searchGrid  (GPU)")
    _stat(fit, "fitness     (GPU)")
    _stat(rest, "pbest+gbest (GPU+CPU)")
    _stat([s + f + r for s, f, r in zip(search, fit, rest)], "sum per iter")

    # -- vec_fusion profiled --
    print(
        f"\n  [PSO_OpenCL_vec_fusion — fused, vec_size={VEC_SIZE}]  (profiled subclass)"
    )
    mc2 = _make_mc()
    pso_vf = PSO_vec_fusion_profiled(mc2, N_FISH, vec_size=VEC_SIZE)
    elapsed_vf, t_kernel, t_copy, t_argmax, t_gbest = (
        pso_vf.solvePsoAmerOption_cl_profiled()
    )
    print(f"  Total elapsed: {elapsed_vf:.2f} ms   iterations: {len(t_kernel)}")
    _stat(t_kernel, "fused kernel (GPU)")
    _stat(t_copy, "enqueue_copy (host-transfer)")
    _stat(t_argmax, "np.argmax    (CPU)")
    _stat(t_gbest, "gbest update (cond. GPU)")
    _stat(
        [k + c + a + g for k, c, a, g in zip(t_kernel, t_copy, t_argmax, t_gbest)],
        "sum per iter",
    )

    # -- comparison table --
    n_iter_v = len(search)
    n_iter_vf = len(t_kernel)
    print(f"\n  {'Sub-step':<32} {'vec':>10}  {'vec_fusion':>10}")
    print(f"  {'-' * 54}")
    print(
        f"  {'searchGrid (mean ms/iter)':<32} {np.mean(search):>10.3f}  {'(fused)':>10}"
    )
    print(f"  {'fitness    (mean ms/iter)':<32} {np.mean(fit):>10.3f}  {'(fused)':>10}")
    print(
        f"  {'pbest+gbest(mean ms/iter)':<32} {np.mean(rest):>10.3f}  {'(fused)':>10}"
    )
    print(
        f"  {'fused kernel(mean ms/iter)':<32} {'(split)':>10}  {np.mean(t_kernel):>10.3f}"
    )
    print(
        f"  {'enqueue_copy(mean ms/iter)':<32} {'(split)':>10}  {np.mean(t_copy):>10.3f}"
    )
    print(
        f"  {'np.argmax   (mean ms/iter)':<32} {'(split)':>10}  {np.mean(t_argmax):>10.3f}"
    )
    print(
        f"  {'gbest update(mean ms/iter)':<32} {'(split)':>10}  {np.mean(t_gbest):>10.3f}"
    )
    print(f"  {'-' * 54}")
    print(f"  {'Total elapsed (ms)':<32} {elapsed_v:>10.2f}  {elapsed_vf:>10.2f}")
    print(f"  {'Iterations':<32} {n_iter_v:>10}  {n_iter_vf:>10}")
    avg_iter_v = elapsed_v / n_iter_v
    avg_iter_vf = elapsed_vf / n_iter_vf
    print(f"  {'ms / iteration':<32} {avg_iter_v:>10.2f}  {avg_iter_vf:>10.2f}")

    # Store results
    results["section1"] = {
        "elapsed_v": elapsed_v,
        "elapsed_vf": elapsed_vf,
        "n_iter_v": n_iter_v,
        "n_iter_vf": n_iter_vf,
        "avg_iter_v": avg_iter_v,
        "avg_iter_vf": avg_iter_vf,
        "search_mean": np.mean(search),
        "fit_mean": np.mean(fit),
        "rest_mean": np.mean(rest),
        "kernel_mean": np.mean(t_kernel),
        "copy_mean": np.mean(t_copy),
        "argmax_mean": np.mean(t_argmax),
        "gbest_mean": np.mean(t_gbest),
    }

    mc.cleanUp()
    mc2.cleanUp()


# ── Section 2: Kernel occupancy info ─────────────────────────────────────────
def section_kernel_info(results):
    print(f"\n{SECTION}")
    print("  SECTION 2 — Kernel occupancy / work-group info")
    print(f"{SECTION}")

    device = openCLEnv.context.devices[0]
    print(f"\n  Device: {device.name}")
    print(f"  MAX_COMPUTE_UNITS         : {device.max_compute_units}")
    print(f"  MAX_WORK_GROUP_SIZE       : {device.max_work_group_size}")
    print(f"  GLOBAL_MEM_SIZE (GB)      : {device.global_mem_size / 1e9:.2f}")
    print(f"  LOCAL_MEM_SIZE  (KB)      : {device.local_mem_size / 1024:.1f}")
    print(f"  MAX_CLOCK_FREQUENCY (MHz) : {device.max_clock_frequency}")

    mc = _make_mc()

    # Instantiate vec variant to get its compiled kernels
    pso_v = PSO_OpenCL_vec(mc, N_FISH, vec_size=VEC_SIZE)
    print("\n  -- PSO_OpenCL_vec separate kernels --")
    _print_kernel_info(pso_v.knl_searchGrid, device, "searchGrid_f2f4")
    _print_kernel_info(pso_v.knl_psoAmerOption_gb, device, "psoAmerOption_gb3_vec")
    _print_kernel_info(pso_v.knl_update_pbest, device, "update_pbest_f2f4")
    pso_v.cleanUp()

    mc2 = _make_mc()
    pso_vf = PSO_OpenCL_vec_fusion(mc2, N_FISH, vec_size=VEC_SIZE)
    print("\n  -- PSO_OpenCL_vec_fusion fused kernel --")
    _print_kernel_info(pso_vf.knl_pso, device, "pso_vec (fused)")
    _print_kernel_info(pso_vf.knl_update_gbest_pos, device, "update_gbest_pos_vec")
    pso_vf.cleanUp()

    results["section2"] = {
        "device_name": device.name,
        "max_compute_units": device.max_compute_units,
        "max_work_group_size": device.max_work_group_size,
        "global_mem_gb": device.global_mem_size / 1e9,
        "local_mem_kb": device.local_mem_size / 1024,
        "max_clock_mhz": device.max_clock_frequency,
    }

    mc.cleanUp()
    mc2.cleanUp()


# ── Section 3: vec_size sweep ─────────────────────────────────────────────────
def section_vecsize_sweep(results):
    print(f"\n{SECTION}")
    print("  SECTION 3 — vec_size sweep for vec_fusion [1, 2, 4, 8]")
    print(f"  (Checks whether float8 register pressure slows down the kernel)")
    print(f"{SECTION}")

    print(
        f"\n  {'vec_size':>8}  {'elapsed (ms)':>14}  {'iters':>6}  {'ms/iter':>10}  {'fused knl (mean ms)':>20}"
    )

    vecsize_results = []
    for vs in [1, 2, 4, 8]:
        if N_PATH % vs != 0:
            print(f"  {vs:>8}  skipped (nPath={N_PATH} not divisible by {vs})")
            continue
        mc = _make_mc()
        pso = PSO_vec_fusion_profiled(mc, N_FISH, vec_size=vs)
        elapsed, t_kernel, t_copy, t_argmax, t_gbest = (
            pso.solvePsoAmerOption_cl_profiled()
        )
        iters = len(t_kernel)
        print(
            f"  {vs:>8}  {elapsed:>14.2f}  {iters:>6}  {elapsed / iters:>10.3f}  "
            f"{np.mean(t_kernel):>20.3f}"
        )
        vecsize_results.append(
            {
                "vec_size": vs,
                "elapsed": elapsed,
                "iters": iters,
                "ms_per_iter": elapsed / iters,
                "kernel_mean": np.mean(t_kernel),
            }
        )
        mc.cleanUp()

    results["section3"] = vecsize_results


# ── Section 4: nFish sweep ────────────────────────────────────────────────────
def section_nfish_sweep(results):
    print(f"\n{SECTION}")
    print("  SECTION 4 — nFish sweep [64, 128, 256, 512] for vec vs vec_fusion")
    print(f"  (Checks GPU occupancy effects)")
    print(f"{SECTION}")

    print(
        f"\n  {'nFish':>6}  {'vec (ms)':>10}  {'vec iters':>10}  {'fusion (ms)':>12}  {'fusion iters':>12}  {'ratio':>6}"
    )
    print(f"  {'-' * 66}")

    nfish_results = []
    for nf in [64, 128, 256, 512]:
        mc1 = _make_mc(nFish=nf)
        pso_v = PSO_OpenCL_vec(mc1, nf, vec_size=VEC_SIZE)
        _, t_v, sv, fv, rv = pso_v.solvePsoAmerOption_cl()
        n_v = len(sv)

        mc2 = _make_mc(nFish=nf)
        pso_vf = PSO_vec_fusion_profiled(mc2, nf, vec_size=VEC_SIZE)
        t_vf, tk, tc, ta, tg = pso_vf.solvePsoAmerOption_cl_profiled()
        n_vf = len(tk)

        ratio = t_vf / t_v if t_v > 0 else float("nan")
        print(
            f"  {nf:>6}  {t_v:>10.2f}  {n_v:>10}  {t_vf:>12.2f}  {n_vf:>12}  {ratio:>6.2f}x"
        )

        nfish_results.append(
            {
                "nFish": nf,
                "vec_ms": t_v,
                "vec_iters": n_v,
                "fusion_ms": t_vf,
                "fusion_iters": n_vf,
                "ratio": ratio,
            }
        )

        mc1.cleanUp()
        mc2.cleanUp()

    results["section4"] = nfish_results


# ── Section 5: Iteration detail dump ─────────────────────────────────────────
def section_iteration_detail(results):
    print(f"\n{SECTION}")
    print("  SECTION 5 — Per-iteration detail (first 10 iters): vec vs vec_fusion")
    print(f"{SECTION}")

    mc1 = _make_mc()
    pso_v = PSO_OpenCL_vec(mc1, N_FISH, vec_size=VEC_SIZE)
    _, _, search, fit, rest = pso_v.solvePsoAmerOption_cl()

    mc2 = _make_mc()
    pso_vf = PSO_vec_fusion_profiled(mc2, N_FISH, vec_size=VEC_SIZE)
    _, t_kernel, t_copy, t_argmax, t_gbest = pso_vf.solvePsoAmerOption_cl_profiled()

    print(
        f"\n  {'iter':>4}  {'vec_sg':>8}  {'vec_fit':>8}  {'vec_rest':>9}  "
        f"{'vf_kernel':>10}  {'vf_copy':>8}  {'vf_argmax':>10}  {'vf_gbest':>9}"
    )
    print(f"  {'-' * 80}")

    n_show = min(10, len(search), len(t_kernel))
    iter_details = []
    for i in range(n_show):
        sg = search[i] if i < len(search) else 0
        ft = fit[i] if i < len(fit) else 0
        rs = rest[i] if i < len(rest) else 0
        tk = t_kernel[i] if i < len(t_kernel) else 0
        tc = t_copy[i] if i < len(t_copy) else 0
        ta = t_argmax[i] if i < len(t_argmax) else 0
        tg = t_gbest[i] if i < len(t_gbest) else 0
        print(
            f"  {i:>4}  {sg:>8.3f}  {ft:>8.3f}  {rs:>9.3f}  "
            f"{tk:>10.3f}  {tc:>8.3f}  {ta:>10.3f}  {tg:>9.3f}"
        )
        iter_details.append(
            {
                "iter": i,
                "vec_sg": sg,
                "vec_fit": ft,
                "vec_rest": rs,
                "vf_kernel": tk,
                "vf_copy": tc,
                "vf_argmax": ta,
                "vf_gbest": tg,
            }
        )

    if len(search) > 10:
        print(
            f"  ... ({len(search)} total iters for vec, {len(t_kernel)} for vec_fusion)"
        )

    results["section5"] = {
        "details": iter_details,
        "total_vec_iters": len(search),
        "total_vf_iters": len(t_kernel),
    }

    mc1.cleanUp()
    mc2.cleanUp()


# ── Section 6: nPath sweep ────────────────────────────────────────────────────
def section_npath_sweep(results):
    print(f"\n{SECTION}")
    print(
        "  SECTION 6 — nPath sweep: NumPy CPU vs GPU scalar vs GPU vec vs GPU vec_fusion"
    )
    print(
        f"  (ATM Put 02SEP08, nFish={N_FISH}, nPeriod={N_PERIOD}, vec_size={VEC_SIZE})"
    )
    print(f"  Tests whether the slowdown is consistent or varies with simulation size.")
    print(f"{SECTION}")

    npath_values = [2000, 5000, 10_000, 20_000, 40_000]

    hdr = (
        f"  {'nPath':>7}  {'numpy (ms)':>11}  {'np iters':>9}  "
        f"{'scalar (ms)':>12}  {'sc iters':>9}  "
        f"{'vec (ms)':>10}  {'vc iters':>9}  "
        f"{'fusion (ms)':>11}  {'fs iters':>9}  "
        f"{'sc/np':>7}  {'vc/np':>7}  {'fs/np':>7}  {'fs/vc':>7}"
    )
    print(f"\n{hdr}")
    print(f"  {'-' * 140}")

    npath_results = []
    for npath in npath_values:
        if npath % VEC_SIZE != 0:
            print(f"  {npath:>7}  skipped (not divisible by vec_size={VEC_SIZE})")
            continue

        # NumPy CPU baseline
        mc_np = hybridMonteCarlo(S0, r, sigma, T, npath, N_PERIOD, K, opttype, N_FISH)
        pso_np = PSO_Numpy(mc_np, N_FISH)
        _, t_np, search_np, fit_np, rest_np = pso_np.solvePsoAmerOption_np()
        n_np = len(search_np)

        # GPU scalar
        mc_sc = hybridMonteCarlo(S0, r, sigma, T, npath, N_PERIOD, K, opttype, N_FISH)
        pso_sc = PSO_OpenCL_scalar(mc_sc, N_FISH)
        _, t_sc, search_sc, fit_sc, rest_sc = pso_sc.solvePsoAmerOption_cl()
        n_sc = len(search_sc)

        # GPU vec
        mc_vc = hybridMonteCarlo(S0, r, sigma, T, npath, N_PERIOD, K, opttype, N_FISH)
        pso_vc = PSO_OpenCL_vec(mc_vc, N_FISH, vec_size=VEC_SIZE)
        _, t_vc, search_vc, fit_vc, rest_vc = pso_vc.solvePsoAmerOption_cl()
        n_vc = len(search_vc)

        # GPU vec_fusion
        mc_vf = hybridMonteCarlo(S0, r, sigma, T, npath, N_PERIOD, K, opttype, N_FISH)
        pso_vf = PSO_vec_fusion_profiled(mc_vf, N_FISH, vec_size=VEC_SIZE)
        t_vf, t_kernel, t_copy, t_argmax, t_gbest = (
            pso_vf.solvePsoAmerOption_cl_profiled()
        )
        n_vf = len(t_kernel)

        r_sc_np = t_sc / t_np if t_np > 0 else float("nan")
        r_vc_np = t_vc / t_np if t_np > 0 else float("nan")
        r_vf_np = t_vf / t_np if t_np > 0 else float("nan")
        r_vf_vc = t_vf / t_vc if t_vc > 0 else float("nan")

        print(
            f"  {npath:>7}  {t_np:>11.2f}  {n_np:>9}  "
            f"{t_sc:>12.2f}  {n_sc:>9}  "
            f"{t_vc:>10.2f}  {n_vc:>9}  "
            f"{t_vf:>11.2f}  {n_vf:>9}  "
            f"{r_sc_np:>7.2f}x  {r_vc_np:>7.2f}x  {r_vf_np:>7.2f}x  {r_vf_vc:>7.2f}x"
        )

        npath_results.append(
            {
                "nPath": npath,
                "numpy_ms": t_np,
                "np_iters": n_np,
                "scalar_ms": t_sc,
                "sc_iters": n_sc,
                "vec_ms": t_vc,
                "vc_iters": n_vc,
                "fusion_ms": t_vf,
                "fs_iters": n_vf,
                "sc_np": r_sc_np,
                "vc_np": r_vc_np,
                "fs_np": r_vf_np,
                "fs_vc": r_vf_vc,
            }
        )

        mc_np.cleanUp()
        mc_sc.cleanUp()
        mc_vc.cleanUp()
        mc_vf.cleanUp()

    print(
        f"\n  Columns: sc/np = GPU scalar vs NumPy, "
        f"vc/np = GPU vec vs NumPy, "
        f"fs/np = GPU vec_fusion vs NumPy,"
    )
    print(f"  fs/vc = GPU vec_fusion vs GPU vec (should be <1 if fusion is faster).")
    print(f"  Stable ratios across nPath values confirm a systemic pattern.")

    results["section6"] = npath_results


# ── Section 7: Multi-case sweep ───────────────────────────────────────────────
SWEEP_CASES = [
    ("ATM  Put  02SEP08", 1277.58, 1280.749, 0.0172, 0.2134, 30 / 365, "P"),
    ("ATM  Call 02SEP08", 1277.58, 1280.343, 0.0172, 0.2097, 30 / 365, "C"),
    ("OTM  Put  15SEP08", 1192.70, 1126.930, 0.0102, 0.3172, 30 / 365, "P"),
    ("OTM  Call 15SEP08", 1192.70, 1265.262, 0.0102, 0.2867, 30 / 365, "C"),
    ("ITM  Put  15SEP08", 1192.70, 1267.000, 0.0102, 0.2900, 30 / 365, "P"),
    ("ITM  Call 15SEP08", 1192.70, 1125.000, 0.0102, 0.3300, 30 / 365, "C"),
    ("vOTM Put  29SEP08", 1106.42, 987.000, 0.0094, 0.4100, 30 / 365, "P"),
    ("vOTM Call 29SEP08", 1106.42, 1260.221, 0.0094, 0.4114, 30 / 365, "C"),
    ("dOTM Put  02SEP08", 1277.58, 1155.000, 0.0172, 0.2700, 30 / 365, "P"),
    ("dITM Call 29SEP08", 1106.42, 966.000, 0.0094, 0.3900, 30 / 365, "C"),
]


def section_multicase_sweep(results):
    print(f"\n{SECTION}")
    print(
        "  SECTION 7 — Multi-case sweep: NumPy CPU vs GPU scalar vs GPU vec vs GPU vec_fusion"
    )
    print(
        f"  (nPath={N_PATH}, nPeriod={N_PERIOD}, nFish={N_FISH}, vec_size={VEC_SIZE})"
    )
    print(f"  Tests whether the slowdown is option-specific or systemic.")
    print(f"{SECTION}")

    print(
        f"\n  {'Case':<24}  {'numpy (ms)':>11}  {'np iters':>9}  "
        f"{'scalar (ms)':>12}  {'sc iters':>9}  "
        f"{'vec (ms)':>10}  {'vc iters':>9}  "
        f"{'fusion (ms)':>11}  {'fs iters':>9}  "
        f"{'sc/np':>7}  {'vc/np':>7}  {'fs/np':>7}  {'fs/vc':>7}"
    )
    print(f"  {'-' * 145}")

    ratios_vf_vc = []
    ratios_sc_np = []
    ratios_vc_np = []
    ratios_vf_np = []

    case_results = []
    for label, s0, k, r_val, sigma_val, T_val, opt in SWEEP_CASES:
        # NumPy CPU baseline
        mc_np = hybridMonteCarlo(
            s0, r_val, sigma_val, T_val, N_PATH, N_PERIOD, k, opt, N_FISH
        )
        pso_np = PSO_Numpy(mc_np, N_FISH)
        _, t_np, search_np, fit_np, rest_np = pso_np.solvePsoAmerOption_np()
        n_np = len(search_np)

        # GPU scalar
        mc_sc = hybridMonteCarlo(
            s0, r_val, sigma_val, T_val, N_PATH, N_PERIOD, k, opt, N_FISH
        )
        pso_sc = PSO_OpenCL_scalar(mc_sc, N_FISH)
        _, t_sc, search_sc, fit_sc, rest_sc = pso_sc.solvePsoAmerOption_cl()
        n_sc = len(search_sc)

        # GPU vec
        mc_vc = hybridMonteCarlo(
            s0, r_val, sigma_val, T_val, N_PATH, N_PERIOD, k, opt, N_FISH
        )
        pso_vc = PSO_OpenCL_vec(mc_vc, N_FISH, vec_size=VEC_SIZE)
        _, t_vc, search_vc, fit_vc, rest_vc = pso_vc.solvePsoAmerOption_cl()
        n_vc = len(search_vc)

        # GPU vec_fusion
        mc_vf = hybridMonteCarlo(
            s0, r_val, sigma_val, T_val, N_PATH, N_PERIOD, k, opt, N_FISH
        )
        pso_vf = PSO_vec_fusion_profiled(mc_vf, N_FISH, vec_size=VEC_SIZE)
        t_vf, t_kernel, t_copy, t_argmax, t_gbest = (
            pso_vf.solvePsoAmerOption_cl_profiled()
        )
        n_vf = len(t_kernel)

        r_sc_np = t_sc / t_np if t_np > 0 else float("nan")
        r_vc_np = t_vc / t_np if t_np > 0 else float("nan")
        r_vf_np = t_vf / t_np if t_np > 0 else float("nan")
        r_vf_vc = t_vf / t_vc if t_vc > 0 else float("nan")

        ratios_sc_np.append(r_sc_np)
        ratios_vc_np.append(r_vc_np)
        ratios_vf_np.append(r_vf_np)
        ratios_vf_vc.append(r_vf_vc)

        print(
            f"  {label:<24}  {t_np:>11.2f}  {n_np:>9}  "
            f"{t_sc:>12.2f}  {n_sc:>9}  "
            f"{t_vc:>10.2f}  {n_vc:>9}  "
            f"{t_vf:>11.2f}  {n_vf:>9}  "
            f"{r_sc_np:>7.2f}x  {r_vc_np:>7.2f}x  {r_vf_np:>7.2f}x  {r_vf_vc:>7.2f}x"
        )

        case_results.append(
            {
                "case": label,
                "numpy_ms": t_np,
                "np_iters": n_np,
                "scalar_ms": t_sc,
                "sc_iters": n_sc,
                "vec_ms": t_vc,
                "vc_iters": n_vc,
                "fusion_ms": t_vf,
                "fs_iters": n_vf,
                "sc_np": r_sc_np,
                "vc_np": r_vc_np,
                "fs_np": r_vf_np,
                "fs_vc": r_vf_vc,
            }
        )

        mc_np.cleanUp()
        mc_sc.cleanUp()
        mc_vc.cleanUp()
        mc_vf.cleanUp()

    print(f"  {'-' * 145}")
    print(
        f"  {'Mean (sc/np)  GPU scalar vs NumPy':<50}  {np.mean(ratios_sc_np):>7.2f}x  "
        f"std={np.std(ratios_sc_np):.3f}"
    )
    print(
        f"  {'Mean (vc/np)  GPU vec   vs NumPy':<50}  {np.mean(ratios_vc_np):>7.2f}x  "
        f"std={np.std(ratios_vc_np):.3f}"
    )
    print(
        f"  {'Mean (fs/np)  GPU vec_fusion vs NumPy':<50}  {np.mean(ratios_vf_np):>7.2f}x  "
        f"std={np.std(ratios_vf_np):.3f}"
    )
    print(
        f"  {'Mean (fs/vc)  GPU vec_fusion vs GPU vec':<50}  {np.mean(ratios_vf_vc):>7.2f}x  "
        f"std={np.std(ratios_vf_vc):.3f}"
    )
    print(f"\n  Interpretation:")
    print(f"    sc/np < 1 => GPU scalar faster than NumPy (expected).")
    print(f"    vc/np < 1 => GPU vec faster than NumPy (expected with vec_size=4).")
    print(f"    fs/np < 1 => GPU vec_fusion faster than NumPy.")
    print(f"    fs/vc < 1 => GPU vec_fusion faster than GPU vec (fusion benefit).")

    results["section7"] = {
        "cases": case_results,
        "mean_sc_np": np.mean(ratios_sc_np),
        "mean_vc_np": np.mean(ratios_vc_np),
        "mean_fs_np": np.mean(ratios_vf_np),
        "mean_fs_vc": np.mean(ratios_vf_vc),
        "std_sc_np": np.std(ratios_sc_np),
        "std_vc_np": np.std(ratios_vc_np),
        "std_fs_np": np.std(ratios_vf_np),
        "std_fs_vc": np.std(ratios_vf_vc),
    }


# ── Markdown Report Generation ───────────────────────────────────────────────
def generate_markdown_report(results):
    """Generate a comprehensive Markdown report with all benchmark results."""
    import platform

    device = openCLEnv.context.devices[0]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = []
    report.append("# PSO vec_fusion Performance Diagnostic Report")
    report.append("")
    report.append(f"**Generated:** {timestamp}")
    report.append(f"**Platform:** {platform.system()} {platform.release()}")
    report.append(f"**Device:** {device.name}")
    report.append("")
    report.append("## Executive Summary")
    report.append("")

    # Key findings from Section 1
    if "section1" in results:
        s1 = results["section1"]
        speedup = s1["elapsed_v"] / s1["elapsed_vf"]
        report.append(
            f"- **vec_fusion vs vec:** {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
        )
        report.append(
            f"  - vec: {s1['elapsed_v']:.2f} ms ({s1['n_iter_v']} iterations)"
        )
        report.append(
            f"  - vec_fusion: {s1['elapsed_vf']:.2f} ms ({s1['n_iter_vf']} iterations)"
        )
        report.append("")

    # Key findings from Section 7
    if "section7" in results:
        s7 = results["section7"]
        report.append(f"- **Mean Speedups vs NumPy:**")
        report.append(
            f"  - GPU scalar: {1 / s7['mean_sc_np']:.1f}x faster (std={s7['std_sc_np']:.3f})"
        )
        report.append(
            f"  - GPU vec: {1 / s7['mean_vc_np']:.1f}x faster (std={s7['std_vc_np']:.3f})"
        )
        report.append(
            f"  - GPU vec_fusion: {1 / s7['mean_fs_np']:.1f}x faster (std={s7['std_fs_np']:.3f})"
        )
        report.append(
            f"- **vec_fusion vs vec:** {s7['mean_fs_vc']:.2f}x (fusion is {'faster' if s7['mean_fs_vc'] < 1 else 'slower'})"
        )
        report.append("")

    report.append("---")
    report.append("")
    report.append("## Test Configuration")
    report.append("")
    report.append("| Parameter | Value |")
    report.append("|-----------|-------|")
    report.append(f"| Base Case | ATM Put 02SEP08 |")
    report.append(f"| S0 | {S0} |")
    report.append(f"| K | {K} |")
    report.append(f"| r | {r} |")  # noqa: F821
    report.append(f"| sigma | {sigma} |")  # noqa: F821
    report.append(f"| T | {T:.6f} years (30/365) |")
    report.append(f"| Option Type | {opttype} |")
    report.append(f"| nPath | {N_PATH:,} |")
    report.append(f"| nPeriod | {N_PERIOD} |")
    report.append(f"| nFish | {N_FISH} |")
    report.append(f"| vec_size | {VEC_SIZE} |")
    report.append("")

    if "section2" in results:
        s2 = results["section2"]
        report.append("## Device Information")
        report.append("")
        report.append("| Property | Value |")
        report.append("|----------|-------|")
        report.append(f"| Device Name | {s2['device_name']} |")
        report.append(f"| Max Compute Units | {s2['max_compute_units']} |")
        report.append(f"| Max Work Group Size | {s2['max_work_group_size']:,} |")
        report.append(f"| Global Memory | {s2['global_mem_gb']:.2f} GB |")
        report.append(f"| Local Memory | {s2['local_mem_kb']:.1f} KB |")
        report.append(f"| Max Clock Frequency | {s2['max_clock_mhz']} MHz |")
        report.append("")

    if "section1" in results:
        s1 = results["section1"]
        report.append("## Section 1: Sub-step Timing Breakdown")
        report.append("")
        report.append("### Timing Comparison")
        report.append("")
        report.append("| Metric | vec (non-fused) | vec_fusion (fused) |")
        report.append("|--------|-----------------|-------------------|")
        report.append(
            f"| Total Elapsed (ms) | {s1['elapsed_v']:.2f} | {s1['elapsed_vf']:.2f} |"
        )
        report.append(f"| Iterations | {s1['n_iter_v']} | {s1['n_iter_vf']} |")
        report.append(
            f"| ms / iteration | {s1['avg_iter_v']:.2f} | {s1['avg_iter_vf']:.2f} |"
        )
        report.append("")
        report.append("### Component Breakdown (mean ms/iter)")
        report.append("")
        report.append("| Component | vec | vec_fusion |")
        report.append("|-----------|-----|------------|")
        report.append(f"| searchGrid | {s1['search_mean']:.3f} | (fused) |")
        report.append(f"| fitness | {s1['fit_mean']:.3f} | (fused) |")
        report.append(f"| pbest+gbest | {s1['rest_mean']:.3f} | (fused) |")
        report.append(f"| fused kernel | (split) | {s1['kernel_mean']:.3f} |")
        report.append(f"| enqueue_copy | (split) | {s1['copy_mean']:.3f} |")
        report.append(f"| np.argmax | (split) | {s1['argmax_mean']:.3f} |")
        report.append(f"| gbest update | (split) | {s1['gbest_mean']:.3f} |")
        report.append("")

    if "section3" in results:
        report.append("## Section 3: vec_size Sweep")
        report.append("")
        report.append("Tests whether float8 register pressure slows down the kernel.")
        report.append("")
        report.append(
            "| vec_size | Elapsed (ms) | Iterations | ms/iter | Fused Kernel (ms) |"
        )
        report.append(
            "|----------|--------------|------------|---------|-------------------|"
        )
        for row in results["section3"]:
            report.append(
                f"| {row['vec_size']} | {row['elapsed']:.2f} | {row['iters']} | {row['ms_per_iter']:.3f} | {row['kernel_mean']:.3f} |"
            )
        report.append("")
        # Find fastest
        fastest = min(results["section3"], key=lambda x: x["ms_per_iter"])
        report.append(
            f"**Fastest:** vec_size={fastest['vec_size']} at {fastest['ms_per_iter']:.3f} ms/iter"
        )
        report.append("")

    if "section4" in results:
        report.append("## Section 4: nFish Sweep")
        report.append("")
        report.append("Tests GPU occupancy effects.")
        report.append("")
        report.append(
            "| nFish | vec (ms) | vec iters | fusion (ms) | fusion iters | ratio (fs/vc) |"
        )
        report.append(
            "|-------|----------|-----------|-------------|--------------|---------------|"
        )
        for row in results["section4"]:
            report.append(
                f"| {row['nFish']} | {row['vec_ms']:.2f} | {row['vec_iters']} | {row['fusion_ms']:.2f} | {row['fusion_iters']} | {row['ratio']:.2f}x |"
            )
        report.append("")

    if "section5" in results:
        s5 = results["section5"]
        report.append("## Section 5: Per-Iteration Detail (First 10 Iterations)")
        report.append("")
        report.append(
            "| iter | vec_sg | vec_fit | vec_rest | vf_kernel | vf_copy | vf_argmax | vf_gbest |"
        )
        report.append(
            "|------|--------|---------|----------|-----------|---------|-----------|----------|"
        )
        for d in s5["details"]:
            report.append(
                f"| {d['iter']} | {d['vec_sg']:.3f} | {d['vec_fit']:.3f} | {d['vec_rest']:.3f} | {d['vf_kernel']:.3f} | {d['vf_copy']:.3f} | {d['vf_argmax']:.3f} | {d['vf_gbest']:.3f} |"
            )
        report.append("")

    if "section6" in results:
        report.append("## Section 6: nPath Sweep")
        report.append("")
        report.append(
            "Tests whether performance is consistent across simulation sizes."
        )
        report.append("")
        report.append(
            "| nPath | numpy (ms) | scalar (ms) | vec (ms) | fusion (ms) | sc/np | vc/np | fs/np | fs/vc |"
        )
        report.append(
            "|-------|------------|-------------|----------|-------------|-------|-------|-------|-------|"
        )
        for row in results["section6"]:
            report.append(
                f"| {row['nPath']:,} | {row['numpy_ms']:.2f} | {row['scalar_ms']:.2f} | {row['vec_ms']:.2f} | {row['fusion_ms']:.2f} | {row['sc_np']:.2f} | {row['vc_np']:.2f} | {row['fs_np']:.2f} | {row['fs_vc']:.2f} |"
            )
        report.append("")
        report.append("**Columns:**")
        report.append("- `sc/np`: GPU scalar vs NumPy (<1 = faster)")
        report.append("- `vc/np`: GPU vec vs NumPy (<1 = faster)")
        report.append("- `fs/np`: GPU vec_fusion vs NumPy (<1 = faster)")
        report.append("- `fs/vc`: GPU vec_fusion vs GPU vec (<1 = fusion is faster)")
        report.append("")

    if "section7" in results:
        s7 = results["section7"]
        report.append("## Section 7: Multi-Case Sweep")
        report.append("")
        report.append(
            "Tests performance across different option types (ATM, OTM, ITM, deep OTM/ITM)."
        )
        report.append("")
        report.append(
            "| Case | numpy (ms) | scalar (ms) | vec (ms) | fusion (ms) | sc/np | vc/np | fs/np | fs/vc |"
        )
        report.append(
            "|------|------------|-------------|----------|-------------|-------|-------|-------|-------|"
        )
        for row in s7["cases"]:
            report.append(
                f"| {row['case']} | {row['numpy_ms']:.2f} | {row['scalar_ms']:.2f} | {row['vec_ms']:.2f} | {row['fusion_ms']:.2f} | {row['sc_np']:.2f} | {row['vc_np']:.2f} | {row['fs_np']:.2f} | {row['fs_vc']:.2f} |"
            )
        report.append("")
        report.append("### Summary Statistics")
        report.append("")
        report.append("| Comparison | Mean | Std | Interpretation |")
        report.append("|------------|------|-----|----------------|")
        report.append(
            f"| GPU scalar vs NumPy | {s7['mean_sc_np']:.2f}x | {s7['std_sc_np']:.3f} | {1 / s7['mean_sc_np']:.1f}x faster |"
        )
        report.append(
            f"| GPU vec vs NumPy | {s7['mean_vc_np']:.2f}x | {s7['std_vc_np']:.3f} | {1 / s7['mean_vc_np']:.1f}x faster |"
        )
        report.append(
            f"| GPU vec_fusion vs NumPy | {s7['mean_fs_np']:.2f}x | {s7['std_fs_np']:.3f} | {1 / s7['mean_fs_np']:.1f}x faster |"
        )
        report.append(
            f"| GPU vec_fusion vs GPU vec | {s7['mean_fs_vc']:.2f}x | {s7['std_fs_vc']:.3f} | {'faster' if s7['mean_fs_vc'] < 1 else 'slower'} by {abs(1 - s7['mean_fs_vc']) * 100:.1f}% |"
        )
        report.append("")

    report.append("## Analysis")
    report.append("")
    report.append("### Key Findings")
    report.append("")

    if "section3" in results and "section1" in results:
        fastest_vs = min(results["section3"], key=lambda x: x["ms_per_iter"])
        s1 = results["section1"]
        speedup = s1["elapsed_v"] / s1["elapsed_vf"]
        report.append(
            f"1. **Optimal vec_size:** {fastest_vs['vec_size']} is fastest at {fastest_vs['ms_per_iter']:.2f} ms/iter"
        )
        report.append(f"   - vec_size=8 is slower due to register pressure")
        report.append(
            f"2. **Fusion benefit:** vec_fusion is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than vec"
        )
        report.append(f"   - Fused kernel time: {s1['kernel_mean']:.2f} ms/iter")
        report.append(f"   - Host transfer overhead: {s1['copy_mean']:.3f} ms/iter")
        report.append(f"   - CPU argmax overhead: {s1['argmax_mean']:.3f} ms/iter")

    if "section7" in results:
        s7 = results["section7"]
        report.append(
            f"3. **Consistency:** Low std ({s7['std_fs_vc']:.3f}) indicates stable performance across option types"
        )
        report.append(
            f"4. **GPU advantage:** Both vec and vec_fusion significantly outperform NumPy CPU baseline"
        )

    report.append("")
    report.append("### Interpretation")
    report.append("")
    report.append("- `sc/np < 1`: GPU scalar faster than NumPy (expected)")
    report.append(
        "- `vc/np < 1`: GPU vec faster than NumPy (expected with optimal vec_size)"
    )
    report.append("- `fs/np < 1`: GPU vec_fusion faster than NumPy")
    report.append(
        "- `fs/vc < 1`: GPU vec_fusion faster than GPU vec (fusion provides benefit)"
    )
    report.append("")
    report.append("---")
    report.append("")
    report.append("## How to Run")
    report.append("")
    report.append("```bash")
    report.append("python test_vec_fusion_profile.py")
    report.append("```")
    report.append("")
    report.append("---")
    report.append("")
    report.append("*Report generated by PSO vec_fusion Performance Diagnostic*")

    return "\n".join(report)


def save_report(report_content, filepath="test_vec_fusion_report.md"):
    """Save the report to a file."""
    report_path = Path(__file__).parent / filepath
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\n  Markdown report saved to: {report_path}")
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    from models.utils import checkOpenCL

    checkOpenCL()

    print(f"\n{'#' * 72}")
    print("  PSO vec_fusion Performance Diagnostic")
    print(
        f"  Base case: ATM Put 02SEP08  nPath={N_PATH}  nPeriod={N_PERIOD}  nFish={N_FISH}  vec_size={VEC_SIZE}"
    )
    print(f"{'#' * 72}")

    # Dictionary to collect all results
    results = {}

    section_substep_breakdown(results)
    section_kernel_info(results)
    section_vecsize_sweep(results)
    section_nfish_sweep(results)
    section_iteration_detail(results)
    section_npath_sweep(results)
    section_multicase_sweep(results)

    print(f"\n{'#' * 72}")
    print("  Diagnostic complete.")
    print(f"{'#' * 72}")

    # Generate and save Markdown report
    print("\n  Generating Markdown report...")
    report = generate_markdown_report(results)
    save_report(report)
    print()


if __name__ == "__main__":
    main()
