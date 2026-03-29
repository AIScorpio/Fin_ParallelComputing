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
  6. nPath sweep [2000, 5000, 10000, 20000, 40000] — NumPy CPU vs scalar vs vec ratio
  7. Multi-case sweep — NumPy CPU vs scalar vs vec across 10 representative option cases
"""

import sys
import time
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pyopencl as cl
from src.models.mc import hybridMonteCarlo
from src.models.pso import PSO_Numpy, PSO_OpenCL_scalar, PSO_OpenCL_vec, PSO_OpenCL_vec_fusion
from src.models.utils import openCLEnv

# ── Test parameters (ATM Put 02SEP08, same as first case in test_pso.py) ─────
S0, K, r, sigma, T, opttype = 1277.58, 1280.749, 0.0172, 0.213411, 30/365, "P"
N_PATH   = 10_000
N_PERIOD = 150
N_FISH   = 256
VEC_SIZE = 8

SECTION = "=" * 72


# ── Profiled subclass of PSO_OpenCL_vec_fusion ────────────────────────────────
class PSO_vec_fusion_profiled(PSO_OpenCL_vec_fusion):
    """
    Overrides solvePsoAmerOption_cl() to add fine-grained per-step timing
    inside each PSO iteration, without modifying the original class.
    """

    def solvePsoAmerOption_cl_profiled(self):
        """Same logic as parent but with split timings per sub-step."""
        t_kernel  = []   # _runPso() fused kernel time
        t_copy    = []   # enqueue_copy(pbest_costs) host-transfer time
        t_argmax  = []   # np.argmax on host
        t_gbest   = []   # conditional gbest update kernel time

        start = time.perf_counter()

        for i in range(self.iterMax):
            # -- fused kernel (searchGrid + fitness + pbest update on GPU) ----
            t0 = time.perf_counter()
            self._runPso()
            t_kernel.append((time.perf_counter() - t0) * 1e3)

            # -- copy pbest_costs from device to host -------------------------
            t0 = time.perf_counter()
            cl.enqueue_copy(openCLEnv.queue, self.pbest_costs, self.pbest_costs_d).wait()
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
                    openCLEnv.queue, self.knl_update_gbest_pos,
                    (self.nDim,), None
                )
                openCLEnv.queue.finish()
            t_gbest.append((time.perf_counter() - t0) * 1e3)

            self.BestCosts = np.concatenate((self.BestCosts, [self.gbest_cost]))
            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1] - self.BestCosts[-2]) < self._criteria:
                break

        elapsed = (time.perf_counter() - start) * 1e3
        self.cleanUp()
        return elapsed, t_kernel, t_copy, t_argmax, t_gbest


# ── Helper ────────────────────────────────────────────────────────────────────
def _stat(lst, label, indent=4):
    if not lst:
        return
        pad = " " * indent
    pad = " " * indent
    print(f"{pad}{label:<30}  mean={np.mean(lst):7.3f}  std={np.std(lst):6.3f}  "
          f"sum={np.sum(lst):8.3f}  n={len(lst)}  ms")


def _make_mc(nPath=N_PATH, nPeriod=N_PERIOD, nFish=N_FISH):
    return hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)


def _print_kernel_info(kernel, device, label="kernel"):
    """Print OpenCL work-group info for a compiled kernel."""
    wgi = cl.kernel_work_group_info
    try:
        wgs  = kernel.get_work_group_info(wgi.WORK_GROUP_SIZE, device)
        lmem = kernel.get_work_group_info(wgi.LOCAL_MEM_SIZE,  device)
        pref = kernel.get_work_group_info(wgi.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device)
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
def section_substep_breakdown():
    print(f"\n{SECTION}")
    print("  SECTION 1 — Sub-step timing breakdown")
    print(f"{SECTION}")

    mc = _make_mc()

    # -- vec (non-fused) --
    print("\n  [PSO_OpenCL_vec — non-fused, vec_size=8]")
    pso_v = PSO_OpenCL_vec(mc, N_FISH, vec_size=VEC_SIZE)
    _, elapsed_v, search, fit, rest = pso_v.solvePsoAmerOption_cl()
    print(f"  Total elapsed: {elapsed_v:.2f} ms   iterations: {len(search)}")
    _stat(search, "searchGrid  (GPU)")
    _stat(fit,    "fitness     (GPU)")
    _stat(rest,   "pbest+gbest (GPU+CPU)")
    _stat([s+f+r for s,f,r in zip(search,fit,rest)], "sum per iter")

    # -- vec_fusion profiled --
    print("\n  [PSO_OpenCL_vec_fusion — fused, vec_size=8]  (profiled subclass)")
    mc2 = _make_mc()   # fresh mc (same seed via hybridMonteCarlo re-init)
    pso_vf = PSO_vec_fusion_profiled(mc2, N_FISH, vec_size=VEC_SIZE)
    elapsed_vf, t_kernel, t_copy, t_argmax, t_gbest = pso_vf.solvePsoAmerOption_cl_profiled()
    print(f"  Total elapsed: {elapsed_vf:.2f} ms   iterations: {len(t_kernel)}")
    _stat(t_kernel, "fused kernel (GPU)")
    _stat(t_copy,   "enqueue_copy (host-transfer)")
    _stat(t_argmax, "np.argmax    (CPU)")
    _stat(t_gbest,  "gbest update (cond. GPU)")
    _stat([k+c+a+g for k,c,a,g in zip(t_kernel,t_copy,t_argmax,t_gbest)], "sum per iter")

    # -- comparison table --
    n_iter_v  = len(search)
    n_iter_vf = len(t_kernel)
    print(f"\n  {'Sub-step':<32} {'vec':>10}  {'vec_fusion':>10}")
    print(f"  {'-'*54}")
    print(f"  {'searchGrid (mean ms/iter)':<32} {np.mean(search):>10.3f}  {'(fused)':>10}")
    print(f"  {'fitness    (mean ms/iter)':<32} {np.mean(fit):>10.3f}  {'(fused)':>10}")
    print(f"  {'pbest+gbest(mean ms/iter)':<32} {np.mean(rest):>10.3f}  {'(fused)':>10}")
    print(f"  {'fused kernel(mean ms/iter)':<32} {'(split)':>10}  {np.mean(t_kernel):>10.3f}")
    print(f"  {'enqueue_copy(mean ms/iter)':<32} {'(split)':>10}  {np.mean(t_copy):>10.3f}")
    print(f"  {'np.argmax   (mean ms/iter)':<32} {'(split)':>10}  {np.mean(t_argmax):>10.3f}")
    print(f"  {'gbest update(mean ms/iter)':<32} {'(split)':>10}  {np.mean(t_gbest):>10.3f}")
    print(f"  {'-'*54}")
    print(f"  {'Total elapsed (ms)':<32} {elapsed_v:>10.2f}  {elapsed_vf:>10.2f}")
    print(f"  {'Iterations':<32} {n_iter_v:>10}  {n_iter_vf:>10}")
    avg_iter_v  = elapsed_v / n_iter_v
    avg_iter_vf = elapsed_vf / n_iter_vf
    print(f"  {'ms / iteration':<32} {avg_iter_v:>10.2f}  {avg_iter_vf:>10.2f}")

    mc.cleanUp()
    mc2.cleanUp()


# ── Section 2: Kernel occupancy info ─────────────────────────────────────────
def section_kernel_info():
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
    _print_kernel_info(pso_v.knl_searchGrid,        device, "searchGrid_f2f4")
    _print_kernel_info(pso_v.knl_psoAmerOption_gb,  device, "psoAmerOption_gb3_vec")
    _print_kernel_info(pso_v.knl_update_pbest,      device, "update_pbest_f2f4")
    pso_v.cleanUp()

    mc2 = _make_mc()
    pso_vf = PSO_OpenCL_vec_fusion(mc2, N_FISH, vec_size=VEC_SIZE)
    print("\n  -- PSO_OpenCL_vec_fusion fused kernel --")
    _print_kernel_info(pso_vf.knl_pso, device, "pso_vec (fused)")
    _print_kernel_info(pso_vf.knl_update_gbest_pos, device, "update_gbest_pos_vec")
    pso_vf.cleanUp()

    mc.cleanUp()
    mc2.cleanUp()


# ── Section 3: vec_size sweep ─────────────────────────────────────────────────
def section_vecsize_sweep():
    print(f"\n{SECTION}")
    print("  SECTION 3 — vec_size sweep for vec_fusion [1, 2, 4, 8]")
    print(f"  (Checks whether float8 register pressure slows down the kernel)")
    print(f"{SECTION}")

    print(f"\n  {'vec_size':>8}  {'elapsed (ms)':>14}  {'iters':>6}  {'ms/iter':>10}  {'fused knl (mean ms)':>20}")

    for vs in [1, 2, 4, 8]:
        if N_PATH % vs != 0:
            print(f"  {vs:>8}  skipped (nPath={N_PATH} not divisible by {vs})")
            continue
        mc = _make_mc()
        pso = PSO_vec_fusion_profiled(mc, N_FISH, vec_size=vs)
        elapsed, t_kernel, t_copy, t_argmax, t_gbest = pso.solvePsoAmerOption_cl_profiled()
        iters = len(t_kernel)
        print(f"  {vs:>8}  {elapsed:>14.2f}  {iters:>6}  {elapsed/iters:>10.3f}  "
              f"{np.mean(t_kernel):>20.3f}")
        mc.cleanUp()


# ── Section 4: nFish sweep ────────────────────────────────────────────────────
def section_nfish_sweep():
    print(f"\n{SECTION}")
    print("  SECTION 4 — nFish sweep [64, 128, 256, 512] for vec vs vec_fusion")
    print(f"  (Checks GPU occupancy effects)")
    print(f"{SECTION}")

    print(f"\n  {'nFish':>6}  {'vec (ms)':>10}  {'vec iters':>10}  {'fusion (ms)':>12}  {'fusion iters':>12}  {'ratio':>6}")
    print(f"  {'-'*66}")

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
        print(f"  {nf:>6}  {t_v:>10.2f}  {n_v:>10}  {t_vf:>12.2f}  {n_vf:>12}  {ratio:>6.2f}x")

        mc1.cleanUp()
        mc2.cleanUp()


# ── Section 5: Iteration detail dump ─────────────────────────────────────────
def section_iteration_detail():
    print(f"\n{SECTION}")
    print("  SECTION 5 — Per-iteration detail (first 10 iters): vec vs vec_fusion")
    print(f"{SECTION}")

    mc1 = _make_mc()
    pso_v = PSO_OpenCL_vec(mc1, N_FISH, vec_size=VEC_SIZE)
    _, _, search, fit, rest = pso_v.solvePsoAmerOption_cl()

    mc2 = _make_mc()
    pso_vf = PSO_vec_fusion_profiled(mc2, N_FISH, vec_size=VEC_SIZE)
    _, t_kernel, t_copy, t_argmax, t_gbest = pso_vf.solvePsoAmerOption_cl_profiled()

    print(f"\n  {'iter':>4}  {'vec_sg':>8}  {'vec_fit':>8}  {'vec_rest':>9}  "
          f"{'vf_kernel':>10}  {'vf_copy':>8}  {'vf_argmax':>10}  {'vf_gbest':>9}")
    print(f"  {'-'*80}")

    n_show = min(10, len(search), len(t_kernel))
    for i in range(n_show):
        sg   = search[i] if i < len(search) else 0
        ft   = fit[i]    if i < len(fit)    else 0
        rs   = rest[i]   if i < len(rest)   else 0
        tk   = t_kernel[i] if i < len(t_kernel) else 0
        tc   = t_copy[i]   if i < len(t_copy)   else 0
        ta   = t_argmax[i] if i < len(t_argmax) else 0
        tg   = t_gbest[i]  if i < len(t_gbest)  else 0
        print(f"  {i:>4}  {sg:>8.3f}  {ft:>8.3f}  {rs:>9.3f}  "
              f"{tk:>10.3f}  {tc:>8.3f}  {ta:>10.3f}  {tg:>9.3f}")

    if len(search) > 10:
        print(f"  ... ({len(search)} total iters for vec, {len(t_kernel)} for vec_fusion)")

    mc1.cleanUp()
    mc2.cleanUp()


# ── Section 6: nPath sweep — NumPy CPU vs scalar vs vec across path counts ────
def section_npath_sweep():
    print(f"\n{SECTION}")
    print("  SECTION 6 — nPath sweep: NumPy CPU vs GPU scalar vs GPU vec")
    print(f"  (ATM Put 02SEP08, nFish={N_FISH}, nPeriod={N_PERIOD}, vec_size={VEC_SIZE})")
    print(f"  Tests whether the slowdown is consistent or varies with simulation size.")
    print(f"{SECTION}")

    npath_values = [2000, 5000, 10_000, 20_000, 40_000]

    hdr = (f"  {'nPath':>7}  {'numpy (ms)':>11}  {'np iters':>9}  "
           f"{'scalar (ms)':>12}  {'sc iters':>9}  "
           f"{'vec (ms)':>10}  {'vc iters':>9}  "
           f"{'sc/np':>7}  {'vc/np':>7}  {'vc/sc':>7}")
    print(f"\n{hdr}")
    print(f"  {'-'*115}")

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

        r_sc_np = t_sc / t_np if t_np > 0 else float("nan")   # <1 = GPU scalar faster
        r_vc_np = t_vc / t_np if t_np > 0 else float("nan")   # <1 = GPU vec faster
        r_vc_sc = t_vc / t_sc if t_sc > 0 else float("nan")   # <1 = vec faster than scalar

        print(f"  {npath:>7}  {t_np:>11.2f}  {n_np:>9}  "
              f"{t_sc:>12.2f}  {n_sc:>9}  "
              f"{t_vc:>10.2f}  {n_vc:>9}  "
              f"{r_sc_np:>7.2f}x  {r_vc_np:>7.2f}x  {r_vc_sc:>7.2f}x")

        mc_np.cleanUp()
        mc_sc.cleanUp()
        mc_vc.cleanUp()

    print(f"\n  Columns: sc/np = GPU scalar vs NumPy (lower = GPU faster), "
          f"vc/np = GPU vec vs NumPy,")
    print(f"  vc/sc = GPU vec vs GPU scalar (should be >1 if vec is slower).")
    print(f"  Stable ratios across nPath values confirm a systemic bottleneck.")


# ── Section 7: Multi-case sweep — scalar vs vec across option cases ────────────
# Representative cases: ATM, OTM, ITM, vOTM, deep-OTM, deep-ITM
SWEEP_CASES = [
    # label,              S0,       K,         r,      sigma,   T,       opttype
    ("ATM  Put  02SEP08", 1277.58,  1280.749,  0.0172, 0.2134,  30/365, "P"),
    ("ATM  Call 02SEP08", 1277.58,  1280.343,  0.0172, 0.2097,  30/365, "C"),
    ("OTM  Put  15SEP08", 1192.70,  1126.930,  0.0102, 0.3172,  30/365, "P"),
    ("OTM  Call 15SEP08", 1192.70,  1265.262,  0.0102, 0.2867,  30/365, "C"),
    ("ITM  Put  15SEP08", 1192.70,  1267.000,  0.0102, 0.2900,  30/365, "P"),
    ("ITM  Call 15SEP08", 1192.70,  1125.000,  0.0102, 0.3300,  30/365, "C"),
    ("vOTM Put  29SEP08", 1106.42,   987.000,  0.0094, 0.4100,  30/365, "P"),
    ("vOTM Call 29SEP08", 1106.42,  1260.221,  0.0094, 0.4114,  30/365, "C"),
    ("dOTM Put  02SEP08", 1277.58,  1155.000,  0.0172, 0.2700,  30/365, "P"),
    ("dITM Call 29SEP08", 1106.42,   966.000,  0.0094, 0.3900,  30/365, "C"),
]

def section_multicase_sweep():
    print(f"\n{SECTION}")
    print("  SECTION 7 — Multi-case sweep: NumPy CPU vs GPU scalar vs GPU vec")
    print(f"  (nPath={N_PATH}, nPeriod={N_PERIOD}, nFish={N_FISH}, vec_size={VEC_SIZE})")
    print(f"  Tests whether the slowdown is option-specific or systemic.")
    print(f"{SECTION}")

    print(f"\n  {'Case':<24}  {'numpy (ms)':>11}  {'np iters':>9}  "
          f"{'scalar (ms)':>12}  {'sc iters':>9}  "
          f"{'vec (ms)':>10}  {'vc iters':>9}  "
          f"{'sc/np':>7}  {'vc/np':>7}  {'vc/sc':>7}")
    print(f"  {'-'*120}")

    ratios_vc_sc = []
    ratios_sc_np = []
    ratios_vc_np = []

    for label, s0, k, r, sigma, T, opttype in SWEEP_CASES:
        # NumPy CPU baseline
        mc_np = hybridMonteCarlo(s0, r, sigma, T, N_PATH, N_PERIOD, k, opttype, N_FISH)
        pso_np = PSO_Numpy(mc_np, N_FISH)
        _, t_np, search_np, fit_np, rest_np = pso_np.solvePsoAmerOption_np()
        n_np = len(search_np)

        # GPU scalar
        mc_sc = hybridMonteCarlo(s0, r, sigma, T, N_PATH, N_PERIOD, k, opttype, N_FISH)
        pso_sc = PSO_OpenCL_scalar(mc_sc, N_FISH)
        _, t_sc, search_sc, fit_sc, rest_sc = pso_sc.solvePsoAmerOption_cl()
        n_sc = len(search_sc)

        # GPU vec
        mc_vc = hybridMonteCarlo(s0, r, sigma, T, N_PATH, N_PERIOD, k, opttype, N_FISH)
        pso_vc = PSO_OpenCL_vec(mc_vc, N_FISH, vec_size=VEC_SIZE)
        _, t_vc, search_vc, fit_vc, rest_vc = pso_vc.solvePsoAmerOption_cl()
        n_vc = len(search_vc)

        r_sc_np = t_sc / t_np if t_np > 0 else float("nan")
        r_vc_np = t_vc / t_np if t_np > 0 else float("nan")
        r_vc_sc = t_vc / t_sc if t_sc > 0 else float("nan")

        ratios_vc_sc.append(r_vc_sc)
        ratios_sc_np.append(r_sc_np)
        ratios_vc_np.append(r_vc_np)

        print(f"  {label:<24}  {t_np:>11.2f}  {n_np:>9}  "
              f"{t_sc:>12.2f}  {n_sc:>9}  "
              f"{t_vc:>10.2f}  {n_vc:>9}  "
              f"{r_sc_np:>7.2f}x  {r_vc_np:>7.2f}x  {r_vc_sc:>7.2f}x")

        mc_np.cleanUp()
        mc_sc.cleanUp()
        mc_vc.cleanUp()

    print(f"  {'-'*120}")
    print(f"  {'Mean (sc/np)  GPU scalar vs NumPy':<50}  {np.mean(ratios_sc_np):>7.2f}x  "
          f"std={np.std(ratios_sc_np):.3f}")
    print(f"  {'Mean (vc/np)  GPU vec   vs NumPy':<50}  {np.mean(ratios_vc_np):>7.2f}x  "
          f"std={np.std(ratios_vc_np):.3f}")
    print(f"  {'Mean (vc/sc)  GPU vec   vs GPU scalar':<50}  {np.mean(ratios_vc_sc):>7.2f}x  "
          f"std={np.std(ratios_vc_sc):.3f}")
    print(f"\n  Interpretation:")
    print(f"    sc/np < 1 => GPU scalar faster than NumPy (expected).")
    print(f"    vc/np > 1 => GPU vec SLOWER than NumPy (register-spill penalty).")
    print(f"    vc/sc > 1 => GPU vec SLOWER than GPU scalar (systemic vec overhead).")
    print(f"    Low std => ratio is stable across option types (systemic, not case-specific).")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    from src.models.utils import checkOpenCL
    checkOpenCL()

    print(f"\n{'#'*72}")
    print("  PSO vec_fusion Performance Diagnostic")
    print(f"  Base case: ATM Put 02SEP08  nPath={N_PATH}  nPeriod={N_PERIOD}  nFish={N_FISH}  vec_size={VEC_SIZE}")
    print(f"{'#'*72}")

    section_substep_breakdown()
    section_kernel_info()
    section_vecsize_sweep()
    section_nfish_sweep()
    section_iteration_detail()
    section_npath_sweep()
    section_multicase_sweep()

    print(f"\n{'#'*72}")
    print("  Diagnostic complete.")
    print(f"{'#'*72}\n")


if __name__ == "__main__":
    main()