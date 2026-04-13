import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.linalg import expm
from scipy.special import jv
from scipy.stats import spearmanr

# ============================================================
# CONFIGURATION
# ============================================================
HBAR = 1.0

GAMMA = 4.0 * pi
M_RETAIN = 1

# Keep ONLY the strong-driving branch where weak-RWA has already broken down.
# Earlier discussion indicates you want to DROP gamma/omega0 < 1.49.
X_KEEP_MIN = 1.49

# Physical / numerical filters
J0_MIN = 1e-3          # drop J0 <= 0 and avoid blow-up near CDT zeros
JM_MIN = 1e-6          # for m=1 this is effectively |J1| > JM_MIN
EPS_THRESH = 0.1

# Fidelity groups
HIGH_THRESH = 0.85
LOW_THRESH = 0.30

# Scan density targets
MIN_OMEGA_POINTS_TO_KEEP = 2000
MIN_ALPHA_POINTS_TO_KEEP = 2500
OMEGA_RANGE = (0.0005 * 2.0 * pi, 3.0 * 2.0 * pi)
ALPHA_RANGE = (0.5, 200.0)
OMEGA_GRID_INITIAL = 12000
ALPHA_GRID_INITIAL = 12000
GRID_GROWTH_FACTOR = 2
GRID_MAX_POINTS = 96000

# Exact propagation settings
N_STEPS_GATE = 4001
TIMING_N_STEPS = 2001
TIMING_MAX_POINTS = 2500   # keep timing diagnostics tractable
TIMING_SEARCH_FACTOR_TPI = 3.0
TIMING_SEARCH_FACTOR_DRIVE = 6.0

# Plot style
COLOR_ALL = "tab:blue"
COLOR_LOW = "tab:blue"
COLOR_HIGH = "tab:orange"
COLOR_FILTER = "tab:orange"
COLOR_MIDDLE = "0.55"
ALPHA_ALL = 0.20
ALPHA_GROUP = 0.80
MARKER_SIZE_ALL = 10
MARKER_SIZE_GROUP = 14

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# BASIC OBJECTS
# ============================================================
def dagger(A: np.ndarray) -> np.ndarray:
    return A.conj().T

I2 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

ket0 = np.array([[1.0], [0.0]], dtype=complex)
ket1 = np.array([[0.0], [1.0]], dtype=complex)
bra1 = dagger(ket1)


def p1_of_state(psi: np.ndarray) -> float:
    amp = (bra1 @ psi).item()
    return float(np.abs(amp) ** 2)


def alpha_of(omega: float, gamma: float) -> float:
    return 2.0 * gamma / omega


def H_lab_param(t: float, omega: float, omega_0: float, gamma: float) -> np.ndarray:
    return 0.5 * HBAR * omega_0 * sigma_z + HBAR * gamma * np.cos(omega * t) * sigma_x


def U_step_from_H(H: np.ndarray, dt: float) -> np.ndarray:
    return expm(-1j * H * dt / HBAR)


def evolve_state_history_on_grid(H_func, times: np.ndarray, psi0: np.ndarray) -> np.ndarray:
    psi = psi0.copy()
    hist = [psi.copy()]
    for k in range(len(times) - 1):
        t_mid = 0.5 * (times[k] + times[k + 1])
        dt = times[k + 1] - times[k]
        H_mid = H_func(t_mid)
        psi = U_step_from_H(H_mid, dt) @ psi
        hist.append(psi.copy())
    return np.array(hist)


def evolve_final_state_to_time(H_func, t_final: float, psi0: np.ndarray, n_steps: int = 4001) -> np.ndarray:
    if not np.isfinite(t_final) or t_final <= 0:
        return psi0.copy()
    times_loc = np.linspace(0.0, float(t_final), int(n_steps))
    hist = evolve_state_history_on_grid(H_func, times_loc, psi0)
    return hist[-1]


# ============================================================
# METHOD-2 RESONANCE LOGIC ONLY
# Enforce delta = m*omega - omega0*J0(alpha) = 0 exactly.
# Keep only J0(alpha) > 0 branch.
# ============================================================
def derive_point_from_omega(omega: float, gamma: float = GAMMA, m: int = M_RETAIN):
    if omega <= 0:
        return None

    alpha = alpha_of(omega, gamma)
    J0 = jv(0, alpha)
    Jm = jv(m, alpha)
    J1_abs = abs(jv(1, alpha))
    J3_abs = abs(jv(3, alpha))
    J5_abs = abs(jv(5, alpha))

    # Keep only the positive-J0 branch (user requested: J0 < 0 is unphysical here).
    if J0 <= J0_MIN:
        return None
    if abs(Jm) < JM_MIN:
        return None

    omega0 = (m * omega) / J0
    x = gamma / omega0
    if x < X_KEEP_MIN:
        return None

    t_pi_A = np.pi / (omega0 * abs(Jm))
    T_drive = 2.0 * np.pi / omega
    r_tpi_over_Tdrive = t_pi_A / T_drive
    r_from_bessel = J0 / (2.0 * m * abs(Jm) + 1e-15)

    Omega_used = omega0 * J0
    delta = m * omega - Omega_used
    fast_gap = abs(m * omega + Omega_used)
    slow_gap = abs(delta)

    eps_fast = (omega0 * abs(Jm)) / (fast_gap + 1e-12)
    eps3 = omega0 * J3_abs / (abs(3.0 * omega - Omega_used) + 1e-12)
    eps5 = omega0 * J5_abs / (abs(5.0 * omega - Omega_used) + 1e-12)
    eps_bad_max = max(eps_fast, eps3, eps5)
    eps_bad_sum = eps_fast + eps3 + eps5
    eps_bad_rss = np.sqrt(eps_fast ** 2 + eps3 ** 2 + eps5 ** 2)

    return {
        "omega": float(omega),
        "omega0": float(omega0),
        "alpha": float(alpha),
        "J0": float(J0),
        "Jm": float(Jm),
        "J1_abs": float(J1_abs),
        "J3_abs": float(J3_abs),
        "J5_abs": float(J5_abs),
        "gamma_over_omega_0": float(x),
        "omega_over_omega_0": float(omega / omega0),
        "t_pi_A": float(t_pi_A),
        "T_drive": float(T_drive),
        "r_tpi_over_Tdrive": float(r_tpi_over_Tdrive),
        "r_from_bessel": float(r_from_bessel),
        "Omega_used": float(Omega_used),
        "delta": float(delta),
        "fast_gap": float(fast_gap),
        "slow_gap": float(slow_gap),
        "eps_fast": float(eps_fast),
        "eps3": float(eps3),
        "eps5": float(eps5),
        "eps_bad_max": float(eps_bad_max),
        "eps_bad_sum": float(eps_bad_sum),
        "eps_bad_rss": float(eps_bad_rss),
    }


def derive_point_from_alpha(alpha: float, gamma: float = GAMMA, m: int = M_RETAIN):
    if alpha <= 0:
        return None
    omega = 2.0 * gamma / alpha
    return derive_point_from_omega(omega=omega, gamma=gamma, m=m)


# ============================================================
# SCAN BUILDERS
# Build dense candidate grids and keep only points that survive:
#   J0 > 0, |Jm| > threshold, gamma/omega0 >= X_KEEP_MIN
# ============================================================
def _prefilter_omega_grid(omega_grid: np.ndarray, gamma: float = GAMMA, m: int = M_RETAIN) -> np.ndarray:
    omega = np.asarray(omega_grid)
    alpha = 2.0 * gamma / omega
    J0 = jv(0, alpha)
    Jm = np.abs(jv(m, alpha))
    with np.errstate(divide="ignore", invalid="ignore"):
        omega0 = (m * omega) / J0
        x = gamma / omega0
    mask = np.isfinite(alpha) & np.isfinite(J0) & np.isfinite(Jm)
    mask &= (J0 > J0_MIN)
    mask &= (Jm > JM_MIN)
    mask &= np.isfinite(x)
    mask &= (x >= X_KEEP_MIN)
    return mask


def _prefilter_alpha_grid(alpha_grid: np.ndarray, gamma: float = GAMMA, m: int = M_RETAIN) -> np.ndarray:
    alpha = np.asarray(alpha_grid)
    omega = 2.0 * gamma / alpha
    return _prefilter_omega_grid(omega, gamma=gamma, m=m)


def build_omega_grid(min_keep: int = MIN_OMEGA_POINTS_TO_KEEP,
                     omega_range=OMEGA_RANGE,
                     n_initial: int = OMEGA_GRID_INITIAL) -> np.ndarray:
    n = int(n_initial)
    while True:
        grid = np.linspace(float(omega_range[0]), float(omega_range[1]), n)
        keep_count = int(np.count_nonzero(_prefilter_omega_grid(grid)))
        if keep_count >= min_keep:
            print(f"Omega-grid prefilter count = {keep_count} using {n} raw points.")
            return grid
        if n >= GRID_MAX_POINTS:
            raise RuntimeError(
                f"Could not reach {min_keep} kept omega points before hitting GRID_MAX_POINTS={GRID_MAX_POINTS}. "
                f"Last keep_count={keep_count}. Widen omega_range or lower min_keep."
            )
        n *= GRID_GROWTH_FACTOR


def build_alpha_grid(min_keep: int = MIN_ALPHA_POINTS_TO_KEEP,
                     alpha_range=ALPHA_RANGE,
                     n_initial: int = ALPHA_GRID_INITIAL) -> np.ndarray:
    n = int(n_initial)
    while True:
        grid = np.linspace(float(alpha_range[0]), float(alpha_range[1]), n)
        keep_count = int(np.count_nonzero(_prefilter_alpha_grid(grid)))
        if keep_count >= min_keep:
            print(f"Alpha-grid prefilter count = {keep_count} using {n} raw points.")
            return grid
        if n >= GRID_MAX_POINTS:
            raise RuntimeError(
                f"Could not reach {min_keep} kept alpha points before hitting GRID_MAX_POINTS={GRID_MAX_POINTS}. "
                f"Last keep_count={keep_count}. Widen alpha_range or lower min_keep."
            )
        n *= GRID_GROWTH_FACTOR


def _empty_results_dict() -> dict:
    return {
        "omega": [],
        "omega0": [],
        "alpha": [],
        "J0": [],
        "Jm": [],
        "J1_abs": [],
        "J3_abs": [],
        "J5_abs": [],
        "gamma_over_omega_0": [],
        "omega_over_omega_0": [],
        "t_pi_A": [],
        "T_drive": [],
        "r_tpi_over_Tdrive": [],
        "r_from_bessel": [],
        "F_gate": [],
        "Omega_used": [],
        "delta": [],
        "fast_gap": [],
        "slow_gap": [],
        "eps_fast": [],
        "eps3": [],
        "eps5": [],
        "eps_bad_max": [],
        "eps_bad_sum": [],
        "eps_bad_rss": [],
        "scan_source": [],
    }


def _append_result(out: dict, point: dict, F_gate: float, scan_source: str) -> None:
    for key in out.keys():
        if key == "F_gate":
            out[key].append(float(F_gate))
        elif key == "scan_source":
            out[key].append(scan_source)
        else:
            out[key].append(point[key])


def finalize_results(out: dict) -> dict:
    final = {}
    for key, val in out.items():
        if key == "scan_source":
            final[key] = np.array(val, dtype=object)
        else:
            final[key] = np.array(val, dtype=float)
    return final


def omega_scan_method2(
    gamma: float = GAMMA,
    m: int = M_RETAIN,
    n_steps: int = N_STEPS_GATE,
    omega_range=OMEGA_RANGE,
    min_keep: int = MIN_OMEGA_POINTS_TO_KEEP,
) -> dict:
    omega_grid = build_omega_grid(min_keep=min_keep, omega_range=omega_range)
    out = _empty_results_dict()

    keep_mask = _prefilter_omega_grid(omega_grid, gamma=gamma, m=m)
    omega_candidates = omega_grid[keep_mask]
    print(f"Running omega scan on {len(omega_candidates)} exact-evolution points...")

    for omega in omega_candidates:
        point = derive_point_from_omega(omega, gamma=gamma, m=m)
        if point is None:
            continue
        H_ex = lambda t, ww=point["omega"], w0=point["omega0"], g=gamma: H_lab_param(t, ww, w0, g)
        psi = evolve_final_state_to_time(H_ex, point["t_pi_A"], ket0, n_steps=n_steps)
        F_gate = p1_of_state(psi)
        _append_result(out, point, F_gate, scan_source="omega")

    res = finalize_results(out)
    if len(res["F_gate"]) < min_keep:
        raise RuntimeError(
            f"Omega scan kept only {len(res['F_gate'])} points after exact propagation; expected at least {min_keep}."
        )
    return res


def alpha_scan_method2(
    gamma: float = GAMMA,
    m: int = M_RETAIN,
    n_steps: int = N_STEPS_GATE,
    alpha_range=ALPHA_RANGE,
    min_keep: int = MIN_ALPHA_POINTS_TO_KEEP,
) -> dict:
    alpha_grid = build_alpha_grid(min_keep=min_keep, alpha_range=alpha_range)
    out = _empty_results_dict()

    keep_mask = _prefilter_alpha_grid(alpha_grid, gamma=gamma, m=m)
    alpha_candidates = alpha_grid[keep_mask]
    print(f"Running alpha scan on {len(alpha_candidates)} exact-evolution points...")

    for alpha in alpha_candidates:
        point = derive_point_from_alpha(alpha, gamma=gamma, m=m)
        if point is None:
            continue
        H_ex = lambda t, ww=point["omega"], w0=point["omega0"], g=gamma: H_lab_param(t, ww, w0, g)
        psi = evolve_final_state_to_time(H_ex, point["t_pi_A"], ket0, n_steps=n_steps)
        F_gate = p1_of_state(psi)
        _append_result(out, point, F_gate, scan_source="alpha")

    res = finalize_results(out)
    if len(res["F_gate"]) < min_keep:
        raise RuntimeError(
            f"Alpha scan kept only {len(res['F_gate'])} points after exact propagation; expected at least {min_keep}."
        )
    return res


# ============================================================
# GROUPING + REPORTING
# ============================================================
def group_masks(res: dict,
                high_thresh: float = HIGH_THRESH,
                low_thresh: float = LOW_THRESH) -> dict:
    F = res["F_gate"]
    mask_high = F >= high_thresh
    mask_low = F <= low_thresh
    mask_middle = (~mask_high) & (~mask_low)
    return {
        "high": mask_high,
        "low": mask_low,
        "middle": mask_middle,
    }


def print_scan_overview(res: dict, label: str) -> None:
    print(f"\n===== {label} =====")
    print(f"Number of kept points          : {len(res['F_gate'])}")
    print(f"gamma                         : {GAMMA:.6f}")
    print(f"m retained                    : {M_RETAIN}")
    print(f"x = gamma/omega0 kept minimum : {X_KEEP_MIN}")
    print(f"J0 branch                     : positive only, J0 > {J0_MIN}")
    print(f"|J1| threshold                : {JM_MIN}")
    print(f"F range                       : [{np.min(res['F_gate']):.6f}, {np.max(res['F_gate']):.6f}]")
    print(f"x range                       : [{np.min(res['gamma_over_omega_0']):.6f}, {np.max(res['gamma_over_omega_0']):.6f}]")
    print(f"alpha range                   : [{np.min(res['alpha']):.6f}, {np.max(res['alpha']):.6f}]")


def print_group_counts(res: dict, masks: dict) -> None:
    print("\n===== FIDELITY GROUP COUNTS =====")
    print(f"Total points  : {len(res['F_gate'])}")
    print(f"High points   : {np.count_nonzero(masks['high'])}   (F >= {HIGH_THRESH})")
    print(f"Low points    : {np.count_nonzero(masks['low'])}   (F <= {LOW_THRESH})")
    print(f"Middle points : {np.count_nonzero(masks['middle'])}   ({LOW_THRESH} < F < {HIGH_THRESH})")


def print_group_summary(res: dict, label: str, mask: np.ndarray, n_rep: int = 10) -> None:
    idx = np.where(mask)[0]
    print(f"\n===== {label.upper()} FIDELITY GROUP =====")
    if len(idx) == 0:
        print("No points in this group.")
        return

    summary_keys = [
        "F_gate", "omega", "omega0", "gamma_over_omega_0", "omega_over_omega_0",
        "alpha", "J0", "J1_abs", "J3_abs", "J5_abs", "t_pi_A", "T_drive",
        "r_tpi_over_Tdrive", "r_from_bessel", "eps_fast", "eps3", "eps5",
        "eps_bad_max", "eps_bad_sum", "eps_bad_rss",
    ]

    print(f"Number of points: {len(idx)}")
    for key in summary_keys:
        arr = res[key][idx]
        print(
            f"{key:18s}: mean={np.mean(arr):.6g}, median={np.median(arr):.6g}, "
            f"min={np.min(arr):.6g}, max={np.max(arr):.6g}"
        )

    # Representative points
    F = res["F_gate"][idx]
    eps = res["eps_bad_sum"][idx]
    tpi = res["t_pi_A"][idx]

    if label.lower() == "high":
        order_local = np.lexsort((tpi, eps, -F))
    elif label.lower() == "low":
        order_local = np.lexsort((tpi, eps, F))
    else:
        target = 0.5 * (HIGH_THRESH + LOW_THRESH)
        order_local = np.lexsort((tpi, eps, np.abs(F - target)))

    idx_show = idx[order_local[:n_rep]]
    print("\nRepresentative points:")
    rep_keys = [
        "F_gate", "omega", "omega0", "gamma_over_omega_0", "omega_over_omega_0",
        "alpha", "J0", "Jm", "J1_abs", "J3_abs", "J5_abs", "t_pi_A", "T_drive",
        "r_tpi_over_Tdrive", "r_from_bessel", "Omega_used", "delta", "fast_gap",
        "slow_gap", "eps_fast", "eps3", "eps5", "eps_bad_max", "eps_bad_sum", "eps_bad_rss"
    ]
    for j, i in enumerate(idx_show, start=1):
        parts = [f"[{j}]"]
        for key in rep_keys:
            parts.append(f"{key}={res[key][i]:.6g}")
        print(", ".join(parts))

def print_tpi_over_topt_summary_by_fidelity_group(
    res: dict,
    timing: dict,
    high_thresh: float = 0.85,
    low_thresh: float = 0.30,
):
    """
    Print summary statistics of t_pi^A / t_opt for the
    high-, middle-, and low-fidelity groups.

    Inputs
    ------
    res["F_gate"]              : fidelity array already in your scan result
    timing["timing_ratio"]     : array of t_pi^A / t_opt from compute_timing_diagnostics(res)
    """

    F = res["F_gate"]
    ratio = timing["timing_ratio"]

    # valid points only
    mask_valid = np.isfinite(F) & np.isfinite(ratio) & (ratio > 0)

    mask_high = mask_valid & (F >= high_thresh)
    mask_low  = mask_valid & (F <= low_thresh)
    mask_mid  = mask_valid & (F > low_thresh) & (F < high_thresh)

    def _print_one_group(label, mask):
        vals = ratio[mask]
        print(f"\n===== {label} =====")
        print(f"Number of points: {len(vals)}")
        if len(vals) == 0:
            print("No valid points in this group.")
            return

        print(f"t_pi^A/t_opt : mean={np.mean(vals):.6g}, "
              f"median={np.median(vals):.6g}, "
              f"min={np.min(vals):.6g}, "
              f"max={np.max(vals):.6g}")

    print("\n==============================================")
    print("SUMMARY OF t_pi^A/t_opt BY FIDELITY GROUP")
    print(f"High fidelity   : F >= {high_thresh}")
    print(f"Middle fidelity : {low_thresh} < F < {high_thresh}")
    print(f"Low fidelity    : F <= {low_thresh}")
    print("==============================================")

    _print_one_group("HIGH FIDELITY GROUP", mask_high)
    _print_one_group("MIDDLE FIDELITY GROUP", mask_mid)
    _print_one_group("LOW FIDELITY GROUP", mask_low)
    

def print_filtered_fidelity_group_counts(
    res: dict,
    eps_key: str = "eps_bad_rss",
    eps_thresh: float = 0.1,
    high_thresh: float = 0.85,
    low_thresh: float = 0.30,
):
    """
    Count how many filtered points fall into low / middle / high fidelity groups.

    low    : F <= low_thresh
    middle : low_thresh < F < high_thresh
    high   : F >= high_thresh
    """
    F = res["F_gate"]
    eps = res[eps_key]

    mask_valid = np.isfinite(F) & np.isfinite(eps)
    mask_filt = mask_valid & (eps < eps_thresh)

    F_filt = F[mask_filt]
    n_total = len(F_filt)

    print("\n===== FILTERED FIDELITY-GROUP COUNTS =====")
    print(f"Indicator used   : {eps_key}")
    print(f"Threshold        : {eps_key} < {eps_thresh}")
    print(f"Total kept points: {n_total}")

    if n_total == 0:
        print("No points survive this filter.")
        return

    n_low = np.count_nonzero(F_filt <= low_thresh)
    n_mid = np.count_nonzero((F_filt > low_thresh) & (F_filt < high_thresh))
    n_high = np.count_nonzero(F_filt >= high_thresh)

    print(f"Low fidelity     : {n_low} ({100*n_low/n_total:.2f}%)   [F <= {low_thresh}]")
    print(f"Middle fidelity  : {n_mid} ({100*n_mid/n_total:.2f}%)   [{low_thresh} < F < {high_thresh}]")
    print(f"High fidelity    : {n_high} ({100*n_high/n_total:.2f}%)   [F >= {high_thresh}]")
    
    
# ============================================================
# TIMING DIAGNOSTICS
# ============================================================
def compute_timing_diagnostics(res: dict,
                               n_steps: int = TIMING_N_STEPS,
                               max_points: int = TIMING_MAX_POINTS) -> dict:
    n_total = len(res["F_gate"])
    if max_points is None or max_points >= n_total:
        idx_use = np.arange(n_total)
    else:
        idx_use = np.linspace(0, n_total - 1, max_points).astype(int)

    F_pred = res["F_gate"][idx_use]
    x = res["gamma_over_omega_0"][idx_use]

    F_opt = np.empty(len(idx_use), dtype=float)
    t_opt = np.empty(len(idx_use), dtype=float)
    t_pi = res["t_pi_A"][idx_use]

    print(f"Computing timing diagnostics on {len(idx_use)} points...")
    for k, i in enumerate(idx_use):
        omega = res["omega"][i]
        omega0 = res["omega0"][i]
        t_pi_A = res["t_pi_A"][i]
        T_drive = res["T_drive"][i]
        t_search = max(TIMING_SEARCH_FACTOR_TPI * t_pi_A,
                       TIMING_SEARCH_FACTOR_DRIVE * T_drive)
        times = np.linspace(0.0, t_search, n_steps)
        H_ex = lambda t, ww=omega, w0=omega0, g=GAMMA: H_lab_param(t, ww, w0, g)
        hist = evolve_state_history_on_grid(H_ex, times, ket0)
        P1 = np.array([p1_of_state(hist[j]) for j in range(len(times))])
        idx_best = int(np.argmax(P1))
        F_opt[k] = float(P1[idx_best])
        t_opt[k] = float(times[idx_best])

    return {
        "idx_use": idx_use,
        "gamma_over_omega_0": x,
        "F_pred": F_pred,
        "F_opt": F_opt,
        "t_pi_A": t_pi,
        "t_opt": t_opt,
        "timing_ratio": t_pi / (t_opt + 1e-15),
    }


# ============================================================
# PLOTTING HELPERS
# ============================================================
def _plot_group_overlays(ax, x, y, masks, xlabel, ylabel, title, logx=False, xlim=None, ylim=(-0.02, 1.02)):
    ax.scatter(x, y, s=MARKER_SIZE_ALL, alpha=ALPHA_ALL, color=COLOR_ALL, label="All points")
    if np.any(masks["low"]):
        ax.scatter(x[masks["low"]], y[masks["low"]], s=MARKER_SIZE_GROUP, alpha=ALPHA_GROUP,
                   color=COLOR_LOW, label=rf"Low fidelity: $F \leq {LOW_THRESH}$")
    if np.any(masks["high"]):
        ax.scatter(x[masks["high"]], y[masks["high"]], s=MARKER_SIZE_GROUP, alpha=ALPHA_GROUP,
                   color=COLOR_HIGH, label=rf"High fidelity: $F \geq {HIGH_THRESH}$")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)


def plot_fidelity_vs_quantity(res: dict, key: str, xlabel: str, title: str, logx: bool = False):
    masks = group_masks(res)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    _plot_group_overlays(ax, res[key], res["F_gate"], masks, xlabel, r"Fidelity  $P_{|1⟩,\rm exact}(t_\pi^A)$", title, logx=logx)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_filtered_gamma_vs_fidelity(res: dict, eps_key: str, eps_thresh: float = EPS_THRESH, title: str = ""):
    x = res["gamma_over_omega_0"]
    F = res["F_gate"]
    eps = res[eps_key]
    mask = eps < eps_thresh

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(x, F, s=MARKER_SIZE_ALL, alpha=ALPHA_ALL, color=COLOR_ALL, label="All points")
    ax.scatter(x[mask], F[mask], s=MARKER_SIZE_GROUP, alpha=ALPHA_GROUP,
               color=COLOR_FILTER, label=rf"Filtered: {eps_key} < {eps_thresh}")
    ax.set_xlabel(r"$\gamma/\omega_0$")
    ax.set_ylabel(r"Fidelity  $P_{|1⟩,\rm exact}(t_\pi^A)$")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"{eps_key}: kept {np.count_nonzero(mask)} / {len(mask)} points with threshold {eps_thresh}.")


def plot_all_histograms(res: dict):
    masks = group_masks(res)
    hist_specs = [
        ("F_gate", False, r"$F$"),
        ("omega", True, r"$\omega$"),
        ("omega0", True, r"$\omega_0$"),
        ("gamma_over_omega_0", True, r"$\gamma/\omega_0$"),
        ("omega_over_omega_0", True, r"$\omega/\omega_0$"),
        ("alpha", True, r"$\alpha$"),
        ("J0", False, r"$J_0(\alpha)$"),
        ("J1_abs", True, r"$|J_1(\alpha)|$"),
        ("J3_abs", True, r"$|J_3(\alpha)|$"),
        ("J5_abs", True, r"$|J_5(\alpha)|$"),
        ("t_pi_A", True, r"$t_\pi^A$"),
        ("T_drive", True, r"$T_{\rm drive}$"),
        ("r_tpi_over_Tdrive", True, r"$t_\pi^A/T_{\rm drive}$"),
        ("r_from_bessel", True, r"$J_0/(2|J_1|)$"),
        ("eps_fast", True, r"$\epsilon_{\rm fast}$"),
        ("eps3", True, r"$\epsilon_3$"),
        ("eps5", True, r"$\epsilon_5$"),
        ("eps_bad_max", True, r"$\epsilon_{\rm bad}^{(\max)}$"),
        ("eps_bad_sum", True, r"$\epsilon_{\rm bad}^{(\Sigma)}$"),
        ("eps_bad_rss", True, r"$\epsilon_{\rm bad}^{(\mathrm{rss})}$"),
    ]

    ncols = 4
    nrows = int(np.ceil(len(hist_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, use_log, xlabel) in zip(axes, hist_specs):
        arr = res[key]
        if use_log:
            arr_plot = arr + 1e-15
        else:
            arr_plot = arr

        ax.hist(arr_plot[masks["low"]], bins=35, density=True, alpha=0.65,
                color=COLOR_LOW, label="Low")
        ax.hist(arr_plot[masks["high"]], bins=35, density=True, alpha=0.65,
                color=COLOR_HIGH, label="High")
        if use_log:
            ax.set_xscale("log")
        ax.set_title(key)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.20)

    for ax in axes[len(hist_specs):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.show()


def print_timing_ratio_counts(timing: dict, tol: float = 0.05):
    """
    Count how many timing-ratio points are:
      - clearly below 1
      - approximately equal to 1 within a tolerance
      - clearly above 1

    tol = 0.05 means a 5% band around 1, i.e. [0.95, 1.05].
    """
    r = timing["timing_ratio"]
    mask = np.isfinite(r) & (r > 0)
    r_use = r[mask]

    if len(r_use) == 0:
        print("\n===== TIMING-RATIO COUNTS =====")
        print("No valid timing-ratio points.")
        return

    n_total = len(r_use)
    n_less = np.count_nonzero(r_use < 1.0 - tol)
    n_near = np.count_nonzero(np.abs(r_use - 1.0) <= tol)
    n_more = np.count_nonzero(r_use > 1.0 + tol)

    print("\n===== TIMING-RATIO COUNTS =====")
    print(f"Tolerance band around 1      : ±{tol:.3f}")
    print(f"Near-1 interval              : [{1.0 - tol:.3f}, {1.0 + tol:.3f}]")
    print(f"Total valid points           : {n_total}")
    print(f"t_pi^A/t_opt < 1 - tol       : {n_less} ({100*n_less/n_total:.2f}%)")
    print(f"|t_pi^A/t_opt - 1| <= tol    : {n_near} ({100*n_near/n_total:.2f}%)")
    print(f"t_pi^A/t_opt > 1 + tol       : {n_more} ({100*n_more/n_total:.2f}%)")

    # Optional extra summary
    print(f"Median(t_pi^A/t_opt)         : {np.median(r_use):.6f}")
    print(f"Mean(t_pi^A/t_opt)           : {np.mean(r_use):.6f}")
    

def plot_log_timing_ratio_histogram(timing: dict, bins: int = 50):
    """
    Histogram of log10(t_pi^A / t_opt).
    Negative values: predicted time too short
    Positive values: predicted time too long
    Zero: perfect timing
    """
    r = timing["timing_ratio"]
    mask = np.isfinite(r) & (r > 0)
    log_r = np.log10(r[mask])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(log_r, bins=bins, density=False, alpha=0.75, color="tab:blue")
    ax.axvline(0.0, linestyle="--", color="black", linewidth=1.5,
               label=r"$\log_{10}(t_\pi^A/t_{\rm opt})=0$")
    ax.set_xlabel(r"$\log_{10}(t_\pi^A/t_{\rm opt})$")
    ax.set_ylabel("Count")
    ax.set_title(r"Distribution of timing mismatch")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_log_timing_ratio_vs_gamma(timing: dict, y_abs_max: float = 2.0):
    """
    Scatter of log10(t_pi^A / t_opt) vs gamma/omega0.
    Negative values: predicted time too short
    Positive values: predicted time too long
    """
    x = timing["gamma_over_omega_0"]
    r = timing["timing_ratio"]

    mask = np.isfinite(x) & np.isfinite(r) & (r > 0)
    x_use = x[mask]
    y_use = np.log10(r[mask])

    # optional clipping for readability
    mask_plot = np.abs(y_use) <= y_abs_max

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(x_use[mask_plot], y_use[mask_plot],
               s=MARKER_SIZE_ALL, alpha=0.65, color=COLOR_ALL)
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1.5,
               label=r"$\log_{10}(t_\pi^A/t_{\rm opt})=0$")
    ax.set_xlabel(r"$\gamma/\omega_0$")
    ax.set_ylabel(r"$\log_{10}(t_\pi^A/t_{\rm opt})$")
    ax.set_title(r"Timing mismatch on log scale")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()

    n_hidden = np.count_nonzero(~mask_plot)
    print(f"log-timing-ratio scatter: hidden {n_hidden} point(s) outside |log10(t_pi^A/t_opt)| <= {y_abs_max}.")


def plot_timing_vs_dynamics(timing: dict):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(timing["F_pred"], timing["F_opt"], s=MARKER_SIZE_ALL, alpha=0.65, color=COLOR_ALL)
    lim = [0.0, 1.02]
    ax.plot(lim, lim, linestyle="--", color="black", linewidth=1.5, label=r"$F_{\rm opt}=F_{\rm pred}$")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(r"$F_{\rm pred}=P_{|1\rangle,\rm exact}(t_\pi^A)$")
    ax.set_ylabel(r"$F_{\rm opt}=\max_t P_{|1\rangle,\rm exact}(t)$")
    ax.set_title("Timing vs dynamics: predicted fidelity vs optimal exact fidelity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_timing_error_indicator(timing: dict, y_max_plot: float = 100.0):
    """
    Plot timing ratio t_pi^A / t_opt, but hide very large outliers
    from the figure for readability.
    """
    x = timing["gamma_over_omega_0"]
    y = timing["timing_ratio"]

    # keep only points within plotting range
    mask_plot = np.isfinite(x) & np.isfinite(y) & (y <= y_max_plot)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(
        x[mask_plot], y[mask_plot],
        s=MARKER_SIZE_ALL, alpha=0.65, color=COLOR_ALL
    )
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.5,
               label=r"$t_\pi^A/t_{\rm opt}=1$")
    ax.set_xlabel(r"$\gamma/\omega_0$")
    ax.set_ylabel(r"$t_\pi^A/t_{\rm opt}$")
    ax.set_title("Timing error indicator")
    ax.set_ylim(0, y_max_plot)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.ylim(0,2)
    plt.tight_layout()
    plt.show()

    n_hidden = np.count_nonzero(~mask_plot)
    print(f"Timing-error plot: hidden {n_hidden} point(s) with t_pi^A/t_opt > {y_max_plot}.")


def plot_fidelity_vs_tpi_over_topt(timing: dict, x_max_plot: float = 100.0):
    """
    Plot fidelity at the predicted Ashhab pi-pulse time versus the timing ratio t_pi^A / t_opt.
    Large outliers can be hidden for readability.
    """
    x = timing["timing_ratio"]   # t_pi^A / t_opt
    y = timing["F_pred"]         # fidelity evaluated at t_pi^A

    mask_plot = np.isfinite(x) & np.isfinite(y) & (x > 0) & (x <= x_max_plot)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(
        x[mask_plot], y[mask_plot],
        s=MARKER_SIZE_ALL, alpha=0.65, color=COLOR_ALL
    )
    ax.axvline(1.0, linestyle="--", color="black", linewidth=1.5,
               label=r"$t_\pi^A/t_{\rm opt}=1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_\pi^A/t_{\rm opt}$")
    ax.set_ylabel(r"$F_{\rm pred}=P_{|1\rangle,\rm exact}(t_\pi^A)$")
    ax.set_title(r"Fidelity vs timing ratio $t_\pi^A/t_{\rm opt}$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()

    n_hidden = np.count_nonzero(~mask_plot)
    print(f"Fidelity-vs-(t_pi^A/t_opt) plot: hidden {n_hidden} point(s) outside plotting range.")


def print_spearman_fidelity_vs_tpi_over_topt(timing: dict):
    """
    Spearman rank test between fidelity F_pred and timing ratio t_pi^A / t_opt.
    Also tests log10(t_pi^A/t_opt), since the ratio spans many decades.
    """
    F = timing["F_pred"]
    r = timing["timing_ratio"]

    mask = np.isfinite(F) & np.isfinite(r) & (r > 0)
    F_use = F[mask]
    r_use = r[mask]

    if len(F_use) < 3:
        print("\n===== SPEARMAN TEST: fidelity vs t_pi^A/t_opt =====")
        print("Not enough valid points to run Spearman test.")
        return

    rho_raw, p_raw = spearmanr(r_use, F_use)

    log_r_use = np.log10(r_use)
    rho_log, p_log = spearmanr(log_r_use, F_use)

    print("\n===== SPEARMAN TEST: fidelity vs t_pi^A/t_opt =====")
    print(f"N valid points            = {len(F_use)}")
    print(f"Raw ratio:     rho        = {rho_raw:.6f}")
    print(f"Raw ratio:     p-value    = {p_raw:.6e}")
    print(f"log10(ratio):  rho        = {rho_log:.6f}")
    print(f"log10(ratio):  p-value    = {p_log:.6e}")

    def strength_label(rho):
        a = abs(rho)
        if a < 0.1:
            return "negligible"
        elif a < 0.3:
            return "weak"
        elif a < 0.5:
            return "moderate"
        else:
            return "strong"

    print("\nInterpretation:")
    print(f"- Raw ratio correlation strength    : {strength_label(rho_raw)}")
    print(f"- log10(ratio) correlation strength : {strength_label(rho_log)}")

    if p_log < 0.05:
        print("- The monotonic correlation using log10(t_pi^A/t_opt) is statistically significant.")
    else:
        print("- The monotonic correlation using log10(t_pi^A/t_opt) is not statistically significant.")

    if rho_log < 0:
        print("- Larger t_pi^A/t_opt tends to be associated with lower fidelity.")
    elif rho_log > 0:
        print("- Larger t_pi^A/t_opt tends to be associated with higher fidelity.")
    else:
        print("- No monotonic trend is detected.")


def print_spearman_fidelity_vs_timing_ratio(res: dict):
    F = res["F_gate"]
    r = res["r_tpi_over_Tdrive"]

    mask = np.isfinite(F) & np.isfinite(r) & (r > 0)
    F_use = F[mask]
    r_use = r[mask]

    if len(F_use) < 3:
        print("\n===== SPEARMAN TEST: fidelity vs t_pi^A/T_drive =====")
        print("Not enough valid points to run Spearman test.")
        return

    rho_raw, p_raw = spearmanr(r_use, F_use)

    log_r_use = np.log10(r_use)
    rho_log, p_log = spearmanr(log_r_use, F_use)

    print("\n===== SPEARMAN TEST: fidelity vs t_pi^A/T_drive =====")
    print(f"N valid points         = {len(F_use)}")
    print(f"Raw ratio:     rho     = {rho_raw:.6f}")
    print(f"Raw ratio:     p-value = {p_raw:.6e}")
    print(f"log10(ratio):  rho     = {rho_log:.6f}")
    print(f"log10(ratio):  p-value = {p_log:.6e}")

    # simple interpretation
    def strength_label(rho):
        a = abs(rho)
        if a < 0.1:
            return "negligible"
        elif a < 0.3:
            return "weak"
        elif a < 0.5:
            return "moderate"
        else:
            return "strong"

    print("\nInterpretation:")
    print(f"- Raw ratio correlation strength    : {strength_label(rho_raw)}")
    print(f"- log10(ratio) correlation strength : {strength_label(rho_log)}")

    if p_log < 0.05:
        print("- The monotonic correlation using log10(ratio) is statistically significant.")
    else:
        print("- The monotonic correlation using log10(ratio) is not statistically significant.")

# ============================================================
# OPTIONAL EXTRA DIAGNOSTIC I RECOMMEND
# This one is often very informative: fidelity coloured by eps_bad_sum
# ============================================================
def plot_fidelity_colored_by_eps_sum(res: dict):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    c = np.log10(res["eps_bad_sum"] + 1e-15)
    sc = ax.scatter(res["gamma_over_omega_0"], res["F_gate"], c=c, s=12, alpha=0.80)
    ax.set_xlabel(r"$\gamma/\omega_0$")
    ax.set_ylabel(r"Fidelity  $P_{|1⟩,\rm exact}(t_\pi^A)$")
    ax.set_title(r"Optional diagnostic: fidelity coloured by $\log_{10}\epsilon_{\rm bad}^{(\Sigma)}$")
    ax.grid(True, alpha=0.25)
    plt.colorbar(sc, ax=ax)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN DRIVER
# ============================================================
def main():
    
    print("\n====================")
    print("ASHHAB-ONLY DIAGNOSTIC SCRIPT")
    print("Method: exact-resonance enforcement only (method 2)")
    print("Keep only strong-driving branch: gamma/omega0 >= 1.49")
    print("Drop J0 <= 0 branch and use |J1|")
    print("====================")
    
    # ------------------------
    # OMEGA SCAN
    # ------------------------
    res_omega = omega_scan_method2()
    
    print_scan_overview(res_omega, "OMEGA SCAN")
    masks_omega = group_masks(res_omega)
    print_group_counts(res_omega, masks_omega)
    print_group_summary(res_omega, "high", masks_omega["high"], n_rep=10)
    print_group_summary(res_omega, "middle", masks_omega["middle"], n_rep=10)
    print_group_summary(res_omega, "low", masks_omega["low"], n_rep=10)
    
    plot_fidelity_vs_quantity(
        res_omega,
        key="r_tpi_over_Tdrive",
        xlabel=r"$t_\pi^A/T_{\rm drive}$",
        title=r"Omega scan: fidelity vs $t_\pi^A/T_{\rm drive}$",
        logx=True,
    )
    plot_fidelity_vs_quantity(
        res_omega,
        key="t_pi_A",
        xlabel=r"$t_\pi^A$",
        title=r"Omega scan: fidelity vs $t_\pi^A$",
        logx=True,
    )
    plot_fidelity_vs_quantity(
        res_omega,
        key="J0",
        xlabel=r"$J_0(\alpha)$",
        title=r"Omega scan: fidelity vs $J_0(\alpha)$",
        logx=False,
    )
    plot_fidelity_vs_quantity(
        res_omega,
        key="J1_abs",
        xlabel=r"$|J_1(\alpha)|$",
        title=r"Omega scan: fidelity vs $|J_1(\alpha)|$",
        logx=True,
    )
    plot_fidelity_vs_quantity(
        res_omega,
        key="alpha",
        xlabel=r"$\alpha = 2\gamma/\omega$",
        title=r"Omega scan: fidelity vs $\alpha$",
        logx=False,
    )

    plot_filtered_gamma_vs_fidelity(
        res_omega,
        eps_key="eps_bad_max",
        eps_thresh=EPS_THRESH,
        title=rf"Omega scan: fidelity vs $\gamma/\omega_0$ filtered by $\epsilon_{{bad}}^{{(\max)}} < {EPS_THRESH}$",
    )
    plot_filtered_gamma_vs_fidelity(
        res_omega,
        eps_key="eps_bad_sum",
        eps_thresh=EPS_THRESH,
        title=rf"Omega scan: fidelity vs $\gamma/\omega_0$ filtered by $\epsilon_{{bad}}^{{(\Sigma)}} < {EPS_THRESH}$",
    )
    plot_filtered_gamma_vs_fidelity(
        res_omega,
        eps_key="eps_bad_rss",
        eps_thresh=EPS_THRESH,
        title=rf"Omega scan: fidelity vs $\gamma/\omega_0$ filtered by $\epsilon_{{bad}}^{{(\mathrm{{rss}})}} < {EPS_THRESH}$",
    )
    
    print_filtered_fidelity_group_counts(res_omega,eps_key="eps_bad_rss", eps_thresh=0.1,high_thresh=0.85,low_thresh=0.30,)  
    print_filtered_fidelity_group_counts(res_omega, eps_key="eps_bad_sum", eps_thresh=0.1)
    print_filtered_fidelity_group_counts(res_omega, eps_key="eps_bad_max", eps_thresh=0.1)
    
    plot_all_histograms(res_omega)

    # ------------------------
    # ALPHA SCAN
    # ------------------------
    res_alpha = alpha_scan_method2()
    print_scan_overview(res_alpha, "ALPHA SCAN")
    masks_alpha = group_masks(res_alpha)
    print_group_counts(res_alpha, masks_alpha)
    print_group_summary(res_alpha, "high", masks_alpha["high"], n_rep=10)
    print_group_summary(res_alpha, "middle", masks_alpha["middle"], n_rep=10)
    print_group_summary(res_alpha, "low", masks_alpha["low"], n_rep=10)

    plot_filtered_gamma_vs_fidelity(
        res_alpha,
        eps_key="eps_bad_max",
        eps_thresh=EPS_THRESH,
        title=rf"Alpha scan: fidelity vs $\gamma/\omega_0$ filtered by $\epsilon_{{bad}}^{{(\max)}} < {EPS_THRESH}$",
    )
    plot_filtered_gamma_vs_fidelity(
        res_alpha,
        eps_key="eps_bad_sum",
        eps_thresh=EPS_THRESH,
        title=rf"Alpha scan: fidelity vs $\gamma/\omega_0$ filtered by $\epsilon_{{bad}}^{{(\Sigma)}} < {EPS_THRESH}$",
    )
    plot_filtered_gamma_vs_fidelity(
        res_alpha,
        eps_key="eps_bad_rss",
        eps_thresh=EPS_THRESH,
        title=rf"Alpha scan: fidelity vs $\gamma/\omega_0$ filtered by $\epsilon_{{bad}}^{{(\mathrm{{rss}})}} < {EPS_THRESH}$",
    )
    
    # ------------------------
    # TIMING DIAGNOSTICS (omega scan only)
    # ------------------------
    timing = compute_timing_diagnostics(res_omega)
    print_tpi_over_topt_summary_by_fidelity_group(
    res_omega,
    timing,
    high_thresh=0.85,
    low_thresh=0.30,
    )
    
    print_timing_ratio_counts(timing, tol=0.05)   # 5% band
    print_timing_ratio_counts(timing, tol=0.02) # optional stricter 2% band
    print_timing_ratio_counts(timing, tol=0.10) # optional looser 10% band
    plot_log_timing_ratio_histogram(timing, bins=50)
    
    plot_timing_vs_dynamics(timing)
    plot_log_timing_ratio_vs_gamma(timing, y_abs_max=2.0)
    plot_fidelity_vs_tpi_over_topt(timing, x_max_plot=100)
    print_spearman_fidelity_vs_tpi_over_topt(timing)
    # ------------------------
    # Optional extra plot I recommend
    # ------------------------
    #plot_fidelity_colored_by_eps_sum(res_omega)

    print("\nDone.")
    print("If intend to KEEP gamma/omega0 < 1.49 instead, change X_KEEP_MIN logic in derive_point_from_omega().")
    
    print_spearman_fidelity_vs_timing_ratio(res_omega)
    
if __name__ == "__main__":
    main()
    
    
    
