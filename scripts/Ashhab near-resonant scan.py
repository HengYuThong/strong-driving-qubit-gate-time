import numpy as np
import matplotlib.pyplot as plt

from numpy import pi
from scipy.linalg import expm
from scipy.special import jv


# ============================================================
# Global constants and basic matrices
# ============================================================
hbar = 1.0

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

ket0 = np.array([[1.0], [0.0]], dtype=complex)
ket1 = np.array([[0.0], [1.0]], dtype=complex)
bra1 = ket1.conj().T


def p1_of_state(psi: np.ndarray) -> float:
    """P(|1>) = |<1|psi>|^2."""
    amp = (bra1 @ psi).item()
    return float(np.abs(amp) ** 2)


# ============================================================
# Midpoint stepping utilities
# ============================================================
def U_step_from_H(H: np.ndarray, dt: float) -> np.ndarray:
    return expm(-1j * H * dt / hbar)


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


def evolve_final_state_to_time(H_func, t_final: float, psi0: np.ndarray, N_steps: int = 4001) -> np.ndarray:
    if t_final <= 0:
        return psi0.copy()
    times_loc = np.linspace(0.0, float(t_final), int(N_steps))
    hist = evolve_state_history_on_grid(H_func, times_loc, psi0)
    return hist[-1]


# ============================================================
# Exact lab-frame Hamiltonian
# H(t) = (ħ ω0/2) σz + ħ γ cos(ω t) σx
# ============================================================
def H_lab_param(t: float, omega: float, omega_0: float, gamma: float) -> np.ndarray:
    return 0.5 * hbar * omega_0 * sigma_z + hbar * gamma * np.cos(omega * t) * sigma_x


# ============================================================
# Ashhab helper quantities
# ============================================================
def alpha_of(omega: float, gamma: float) -> float:
    return 2.0 * gamma / omega


def t_pi_ashhab(m: int, omega: float, omega_0: float, gamma: float) -> float:
    """
    Ashhab on-resonance estimate:
        Ω_R = ω0 |J_m(alpha)|,  t_pi = π / Ω_R
    """
    a = alpha_of(omega, gamma)
    Jm = jv(m, a)
    Omega_R = omega_0 * abs(Jm)
    if Omega_R < 1e-15:
        return np.inf
    return np.pi / Omega_R


# ============================================================
# Method 1 resonance search:
# fixed omega0, fixed gamma, sweep omega and odd m
# choose best near-resonant candidate
# ============================================================
def resonance_sweep_find_best(
    omega_0: float,
    gamma: float,
    omega_grid: np.ndarray,
    m_max: int = 31,
    eps: float = 1e-12,
    delta_cut: float = None,
):
    best_abs = None
    best_score = None
    near_resonant = []

    for w in omega_grid:
        if w <= 0:
            continue

        a = alpha_of(w, gamma)
        J0 = jv(0, a)
        Om = omega_0 * J0

        for m in range(1, m_max + 1, 2):  # odd m only
            Jm = jv(m, a)
            delta = m * w - Om
            score = abs(Jm) / (abs(delta) + eps)

            cand = {
                "omega": float(w),
                "m": int(m),
                "delta": float(delta),
                "abs_delta": float(abs(delta)),
                "score": float(score),
                "alpha": float(a),
                "J0": float(J0),
                "Omega_phys": float(Om),
                "Jm": float(Jm),
            }

            if (best_abs is None) or (cand["abs_delta"] < best_abs["abs_delta"]):
                best_abs = cand

            if (best_score is None) or (cand["score"] > best_score["score"]):
                best_score = cand

            if delta_cut is not None and cand["abs_delta"] <= delta_cut:
                near_resonant.append(cand)

    if delta_cut is not None and len(near_resonant) > 0:
        best_choice = max(near_resonant, key=lambda d: d["score"])
    else:
        best_choice = best_abs

    return best_abs, best_score, best_choice


def refine_best_local(
    omega_0: float,
    gamma: float,
    coarse_best: dict,
    m_max: int,
    window_frac: float = 0.01,
    N_refine: int = 8001,
    delta_cut: float = None,
):
    w0 = coarse_best["omega"]
    wL = max(1e-9, (1.0 - window_frac) * w0)
    wR = (1.0 + window_frac) * w0
    omega_local = np.linspace(wL, wR, N_refine)

    best_abs, best_score, best_choice = resonance_sweep_find_best(
        omega_0=omega_0,
        gamma=gamma,
        omega_grid=omega_local,
        m_max=m_max,
        delta_cut=delta_cut,
    )
    return best_abs, best_score, best_choice


# ============================================================
# Exact fidelity at chosen t_eval
# ============================================================
def gate_fidelity_exact_for_params(
    omega_use: float,
    omega_0: float,
    gamma_use: float,
    t_eval: float,
    psi0=ket0,
    N_steps: int = 4001,
) -> float:
    H_ex = lambda t: H_lab_param(t, omega_use, omega_0, gamma_use)
    psi = evolve_final_state_to_time(H_ex, t_eval, psi0, N_steps=N_steps)
    return p1_of_state(psi)


# ============================================================
# Method 1 gamma sweep, Ashhab only
# This follows the draft-3 logic behind the original plot:
# for each gamma, find best (omega, m) by resonance search
# then evaluate exact fidelity at t_pi^A
# ============================================================
def gamma_sweep_method1_ashhab_only(
    omega_0: float,
    gamma_grid: np.ndarray,
    omega_grid_coarse: np.ndarray,
    m_max: int = 31,
    delta_cut: float = None,
    N_steps: int = 4001,
    window_frac: float = 0.01,
    N_refine: int = 8001,
):
    out = {
        "gamma_grid": [],
        "gamma_over_omega0": [],
        "F_A": [],
        "omega_best": [],
        "omega_best_over_omega0": [],
        "m_best": [],
        "abs_delta": [],
        "delta": [],
        "alpha_best": [],
        "J0_best": [],
        "Jm_best": [],
        "t_pi_A": [],
        "omega0_fixed": [],
    }

    for gam in gamma_grid:
        # coarse sweep
        best_abs_c, best_score_c, best_c = resonance_sweep_find_best(
            omega_0=omega_0,
            gamma=gam,
            omega_grid=omega_grid_coarse,
            m_max=m_max,
            delta_cut=delta_cut,
        )

        # refine locally around coarse best
        best_abs_f, best_score_f, best_f = refine_best_local(
            omega_0=omega_0,
            gamma=gam,
            coarse_best=best_c,
            m_max=m_max,
            window_frac=window_frac,
            N_refine=N_refine,
            delta_cut=delta_cut,
        )

        best = best_f
        omega_best = best["omega"]
        m_best = best["m"]
        delta = best["delta"]
        abs_delta = abs(delta)
        alpha_best = best["alpha"]
        J0_best = best["J0"]
        Jm_best = best["Jm"]

        tA = t_pi_ashhab(m_best, omega_best, omega_0, gam)
        FA = gate_fidelity_exact_for_params(
            omega_use=omega_best,
            omega_0=omega_0,
            gamma_use=gam,
            t_eval=tA,
            N_steps=N_steps,
        )

        out["gamma_grid"].append(float(gam))
        out["gamma_over_omega0"].append(float(gam / omega_0))
        out["F_A"].append(float(FA))
        out["omega_best"].append(float(omega_best))
        out["omega_best_over_omega0"].append(float(omega_best / omega_0))
        out["m_best"].append(int(m_best))
        out["abs_delta"].append(float(abs_delta))
        out["delta"].append(float(delta))
        out["alpha_best"].append(float(alpha_best))
        out["J0_best"].append(float(J0_best))
        out["Jm_best"].append(float(Jm_best))
        out["t_pi_A"].append(float(tA))
        out["omega0_fixed"].append(float(omega_0))

    for k in out:
        out[k] = np.array(out[k], dtype=float)

    return out

def weak_rwa_fidelity_sweep_same_omega0(
    omega_0: float,
    gamma_grid: np.ndarray,
    N_steps: int = 4001,
):
    """
    Weak-RWA baseline on its own resonance:
        omega = omega0,  t_pi^RWA = pi/gamma
    Fidelity is computed from the exact lab evolution at that RWA pulse time.
    """
    F_rwa = []

    for gam in gamma_grid:
        t_pi_rwa = np.pi / abs(gam)
        H_ex = lambda t, g=gam: H_lab_param(t, omega_0, omega_0, g)
        psi = evolve_final_state_to_time(H_ex, t_pi_rwa, ket0, N_steps=N_steps)
        F_rwa.append(p1_of_state(psi))

    return np.array(F_rwa, dtype=float)

# ============================================================
# Plot 1:
# 3 panels: fidelity / J0(alpha) / omega0
# Note: omega0 is fixed in method 1, so bottom panel is constant.
# ============================================================
def plot_three_panel_method1(results: dict):
    x = results["gamma_over_omega0"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # top: fidelity
    axes[0].plot(x, results["F_A"], linewidth=2)
    axes[0].set_ylabel(r"Fidelity  $P_{|1⟩,\rm exact}(t_\pi^A)$")
    axes[0].set_title(r"Fidelity scan, $J_0(\alpha)$, and chosen $\omega_{\rm best}/\omega_0$")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.3)

    # middle: J0(alpha)
    axes[1].plot(x, results["J0_best"], linewidth=2)
    axes[1].set_ylabel(r"$J_0(\alpha)$")
    axes[1].grid(True, alpha=0.3)

    # bottom: omega_best / omega0
    axes[2].plot(x, results["omega_best_over_omega0"], linewidth=2)
    axes[2].set_ylabel(r"$\omega_{\rm best}/\omega_0$")
    axes[2].set_xlabel(r"$\gamma/\omega_0$")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# Plot 2:
# one plane with fidelity only
# ============================================================
def plot_fidelity_only_method1(results: dict, F_rwa: np.ndarray = None):
    x = results["gamma_over_omega0"]

    plt.figure(figsize=(8, 5))
    plt.plot(x, results["F_A"], linewidth=2, color="tab:blue",
             label="Ashhab (method 1)")

    if F_rwa is not None:
        plt.plot(x, F_rwa, linewidth=2, color="tab:orange",
                 label=r"Weak-RWA: $\omega=\omega_0,\ t_\pi=\pi/\gamma$")

    plt.xlabel(r"$\gamma/\omega_0$")
    plt.ylabel(r"Gate fidelity  $P_{1,\rm exact}(t_\pi)$")
    plt.title(r"Fidelity scan over drive strength")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Plot 3:
# top: fidelity
# bottom: m_best chosen from method 1
# ============================================================
def plot_fidelity_and_mbest_method1(results: dict, F_rwa: np.ndarray = None):
    x = results["gamma_over_omega0"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Top panel
    axes[0].plot(
        x, results["F_A"],
        linewidth=2,
        color="tab:blue",
        label="Ashhab-inspired RWA"
    )

    if F_rwa is not None:
        axes[0].plot(
            x, F_rwa,
            linewidth=2,
            color="tab:orange",
            label=r"Weak-RWA: $\omega=\omega_0,\ t_\pi=\pi/\gamma$"
        )

    axes[0].set_ylabel(r"Fidelity  $P_{|1\rangle,\rm exact}(t_\pi)$")
    axes[0].set_title(r"Fidelity scan and selected $m_{\rm best}$")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Bottom panel
    axes[1].scatter(x, results["m_best"], s=18)
    axes[1].set_ylabel(r"$m_{\rm best}$")
    axes[1].set_xlabel(r"$\gamma/\omega_0$")
    axes[1].set_yticks(np.arange(1, int(np.max(results["m_best"])) + 1, 2))
    axes[1].grid(True, alpha=0.3)
    
    plt.xlim(0,10)
    plt.tight_layout()
    plt.show()


# ============================================================
# Plot 4:
# top: fidelity
# middle: J1(alpha)
# bottom: t_pi^A
# ============================================================    
def plot_fidelity_J1_tpi_method1(results: dict):
    x = results["gamma_over_omega0"]
    J1_best = jv(1, results["alpha_best"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # top: fidelity
    axes[0].plot(x, results["F_A"], linewidth=2)
    axes[0].set_ylabel(r"Fidelity  $P_{|1⟩,\rm exact}(t_\pi^A)$")
    axes[0].set_title(r"Fidelity scan, $J_1(\alpha)$, and $t_\pi^A$")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.3)

    # middle: J1(alpha)
    axes[1].plot(x, J1_best, linewidth=2)
    axes[1].set_ylabel(r"$J_1(\alpha)$")
    axes[1].grid(True, alpha=0.3)

    # bottom: t_pi^A
    axes[2].plot(x, results["t_pi_A"], linewidth=2)
    axes[2].set_ylabel(r"$t_\pi^A$")
    axes[2].set_xlabel(r"$\gamma/\omega_0$")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# Exact same method-1 style settings as the draft-3 gamma sweep
# ============================================================
if __name__ == "__main__":
    omega_0 = 2 * pi * 1.0
    m_max = 31
    delta_cut = 0.02 * omega_0

    # same draft-3 gamma sweep range
    gamma_grid = np.linspace(0.1 * omega_0, 10.0 * omega_0, 200)
    gamma_grid = np.sort(np.unique(np.r_[gamma_grid, 2.0 * omega_0]))

    # to match the actual draft-3 behavior more closely,
    # use the coarse omega grid that the original script had globally
    omega_grid_coarse = np.linspace(0.05 * omega_0, 2.0 * omega_0, 5000)

    results = gamma_sweep_method1_ashhab_only(
        omega_0=omega_0,
        gamma_grid=gamma_grid,
        omega_grid_coarse=omega_grid_coarse,
        m_max=m_max,
        delta_cut=delta_cut,
        N_steps=4001,
        window_frac=0.01,
        N_refine=8001,
    )
    
    F_rwa = weak_rwa_fidelity_sweep_same_omega0(
    omega_0=omega_0,
    gamma_grid=gamma_grid,
    N_steps=4001,
)

    #1) 3-panel: fidelity / J0(alpha) / omega0
    plot_three_panel_method1(results)

    #2) one-plane fidelity only
    plot_fidelity_only_method1(results)

    #3) 2-panel: fidelity / m_best
    plot_fidelity_and_mbest_method1(results, F_rwa = F_rwa)
    
    #4) 3-panel: fidelity / J1(alpha) / t_pi^A
    plot_fidelity_J1_tpi_method1(results)
