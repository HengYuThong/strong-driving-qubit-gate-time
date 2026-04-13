import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import expm
from scipy.special import jv  # Bessel J_n


# ============================================================
# Global constants and helpers
# ============================================================
hbar = 1.0  # set ħ = 1

def dagger(A: np.ndarray) -> np.ndarray:
    """Hermitian conjugate."""
    return A.conj().T

def mat_norm_inf(A: np.ndarray) -> float:
    """Infinity norm: max absolute matrix element."""
    return float(np.max(np.abs(A)))

# ============================================================
# Step 1: Define basic matrices (2-level system)
# ============================================================
I2 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

ket0 = np.array([[1.0], [0.0]], dtype=complex)
ket1 = np.array([[0.0], [1.0]], dtype=complex)
bra0 = dagger(ket0)
bra1 = dagger(ket1)

sigma_p = ket0 @ bra1  # |0><1|
sigma_m = ket1 @ bra0  # |1><0|

def p1_of_state(psi: np.ndarray) -> float: # Also to calculate gate fidelity
    """P(|1>) = |<1|psi>|^2."""
    amp = (bra1 @ psi).item()
    return float(np.abs(amp) ** 2)

# ============================================================
# Step 2: Midpoint stepping utilities
# ============================================================
def U_step_from_H(H: np.ndarray, dt: float) -> np.ndarray:
    """One step unitary: exp(-i H dt / ħ)."""
    return expm(-1j * H * dt / hbar)

def evolve_state_history_on_grid(H_func, times: np.ndarray, psi0: np.ndarray) -> np.ndarray:
    """
    Evolve the state and return the full state history.
    Output shape: (len(times), 2, 1).
    """
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
    """
    Return psi(t_final) by midpoint stepping with N_steps points from 0..t_final.
    """
    if t_final <= 0:
        return psi0.copy()
    times_loc = np.linspace(0.0, float(t_final), int(N_steps))
    hist = evolve_state_history_on_grid(H_func, times_loc, psi0)
    return hist[-1]

# ============================================================
# Step 3: Exact LAB Hamiltonian (parametric)
# H(t) = (ħ ω0/2) σz + ħ γ cos(ω t) σx
# ============================================================
def H_lab_param(t: float, omega: float, omega_0: float, gamma: float) -> np.ndarray:
    return 0.5 * hbar * omega_0 * sigma_z + hbar * gamma * np.cos(omega * t) * sigma_x

# ============================================================
# Step 4: Ashhab objects (alpha, dressed splitting, kick/rot frames)
# We use Ω_phys = ω0 * J0(alpha) 
# ============================================================
def alpha_of(omega: float, gamma: float) -> float:
    return 2.0 * gamma / omega

def Omega_phys(omega: float, omega_0: float, gamma: float) -> float:
    a = alpha_of(omega, gamma)
    return omega_0 *jv(0, a)

def V_kick_param(t: float, omega: float, gamma: float) -> np.ndarray:
    """
    V(t) = exp[- i/2 * alpha * sin(ωt) σx], alpha = 2γ/ω
    """
    a = alpha_of(omega, gamma)
    f = a * np.sin(omega * t)
    return expm(-0.5j * f * sigma_x)

def R_rot(t: float, Omega_val: float) -> np.ndarray:
    """R(t) = exp(-i Ω t σz / 2)."""
    return expm(-0.5j * Omega_val * t * sigma_z)

# ============================================================
# Step 5: Ashhab effective Hamiltonian (single odd harmonic m)
# H_eff(t) = (ħ ω0 J_m(alpha)/2) [ σ+ e^{+i δ t} + σ- e^{-i δ t} ]
# δ = mω - Ω_phys, Ω_phys = ω0J0(alpha)
# ============================================================
def H_eff_single_param(t: float, m: int, omega: float, omega_0: float, gamma: float) -> np.ndarray:
    a = alpha_of(omega, gamma)
    Jm = jv(m, a)
    Om = Omega_phys(omega, omega_0, gamma)
    delta = m * omega - Om
    return 0.5 * hbar * omega_0 * Jm * (sigma_p * np.exp(-1j * delta * t) + sigma_m * np.exp(1j * delta * t))

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

# --------------------------------------------------------
# Weak-RWA predicted gate times (two benchmarking options)
# --------------------------------------------------------
def t_pi_weak_detuned(omega: float, omega_0: float, gamma: float) -> float:
    Delta = omega_0 - omega
    Omega_gen = float(np.sqrt(Delta * Delta + gamma * gamma))
    if Omega_gen < 1e-15:
        return np.inf
    return np.pi / Omega_gen

# ============================================================
# Step 6: Resonance search (fixed gamma, m, omega):  δ = 0
# compute:
#   omega_0 = omega/J_0(alpha)
# ============================================================
def method2_use_same_omega_solve_omega0_and_fidelity(
    omega_from_method1: float,
    gamma: float,
    m: int = 1,
    J0_min: float = 1e-6,
    N_steps: int = 4001,
):
    """
    Method 2:
      - fix gamma and m
      - use the supplied omega
      - solve omega0 from delta = m*omega - omega0*J0(alpha) = 0
      - keep only positive-J0 branch
      - compute t_pi^A = pi / (omega0*|Jm(alpha)|)
      - evaluate exact LAB fidelity at t_pi^A
    """
    omega = float(omega_from_method1)
    if omega <= 0:
        return None

    alpha = 2.0 * gamma / omega
    J0 = jv(0, alpha)

    # keep only positive-J0 branch
    if J0 <= J0_min:
        return None

    omega0_solved = (m * omega) / J0

    Jm = jv(m, alpha)
    if abs(Jm) < 1e-15:
        return None

    t_pi_A = np.pi / (omega0_solved * abs(Jm))

    H_ex = lambda t: H_lab_param(t, omega, omega0_solved, gamma)
    psi = evolve_final_state_to_time(H_ex, t_pi_A, ket0, N_steps=N_steps)
    F_gate = p1_of_state(psi)

    delta_check = m * omega - omega0_solved * J0

    return {
        "omega": omega,
        "gamma": float(gamma),
        "m": int(m),
        "alpha": float(alpha),
        "J0": float(J0),
        "omega0_solved": float(omega0_solved),
        "Jm": float(Jm),
        "t_pi_A": float(t_pi_A),
        "F_gate": float(F_gate),
        "delta_check": float(delta_check),
    }


# ============================================================
# Step 7: Frame-corrected comparison
# Exact LAB: |psi_exact,lab(t)>
# Effective frame: |psi_eff(t)> evolved under H_eff
# Map back to LAB:
#   |psi_eff,lab(t)> = V†(t) R†(t) |psi_eff(t)>
# ============================================================
def map_eff_to_lab_state(
    t: float,
    psi_eff: np.ndarray,
    omega: float,
    omega_0: float,
    gamma: float,
) -> np.ndarray:
    Om = Omega_phys(omega, omega_0, gamma)
    V = V_kick_param(t, omega, gamma)
    R = R_rot(t, Om)
    return V @ R @ psi_eff
 
# ============================================================
# Step 8b : Omega-sweep plots (gate fidelity against gamma) (Resonance 2)
# ============================================================
def resonance_scan_solve_omega0_and_fidelity(
    gamma: float,
    omega_scan: np.ndarray,
    m: int = 1,
    J0_min: float = 1e-3,
    J1_min: float = 1e-6,
    N_steps: int = 4001,
):
    """
    Fix (gamma, m). Scan omega.
    For each omega, enforce delta=0 by solving
        omega0 = m*omega / J0(alpha),
    with alpha = 2gamma/omega,
    keeping only the positive-J0 branch.
    """
    out = {
        "omega": [],
        "alpha": [],
        "J0": [],
        "omega0": [],
        "gamma_over_omega_0": [],
        "omega_over_omega_0": [],
        "t_pi_A": [],
        "F_gate": [],
        "J1_abs": [],
        "J3_abs": [],
        "J5_abs": [],
        "eps_fast": [],
        "eps3": [],
        "eps5": [],
        "eps_bad":[],
        "eps_bad_max": [],
        "eps_bad_sum": [],
        "eps_bad_rss": [],
        "T_drive": [],
        "r_tpi_over_Tdrive": [],
    }

    for omega in omega_scan:
        if omega <= 0:
            continue

        alpha = 2.0 * gamma / omega
        J0 = jv(0, alpha)

        # keep only positive-J0 branch
        if J0 <= J0_min:
            continue

        omega0 = (m * omega) / J0

        Jm = jv(m, alpha)
        if abs(Jm) < J1_min:
            continue

        t_pi_A = np.pi / (omega0 * abs(Jm))

        H_ex = lambda t: H_lab_param(t, omega, omega0, gamma)
        psi = evolve_final_state_to_time(H_ex, t_pi_A, ket0, N_steps=N_steps)
        F = p1_of_state(psi)

        J1_abs = abs(jv(1, alpha))
        J3_abs = abs(jv(3, alpha))
        J5_abs = abs(jv(5, alpha))

        eps_fast = omega0 * abs(Jm) / (2.0 * m * omega + 1e-12)
        eps3 = omega0 * J3_abs / (2.0 * omega + 1e-12)
        eps5 = omega0 * J5_abs / (4.0 * omega + 1e-12)

        eps_bad_max = max(eps_fast, eps3, eps5)
        eps_bad_sum = eps_fast + eps3 + eps5
        eps_bad_rss = np.sqrt(eps_fast**2 + eps3**2 + eps5**2)
        eps_bad = eps_bad_max

        T_drive = 2.0 * np.pi / omega
        r_tpi_over_Tdrive = t_pi_A / T_drive

        out["omega"].append(float(omega))
        out["alpha"].append(float(alpha))
        out["J0"].append(float(J0))
        out["omega0"].append(float(omega0))
        out["gamma_over_omega_0"].append(float(gamma / omega0))
        out["omega_over_omega_0"].append(float(omega / omega0))
        out["t_pi_A"].append(float(t_pi_A))
        out["F_gate"].append(float(F))
        out["J1_abs"].append(float(J1_abs))
        out["J3_abs"].append(float(J3_abs))
        out["J5_abs"].append(float(J5_abs))
        out["eps_fast"].append(float(eps_fast))
        out["eps3"].append(float(eps3))
        out["eps5"].append(float(eps5))
        out["eps_bad"].append(float(eps_bad))
        out["eps_bad_max"].append(float(eps_bad_max))
        out["eps_bad_sum"].append(float(eps_bad_sum))
        out["eps_bad_rss"].append(float(eps_bad_rss))
        out["T_drive"].append(float(T_drive))
        out["r_tpi_over_Tdrive"].append(float(r_tpi_over_Tdrive))

    for k in out:
        out[k] = np.array(out[k], dtype=float)

    return out


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

        
    # ============================================================
    # SINGLE-RUN DRIVER (flexible)
    #   You can choose what to fix / scan / solve:
    #     omega0_mode = "fixed" OR "solve_from_delta0"
    #     omega_mode  = "manual" OR "scan"
    #
    # Notes:
    #  - If omega0_mode="fixed" and omega_mode="scan":
    #       we scan omega to MINIMIZE |delta| for the chosen m_run
    #  - If omega0_mode="solve_from_delta0" and omega_mode="manual":
    #       we solve omega0 for your chosen omega so delta=0 exactly
    #  - If omega0_mode="solve_from_delta0" and omega_mode="scan":
    #       we scan omega, solve omega0 at each omega (delta=0), then PICK 1 point
    # ============================================================

    DO_SINGLE_RUN = True
    
    if DO_SINGLE_RUN:
    
        # -----------------------------
        # USER SETTINGS (edit these)
        # -----------------------------
        gamma_run = 5.03133 * 2.49762
        m_run = 1
    
        omega0_mode = "fixed"   # "fixed" or "solve_from_delta0"
        omega_mode  = "manual"              # "manual" or "scan"
    
        # Manual values
        omega0_fixed = 5.03133            # only used if omega0_mode="fixed"
        omega_manual = 0.363276            # used if omega_mode="manual"
    
        # Scan grid for omega (used if omega_mode="scan")
        # IMPORTANT: if omega0_mode="solve_from_delta0", omega0_fixed is just a convenient scale
        omega_scan_ref = omega0_fixed
        omega_scan_grid = np.linspace(0.05 * omega_scan_ref, 2.0 * omega_scan_ref, 8000)
    
        # If you do (solve_from_delta0 + scan), you must choose how to pick ONE point from the scan
        scan_pick = "max_F"     # "max_F" or "min_tpi" or "closest_x"
        x_target = 2.0          # only used if scan_pick="closest_x"
    
        # Filtering for method-2 scan (avoid blow-ups near J0~0 or Jm~0)
        J0_min = 1e-3
        Jm_min = 1e-6
    
        # Numerics
        N_steps_gate = 4001
        N_pts_series = 5001
    
        # -----------------------------
        # PICK (omega_run, omega0_run)
        # -----------------------------
        if omega0_mode == "solve_from_delta0":
    
            if omega_mode == "manual":
                # omega chosen manually, omega0 solved so delta=0 exactly
                omega_run = float(omega_manual)
    
                tmp = method2_use_same_omega_solve_omega0_and_fidelity(
                    omega_from_method1=omega_run,
                    gamma=gamma_run,
                    m=m_run,
                    N_steps=N_steps_gate,
                )
                if tmp is None:
                    raise RuntimeError("Cannot solve omega0 (J0 too close to 0 or omega<=0). Try different omega.")
    
                omega0_run = float(tmp["omega0_solved"])
                alpha_run  = float(tmp["alpha"])
                J0_run     = float(tmp["J0"])
                Jm_run     = float(tmp["Jm"])
                delta_run  = float(m_run * omega_run - omega0_run * J0_run)
    
            elif omega_mode == "scan":
                # scan omega, solve omega0 at each omega (delta=0), then pick ONE point
                scan = resonance_scan_solve_omega0_and_fidelity(
                    gamma=gamma_run,
                    omega_scan=omega_scan_grid,
                    m=m_run,
                    J0_min=J0_min,
                    J1_min=Jm_min,      # reuse the name (it is Jm_min when m_run=1)
                    N_steps=N_steps_gate,
                )
                if len(scan["omega"]) == 0:
                    raise RuntimeError("Scan returned no valid points. Try widening omega_scan_grid or lowering J0_min/Jm_min.")
    
                x_scan   = scan["gamma_over_omega_0"]
                F_scan   = scan["F_gate"]
                tpi_scan = scan["t_pi_A"]
    
                if scan_pick == "max_F":
                    i = int(np.nanargmax(F_scan))
                elif scan_pick == "min_tpi":
                    i = int(np.nanargmin(tpi_scan))
                elif scan_pick == "closest_x":
                    i = int(np.nanargmin(np.abs(x_scan - x_target)))
                else:
                    raise ValueError("scan_pick must be 'max_F', 'min_tpi', or 'closest_x'.")
    
                omega_run = float(scan["omega"][i])
                omega0_run = float(scan["omega0"][i])
                alpha_run = float(scan["alpha"][i])
                J0_run = float(scan["J0"][i])
                Jm_run = float(jv(m_run, alpha_run))
                delta_run = float(m_run * omega_run - omega0_run * J0_run)  # should be ~0
    
            else:
                raise ValueError("omega_mode must be 'manual' or 'scan'.")
    
        elif omega0_mode == "fixed":
    
            omega0_run = float(omega0_fixed)
    
            if omega_mode == "manual":
                omega_run = float(omega_manual)
    
            elif omega_mode == "scan":
                # scan omega to minimize |delta| for chosen m_run (near-resonant)
                use_abs_J0 = True
                best = None
                for w in omega_scan_grid:
                    if w <= 0:
                        continue
                    a = alpha_of(w, gamma_run)
                    J0 = jv(0, a)
                    J0_used = abs(J0) if use_abs_J0 else J0
                    delta = m_run * w - omega0_run * J0_used
    
                    cand = (abs(delta), float(w), float(a), float(J0))
                    if (best is None) or (cand[0] < best[0]):
                        best = cand
    
                abs_delta_min, omega_run, alpha_run, J0_run = best
                J0_used = abs(J0_run) if use_abs_J0 else J0_run
                delta_run = float(m_run * omega_run - omega0_run * J0_used)
    
            else:
                raise ValueError("omega_mode must be 'manual' or 'scan'.")
    
            # compute these for printing
            alpha_run = float(alpha_of(omega_run, gamma_run))
            J0_run = float(jv(0, alpha_run))
            Jm_run = float(jv(m_run, alpha_run))
            delta_run = float(m_run * omega_run - omega0_run * J0_run)
    
        else:
            raise ValueError("omega0_mode must be 'fixed' or 'solve_from_delta0'.")
    
        # -----------------------------
        # PRINT KEY INFO
        # -----------------------------
        J1_run = float(jv(1, alpha_run))
        Omega_used_run = float(omega0_run * J0_run)
        t_pi_A = float(t_pi_ashhab(m_run, omega_run, omega0_run, gamma_run))
        t_pi_RWA = float(np.pi / abs(gamma_run))  # your RWA definition
    
        H_exact = lambda t: H_lab_param(t, omega_run, omega0_run, gamma_run)
        psi_exact_tpiA = evolve_final_state_to_time(H_exact, t_pi_A, ket0, N_steps=N_steps_gate)
        F_gate_A = p1_of_state(psi_exact_tpiA)
    
        H_exact_RWA = lambda t: H_lab_param(t, omega0_run, omega0_run, gamma_run)
        psi_exact_tpiRWA = evolve_final_state_to_time(H_exact_RWA, t_pi_RWA, ket0, N_steps=N_steps_gate)
        F_gate_RWA = p1_of_state(psi_exact_tpiRWA)
    
        print("\n=== SINGLE RUN (flexible driver) ===")
        print(f"omega0_mode={omega0_mode}, omega_mode={omega_mode}")
        if omega_mode == "scan" and omega0_mode == "solve_from_delta0":
            print(f"scan_pick={scan_pick}" + (f", x_target={x_target}" if scan_pick=="closest_x" else ""))
    
        print(f"gamma = {gamma_run:.6g}")
        print(f"m = {m_run}")
        print(f"omega_run = {omega_run:.6g}")
        print(f"omega0_run = {omega0_run:.6g}")
        print(f"alpha = 2gamma/omega = {alpha_run:.6g}")
        print(f"J0(alpha) = {J0_run:.6g}   |J0|={abs(J0_run):.6g}")
        print(f"J1(alpha) = {J1_run:.6g}   |J1|={abs(J1_run):.6g}")
        print(f"Jm(alpha) = {Jm_run:.6g}   |Jm|={abs(Jm_run):.6g}")
        print(f"Omega_used = omega0 * J0 = {Omega_used_run:.6g}")
        print(f"delta = m*omega - Omega_used = {delta_run:.6e}   (0 means resonance)")
        print(f"t_pi^A = {t_pi_A:.6g}   F_gate^A = {100*F_gate_A:.2f}%")
        print(f"t_pi^RWA (omega=omega0) = {t_pi_RWA:.6g}   "
              f"F_gate^RWA (exact, omega=omega0) = {100*F_gate_RWA:.2f}%")
    
        # -----------------------------
        # TIME-SERIES (Exact vs Ashhab)
        # -----------------------------
        T_drive = 2.0 * np.pi / omega_run
        T_total = max(3.0 * t_pi_A, 3.0 * t_pi_RWA, 6.0 * T_drive)
        times = np.linspace(0.0, T_total, N_pts_series)
    
        hist_exact = evolve_state_history_on_grid(H_exact, times, ket0)
        P1_exact = np.array([p1_of_state(hist_exact[k]) for k in range(len(times))])
    
        H_eff = lambda t: H_eff_single_param(t, m_run, omega_run, omega0_run, gamma_run)
        hist_eff = evolve_state_history_on_grid(H_eff, times, ket0)
    
        P1_eff_lab = np.empty(len(times), dtype=float)
        for k, t in enumerate(times):
            psi_lab = map_eff_to_lab_state(t, hist_eff[k], omega_run, omega0_run, gamma_run)
            P1_eff_lab[k] = p1_of_state(psi_lab)
    
        P1_eff_frame = np.array([p1_of_state(hist_eff[k]) for k in range(len(times))])
    
        # -----------------------------
        # PLOTS
        # -----------------------------
        fig, ax = plt.subplots()
        ax.plot(times, P1_exact,color="tab:orange",label="exact dynamics", linewidth=2)
        ax.plot(times, P1_eff_lab,color="tab:blue",label="Ashhab-inspired RWA", linewidth=2)
        ax.axvline(t_pi_A, color="tab:red", linestyle="--", linewidth=2, label=r"$t_\pi^A$")
        ax.axvline(t_pi_RWA, color="black", linestyle=":", linewidth=2, label=r"$t_\pi^{\rm RWA}$ (ω = $ω_0$)")
        ax.set_xlabel("t")
        ax.set_ylabel("P(|1⟩)")
        ax.set_title("Lab-frame comparison: Exact vs Ashhab (frame-corrected)")
        ax.legend()
        plt.tight_layout()
        plt.xlim(100,120)
    
        fig, ax = plt.subplots()
        ax.plot(times, P1_exact,color="tab:orange",label="exact dynamics", linewidth=2)
        ax.plot(times, P1_eff_frame,color="tab:blue",label="Ashhab-inspired RWA in effective frame", linewidth=2)
        ax.axvline(t_pi_A, color="tab:red", linestyle="--", linewidth=2, label=r"$t_\pi^A$")
        ax.axvline(t_pi_RWA, color="black", linestyle=":", linewidth=2, label=r"$t_\pi^{\rm RWA}$ (omega=omega0)")
        ax.set_xlabel("t")
        ax.set_ylabel("P(|1⟩)")
        ax.set_title("Unfair comparison: Exact vs Ashhab (in different frame)")
        ax.legend()
        plt.tight_layout()
        plt.xlim(0,3)
    
        plt.show()
        
       

    
    


    
