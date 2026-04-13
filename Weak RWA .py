import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# -----------------------------
# Global constants and helpers
# -----------------------------
ℏ = 1.0 #set ℏ = 1
def dagger(A:np.ndarray)->np.ndarray:  #to return dagger
    return A.conj().T  #complex conjugate + transpose

# -----------------------------
# Stimulation parameters 
# -----------------------------
omega_0 = 2 * pi * 1     # higher freq, lower error & higher fidelity 
omega_d = omega_0         # choose exact resonance
gamma   = 0.2 * omega_0   # drive amplitude γ #lower gamma, lower error & higher fidelity
T_total = 2 * np.pi/gamma  # total simulation time
N_pts   = 4001             # number of sample points
times   = np.linspace(0.0, T_total, N_pts)   #linearly sapced array of actual time points

# -----------------------------
# Step 1 define basic matrices 
# -----------------------------
I2 = np.eye(2,dtype=complex) 
sigma_x = np.array([[0,1],[1,0]], dtype=complex)
sigma_y = np.array([[0,-1j],[1j,0]], dtype=complex)
sigma_z = np.array([[1,0],[0,-1]],dtype=complex)

ket0 = np.array([[1.0],[0.0]], dtype=complex)
ket1 = np.array([[0.0],[1.0]], dtype=complex)
bra0 = dagger(ket0)
bra1 = dagger(ket1)

sigma_p = ket0 @ bra1 #|0><1|
sigma_m = ket1 @ bra0 #|1><0|


# -----------------------------
# Step 2 define H0 (lab frame) 
# -----------------------------
H0 = 1/2 * ℏ * omega_0 * sigma_z

# -----------------------------
# Step 3 define U0(t) and U0^t(t)
# -----------------------------
def U0(t:float) -> np.ndarray:
    first_row = np.exp(1 * (-0.5j * omega_0 * t))
    second_row = np.exp(-1 * (-0.5j * omega_0 * t))
    return np.array([[first_row, 0.0],[0.0, second_row]], dtype=complex)

def U0_dag(t:float) -> np.ndarray:
    return dagger(U0(t))

# -----------------------------
# Step 4 define σ_±^I(t), σ_z^I(t)   #Actually not needed here
# -----------------------------
def sigma_p_I(t:float)-> np.ndarray:
    return np.exp(1j * omega_0 * t) * sigma_p

def sigma_m_I(t:float)-> np.ndarray:
    return np.exp(-1j * omega_0 * t) * sigma_m

def sigma_z_I(t:float)-> np.ndarray:
    return sigma_z

# -----------------------------
# Step 5 define H_drive in lab
# -----------------------------
def H_drive_ii(t:float)-> np.ndarray:
    return ℏ * gamma * np.cos(omega_d * t) * (sigma_p + sigma_m)

# -----------------------------
# Step 6 define H_drive in IP (without RWA)
# -----------------------------
def H_drive_ii_IP(t:float)->np.ndarray:
    return U0_dag(t) @ H_drive_ii(t) @ U0(t)


# -----------------------------
# Step 6a Apply RWA: keep omega_0 - omega_d (slow), drop omega_d + omega_0 (fast)
# -----------------------------
def H_drive_ii_IP_RWA(t:float, keep="diff")->np.ndarray:
    A_sum = 0.5 * ℏ * gamma * (np.exp(1j * (omega_d + omega_0) * t) * sigma_p 
                               + np.exp(-1j * (omega_d + omega_0) * t) * sigma_m)
    
    A_diff = 0.5 * ℏ * gamma * (np.exp(1j * (omega_d - omega_0) * t) * sigma_m 
                               + np.exp(-1j * (omega_d - omega_0) * t) * sigma_p)
    if keep == "sum":
        return A_sum
    elif keep == "diff":
        return A_diff
    else:
        raise ValueError("Keep must be 'sum' or 'diff'")
    
 
# -----------------------------
# Check resonance (It's done by omega_d = omega_0)
# -----------------------------
def H_drive_ii_IP_RWA_resonant(t:float):
    return H_drive_ii_IP_RWA(t, keep="diff")

# Sanity check 
target_ii = 0.5 * ℏ * gamma * sigma_x
for t in [0.0, 0.123, 3.1415, 17.0]:
    Hii = H_drive_ii_IP_RWA_resonant(t)
    assert np.allclose(Hii, target_ii, atol=1e-12), f"H_drive for case (ii) Mismatch at t={t}"
    
# -----------------------------
# Step 7 define U_I(t) (with RWA)
# -----------------------------
def U_I_closed_form_ii(t:float)->np.ndarray:  
    b = 0.5 * gamma * t
    return np.cos(b) * I2 - 1j * np.sin(b) * sigma_x

def U_I_numeric_ii(t:float)->np.ndarray:
    w_ii,V_ii = np.linalg.eigh(H_drive_ii_IP_RWA_resonant(t))  
    phase = np.exp(-1j * w_ii * t / ℏ)
    return (V_ii @ np.diag(phase) @ V_ii.conj().T ) 

# Sanity check
for t in [0.0, 0.123, 1.7, 5.3]: #check that 2 methods match
    assert np.allclose(U_I_closed_form_ii(t), U_I_numeric_ii(t), atol=1e-12), f"U_I Mismatched for case (ii) at t={t}"
    
for t in [0.0, 0.123, 1.7, 5.3]: 
    U_ii = U_I_numeric_ii(t)
    assert np.allclose(U_ii.conj().T @ U_ii, I2, atol=1e-12), f"U_I Mismatched for case (ii) at t={t}"


# -----------------------------
# Step 8 |ψ_I(t)> = U_I(t)|ψ(0)> (with RWA)
# -----------------------------
psi0 = ket0 

def psi_I_ii(t:float)->np.ndarray:
    return U_I_numeric_ii(t) @ psi0


# -----------------------------
# Step 9 Find Probability of finding in |0> (with RWA)
# -----------------------------
def P0_ii_RWA(t:float)->float:
    amp_ii = (bra0 @ psi_I_ii(t)).item()#not an array anymore, its a float or complex now
    return np.abs(amp_ii**2) #dont need np.real and float

# Sanity check
for t in [0.0, 0.123, 3.124, 5.1]:
    assert np.allclose(P0_ii_RWA(t), np.cos(0.5 * gamma * t)**2, atol=1e-12), f"Probability mismatched for case (ii) at t={t}"
    
### Probabiity of finding in |1>
def P1_ii_RWA(t:float)->float:
    amp_ii = (bra1 @ psi_I_ii(t)).item()#not an array anymore, its a float or complex now
    return np.abs(amp_ii**2) #dont need np.real and float

# Sanity check
for t in [0.0, 0.123, 3.124, 5.1]:
    assert np.allclose(P1_ii_RWA(t), np.sin(0.5 * gamma * t)**2, atol=1e-12), f"Probability mismatched for case (ii) at t={t}"
    
# -----------------------------
# Step 7b define U_I(t)= exp(-iHΔt/ℏ) (without RWA) 
# -----------------------------
# from step 6:

def U_step_from_H(H:np.ndarray, dt:float)->np.ndarray:
    w, V = np.linalg.eigh(H)  #diagonalize H = V diag(w) V^\dagger
    phase = np.exp(-1j * w * dt / ℏ) # exp(-iwdt/ℏ)
    return V @ np.diag(phase) @ V.conj().T # exp(-iHdt/ℏ)

# -----------------------------
# Step 8b & 9b |ψ_I(t)> = U_I(t)|ψ(0)> & Probability to find in |0> (without RWA)
# -----------------------------
def evolve_no_RWA(HI_func, times): #return list of P0(t) without RWA (exact) and the final state psi
    U = I2.copy() 
    P1_list = [0.0]  # Initial: at t=times[0], state is |0>, P0=1
    for k in range(len(times)-1):
        t_mid = 0.5 * (times[k] + times[k+1])
        dt = times[k+1] - times[k]
        H_mid = HI_func(t_mid)  #evaluate the exact H_drive_i_IP at the midpoint
        U = U_step_from_H(H_mid, dt) @ U #left-multiply the new small-step unitary to accumulate 
        psi = U @ ket0  # |ψ_I(t)> = U_I(t)|ψ(0)>
        amp = (bra1 @ psi).item()  # amplitude = <0|ψ(t_k)>
        P1_list.append(abs(amp)**2)
    return np.array(P1_list), psi  #|psi(times[-1])>

    
# -----------------------------
# Step 10 Plot both with RWA and without RWA 
# -----------------------------
P1_ii_RWA_vals = [P1_ii_RWA(t) for t in times]
P1_ii_noRWA, psi_final = evolve_no_RWA(H_drive_ii_IP, times)
plt.figure()
plt.plot(times, P1_ii_RWA_vals, color='blue', label='with RWA', linewidth=2)
plt.plot(times, P1_ii_noRWA, color='red', label='exact dynamics', linewidth=2)
plt.ylabel("P(|1⟩)")
plt.xlabel("t")
plt.title("Weak-driving Rabi oscillation with and without RWA")
plt.legend()
plt.tight_layout()

#quantify error (wiggles)
err_ii = np.abs(P1_ii_noRWA - P1_ii_RWA_vals)
print("max |error|:", err_ii.max())
plt.figure()
plt.plot(times, err_ii)
plt.xlabel("t")
plt.ylabel(r"|$P_|1⟩^{exact} - P_|1⟩^{RWA}$|")
plt.title("Deviation from RWA (fast-term wiggles)")
plt.ylim(0.01,0.55)
plt.xlim(0.0,0.5)
plt.tight_layout()
plt.show()

# -----------------------------
# Step 11 Compute fidelity 
# -----------------------------
t_prime = np.pi / gamma # cos(0.5 * gamma * t') = 0 t' is the time when the state reaches |1> under RWA
times_prime = np.linspace(0.0, t_prime, 2001) #new array of times, evolve half the total time only
P0_prime, psi_t_prime = evolve_no_RWA(H_drive_ii_IP, times_prime) #psi_t_prime is the state at t_prime without RWA
amp_1 = (bra1 @ psi_t_prime).item()
F = np.abs(amp_1)**2 * 100
print("Fidelity = ",F,"%")

# -----------------------------
# Plotting resonance width
# -----------------------------
def _halfmax_width(delta, P): #to find FWHM width
    y = P - 0.5
    idx = np.where(np.signbit(y[:-1]) != np.signbit(y[1:]))[0] #the function crossed zero, a half max crossing exist
    left_idx  = [i for i in idx if delta[i] < 0 and delta[i+1] < 0]  #split crossings into those on left & right of zero 
    right_idx = [i for i in idx if delta[i] > 0 and delta[i+1] > 0]

    def linroot(i): #linear interpolation to find the points where P(delta)=0.5, closest to x=0 (peak)
        x1, x2 = delta[i], delta[i+1]
        y1, y2 = y[i], y[i+1]
        return x1 - y1 * (x2 - x1) / (y2 - y1) #form linear eqn x=my+c, find c (x intercept)

    if len(left_idx) == 0 or len(right_idx) == 0:
        return np.nan #not a number

    ΔL = linroot(max(left_idx)) #negative value max is closest to 0
    ΔR = linroot(min(right_idx))  #positive value min is closest to 0
    return float(ΔR - ΔL) #FWHM: measure how broad the resonance is, narrow-> highly-freq selective, broader-> higher tolerant to detuning

def resonance_curve(case='i', detuning_span=5.0, num_points=4001):
    if case == 'i':
        Omega_R = 2.0 * gamma
        t_pi    = np.pi / (2.0 * gamma)
    elif case == 'ii':
        Omega_R = 1.0 * gamma
        t_pi    = np.pi / (1.0 * gamma)
    else:
        raise ValueError("case must be 'i' or 'ii'")

    tri = np.linspace(-detuning_span*gamma, detuning_span*gamma, num_points) #building x axis 
    omega_eff = np.sqrt(Omega_R**2 + tri**2) #standard 2 level formula for effective rabi frequency at detuning
    P1 = (Omega_R**2 / omega_eff**2) * (np.sin(0.5 * omega_eff * t_pi))**2 #probability to be in |1> after the pulse (y-axis)

    fwhm = _halfmax_width(tri, P1)
    return tri, P1, fwhm

Δ_ii, P1_ii, fwhm_ii = resonance_curve('ii', detuning_span=5.0, num_points=4001)

print(f"FWHM = {fwhm_ii:.4f} rad/s (≈ {fwhm_ii/gamma:.4f} × γ)")

plt.figure()
plt.plot(Δ_ii/gamma, P1_ii, label=f"FWHM ≈ {fwhm_ii/gamma:.3f}·γ", linewidth=2)
plt.axhline(0.5, linestyle='--', linewidth=1)  # Half-maximum line
if np.isfinite(fwhm_ii):
    plt.axvline(+0.5*(fwhm_ii/gamma), linestyle='-.', linewidth=1)
    plt.axvline(-0.5*(fwhm_ii/gamma), linestyle='-.', linewidth=1)
plt.xlabel(r"Detuning  $\Delta/\gamma$")
plt.ylabel(r"Excitation at $\pi$-pulse:  $P_{|1\rangle}(\Delta)$")
plt.title("Resonance curves (weak-driving RWA)")
plt.legend()

# ====================================================
# Plot resonance curves (absolute Δ, not normalized)
# ====================================================
plt.figure()
plt.plot(Δ_ii, P1_ii, linewidth=2)
plt.axvline(0.0, linestyle='--', linewidth=1)
plt.xlabel(r"Detuning  $\delta$ (rad/s)")   # absolute, not normalized
plt.ylabel(r"Excitation at $\pi$-pulse:  $P_{|1\rangle}(t_π)$")
plt.title("Resonance curve")

plt.legend()
plt.tight_layout()
#plt.show()

# ========================================================
# Plotting fidelity against gamma/omega_0
# ========================================================
# ----- Parametric U0 for H0 = (ħ ω0/2) σz -----
def U0_param(t: float, omega_0: float) -> np.ndarray:
    return np.array([
        [np.exp(-0.5j * omega_0 * t), 0.0],
        [0.0, np.exp(+0.5j * omega_0 * t)]
    ], dtype=complex)

def U0_dag_param(t: float, omega_0: float) -> np.ndarray:
    return dagger(U0_param(t, omega_0))

# ----- Case (ii) drive in LAB and in Interaction Picture -----
def H_drive_ii_lab(t: float, gamma: float, omega_d: float) -> np.ndarray:
    return ℏ * gamma * np.cos(omega_d * t) * (sigma_p + sigma_m)

def H_drive_ii_IP_param(t: float, gamma: float, omega_d: float, omega_0: float) -> np.ndarray:
    U0t = U0_param(t, omega_0)
    return U0_dag_param(t, omega_0) @ H_drive_ii_lab(t, gamma, omega_d) @ U0t

# ----- Gate fidelity for case (ii), no-RWA, evaluated at t_pi^RWA = pi/gamma -----
def gate_fidelity_caseii_noRWA_at_RWA_tpi(
    gamma: float,
    omega_0: float,
    omega_d: float = None,
    N_steps: int = 2001,
) -> float:
    if omega_d is None:
        omega_d = omega_0  # resonance choice

    t_pi = np.pi / abs(gamma)  # case (ii) RWA pi-time
    times = np.linspace(0.0, t_pi, N_steps)

    HI_func = lambda t: H_drive_ii_IP_param(t, gamma, omega_d, omega_0)

    # reuse your existing evolve_no_RWA(HI_func, times) which returns (P0_list, psi_final)
    _, psi_final = evolve_no_RWA(HI_func, times)

    amp_1 = (bra1 @ psi_final).item()
    F = np.abs(amp_1) ** 2  # probability in |1>
    return float(F)

# ============================================================
# Plot: x = gamma/omega0, y = gate fidelity (case ii, no-RWA at RWA t_pi)
# ============================================================
omega_d = omega_0  # enforce resonance in the weak-RWA sense

gamma_ratio_grid = np.linspace(0.1, 6.0, 200)  # x-axis points: gamma/omega0
F_list = []

for r in gamma_ratio_grid:
    gam = r * omega_0
    F = gate_fidelity_caseii_noRWA_at_RWA_tpi(
        gamma=gam,
        omega_0=omega_0,
        omega_d=omega_d,
        N_steps=2001
    )
    F_list.append(F)

F_list = np.array(F_list)

plt.figure()
plt.plot(gamma_ratio_grid, F_list, linewidth=2)

# --- mark local dip near x ~ 1 by restricting to a window ---
x = gamma_ratio_grid
y = F_list

xL, xR = 0.7, 1.4   # adjust if you want a wider/narrower search window
mask = (x >= xL) & (x <= xR)

idx_local = np.nanargmin(y[mask])
idx_dip = np.where(mask)[0][idx_local]

x_dip = x[idx_dip]
y_dip = y[idx_dip]

plt.axvline(x_dip, color="black", linestyle=":", linewidth=2)
plt.plot([x_dip], [y_dip], marker="o", color="black")
plt.annotate(
    rf"local dip: $\gamma/\omega_0={x_dip:.2f}$",
    xy=(x_dip, y_dip),
    xytext=(6, 10),
    textcoords="offset points",
    ha="left",
    va="bottom",
    fontsize=9,
)


# --- mark local maximum after the dip ---
xP_L, xP_R = x_dip, 1.8   # adjust right bound if needed
mask_peak = (x >= xP_L) & (x <= xP_R)

idx_local_peak = np.nanargmax(y[mask_peak])
idx_peak = np.where(mask_peak)[0][idx_local_peak]

x_peak = x[idx_peak]
y_peak = y[idx_peak]

plt.axvline(x_peak, color="red", linestyle="--", linewidth=2)
plt.plot([x_peak], [y_peak], marker="o", color="gray")
plt.annotate(rf" $\gamma/\omega_0={x_peak:.2f}$",
    xy=(x_peak, y_peak),
    xytext=(6, -18),
    textcoords="offset points",
    ha="left",
    va="top",
    fontsize=9,
    color="red",
)

plt.xlabel(r"$\gamma/\omega_0$")
plt.ylabel(r"Fidelity  $|\langle 1|\psi_{lab}(t_\pi^{\rm RWA})\rangle|^2$")
plt.title(r"Fidelity against $\gamma/\omega_0$")
plt.ylim(-0.02, 1.02)
plt.tight_layout()
plt.show()

