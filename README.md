# Strong-Driving Qubit Gate-Time Prediction

Python code for a Final Year Project on gate-time prediction in a linearly driven two-level qubit beyond the standard weak-driving rotating-wave approximation (RWA).

This repository compares two analytical descriptions against exact numerical evolution of the laboratory-frame Hamiltonian:

- the standard weak-driving RWA baseline
- an Ashhab-inspired strong-driving approximation adapted to a transverse harmonic drive

The code is built around the project described in the thesis:

**Gate-Time Prediction in a Strongly Driven Linearly Coupled Qubit Beyond the Weak Rotating-Wave Approximation**

## Project overview

The model studied in this repository is a driven two-level system with Hamiltonian

$$
H(t)=\frac{\hbar \omega_0}{2}\sigma_z + \hbar \gamma \cos(\omega t)\sigma_x
$$

where:

- `omega_0` is the bare qubit splitting
- `gamma` is the drive amplitude
- `omega` is the drive frequency
- the drive acts transversely through `sigma_x`

The main goal is to test how well different approximate pulse-time formulas predict high-fidelity population transfer from `|0>` to `|1>`.

## What is included

### 1. Weak-driving RWA baseline

**File:** `scripts/weak_rwa.py`

This script implements the standard weak-driving RWA and compares it against exact numerical evolution.

Main tasks:

- constructs the lab-frame and interaction-picture Hamiltonians
- applies the weak-driving RWA at near resonance
- compares exact and RWA population dynamics
- computes fidelity at the RWA-predicted `pi`-pulse time
- generates resonance-width and fidelity-versus-drive-strength plots

Use this script as the baseline reference for the weak-driving regime.

### 2. Ashhab-inspired near-resonant scan

**File:** `scripts/ashhab_near_resonant_scan.py`

This script performs a near-resonant search inspired by the strong-driving treatment of Ashhab et al.

Main tasks:

- fixes `omega_0`
- sweeps over drive amplitude `gamma`
- searches over odd harmonic order `m` and drive frequency `omega`
- selects the best near-resonant candidate using a detuning-based score
- evaluates exact fidelity at the Ashhab-predicted pulse time
- compares Ashhab-inspired predictions against the weak-RWA baseline

This script is useful for generating continuous scans in `gamma/omega_0` when exact resonance is not enforced pointwise.

### 3. Ashhab-inspired resonance-enforced scan and diagnostics

**File:** `scripts/ashhab_resonance_enforced_scan_and_diagnostics.py`

This is the main diagnostic script for the strong-driving regime.

Main tasks:

- enforces resonance through the retained harmonic condition
- keeps the `m = 1` branch used in the project analysis
- solves for `omega_0` or scans over `omega` / `alpha` accordingly
- computes exact fidelity at the Ashhab-predicted pulse time
- groups points into high-, middle-, and low-fidelity sets
- evaluates timing mismatch using `t_pi^A / t_opt`
- examines Bessel-function structure through `J0`, `J1`, `J3`, and `J5`
- estimates the residual influence of neglected fast and higher-harmonic terms through diagnostic indicators such as
  - `eps_fast`
  - `eps3`
  - `eps5`
  - `eps_bad_max`
  - `eps_bad_sum`
  - `eps_bad_rss`
- performs summary statistics and Spearman-rank tests

This script is the best starting point for reproducing the diagnostic part of the thesis.

### 4. Ashhab-inspired single-run comparison

**File:** `scripts/ashhab_single_run.py`

This script produces representative time-domain comparisons between:

- exact laboratory-frame dynamics
- Ashhab-inspired effective-frame dynamics
- frame-corrected Ashhab results mapped back to the laboratory frame

Main tasks:

- supports manual parameter choice or scanned parameter selection
- applies the kick-frame and rotating-frame logic behind the Ashhab-inspired treatment
- compares fair and unfair frame comparisons
- highlights why fidelity should be evaluated only after mapping the effective-frame state back to the laboratory frame

This script is useful for producing the illustrative single-run plots used in the results chapter.

## Repository structure

```text
strong-driving-qubit-gate-time/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── weak_rwa.py
│   ├── ashhab_near_resonant_scan.py
│   ├── ashhab_resonance_enforced_scan_and_diagnostics.py
│   └── ashhab_single_run.py
└── docs/
```

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The scripts use:

- `numpy`
- `scipy`
- `matplotlib`

## How to run

Run any script directly, for example:

```bash
python scripts/weak_rwa.py
```

or

```bash
python scripts/ashhab_resonance_enforced_scan_and_diagnostics.py
```

Most figures are produced directly through `matplotlib` and displayed at runtime.

## Numerical approach

Across the repository, the exact time evolution is computed from the laboratory-frame Hamiltonian using midpoint propagation with small time steps. The approximate models are then tested by evaluating the exact dynamics at the pulse times predicted by each approximation.

The main observables used throughout the project are:

- target-state population `P(|1>)`
- state-transfer fidelity at the predicted pulse time
- predicted `pi`-pulse time
- timing mismatch relative to the exact optimal transfer time

## Notes on conventions

The code uses the basis

- `|0> = [1, 0]^T`
- `|1> = [0, 1]^T`

with Pauli matrices defined in the usual matrix form. In the scripts, the ladder operators are written as

- `sigma_p = |0><1|`
- `sigma_m = |1><0|`

so it is best to follow the code conventions directly when comparing formulas.

Also note that most scripts set

- `hbar = 1`

for convenience.

## Suggested use cases

- **Reproduce weak-RWA validation plots** -> run `weak_rwa.py`
- **Generate near-resonant Ashhab scans** -> run `ashhab_near_resonant_scan.py`
- **Study strong-driving diagnostics and validity indicators** -> run `ashhab_resonance_enforced_scan_and_diagnostics.py`
- **Generate representative single-run comparisons** -> run `ashhab_single_run.py`

## Possible future cleanup

If this repository is going to be shared publicly, a good next step would be to:

- refactor repeated utilities into a shared module
- standardize variable names across scripts
- add figure-saving options instead of only interactive plotting
- separate plotting, numerics, and scan logic into reusable functions
- include a PDF version of the thesis in `docs/`

## Suggested repository description

**Strong-driving qubit gate-time prediction beyond weak RWA: exact simulations, Ashhab-inspired scans, and fidelity diagnostics for a transverse harmonic drive.**
