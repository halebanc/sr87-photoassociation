# Adiabatic curves and bound states of the ⁸⁷Sr₂ dimer — photoassociation

Computes, for the study of **photoassociation** of ultracold ⁸⁷Sr atoms:

1. the **adiabatic potential curves** of the Sr₂ dimer (coupled Hamiltonian:
   hyperfine + fine structure + long-range molecular potentials), then
2. the **bound-state energies** of the vibrational levels supported by each of
   these curves, using **two independent methods**:
   - **DVR** = the **mapped Fourier grid method** (quantum, "exact"),
   - **WKB** = **Bohr-Sommerfeld** quantization (semi-classical).

The dual DVR/WKB computation serves as a cross-check: two methods of different
nature applied **to the same interpolated potential** should give the same
spectrum (to within the expected accuracy of the semi-classical method).

---

## 1. Physical overview

Two ⁸⁷Sr atoms separated by `R` interact through a molecular potential `V(R)`.
For photoassociation we are interested in atoms excited toward the
`5s² ¹S₀ + 5s5p ³P₁` asymptote, where the **resonant dipole** interaction
(`±C₃/R³`) dominates the long range. ⁸⁷Sr has nuclear spin `I = 9/2`: the
**hyperfine structure** couples the molecular channels and multiplies the number
of curves.

The result is a large number (≈ 980) of **adiabatic curves** `Eₐ(R)`, obtained by
diagonalizing a channel-by-channel coupled Hamiltonian on an `R` grid. Each curve
is an effective 1D potential; atoms can form **bound molecules** in it
(vibrational states `v = 0, 1, …`) whose energies we want.

> **Isotope note.** The base molecular potentials (`molecular_potentials.py`) are
> fitted to ⁸⁸Sr ab initio data (Borkowski 2014). The transition to ⁸⁷Sr is done
> through the **reduced mass** (mass scaling) and by adding the **hyperfine
> Hamiltonian** (constants `A`, `B`, spin `I = 9/2`). This is the job of
> `compute_adiabatic_curves.py`.

---

## 2. Pipeline — execution order

```
1.  build_cg_tables.py            →  precomputed_cg_tables.npz      (once, slow)
2.  compute_adiabatic_curves.py   →  adiabatic_curves_physical.npz  (the 980 curves)
3a. compute_binding_energies.py      →  binding_energies_dvr.npz    (bound states, DVR)
3b. compute_binding_energies_wkb.py  →  binding_energies_wkb.npz    (bound states, WKB)
4.  compute_pa_linewidths.py      →  pa_linewidths.npz              (FCF + PA linewidths)
```

> **All `.npz` files live in the `npz_data/` sub-folder.** Each script reads its
> inputs from and writes its outputs to that folder, which is created
> automatically when needed. (The ground-state potential read by step 4 is a
> `.tsv` file that lives in `data/`.)

- Steps **3a** and **3b** are **independent** of each other and can run in any
  order, but both **after** step 2. Step **4** (FCF + photoassociation
  linewidths) also comes after step 2 (it recomputes the excited bound states it
  needs on its own).
- `molecular_potentials.py` is a **library imported** by step 2, not a script to
  run (except to visually check the potentials).
- The DVR / WKB / FCF methods were originally ported from a ⁸⁸Sr₂ Mathematica
  notebook; those functions have since been **copied into** scripts 3a/3b/4, which
  are therefore **self-contained** (no external module import).

```bash
# Prerequisites (once)
python3 build_cg_tables.py
python3 compute_adiabatic_curves.py

# Bound states (both methods)
python3 compute_binding_energies.py        # DVR  (~70 min, 375 curves)
python3 compute_binding_energies_wkb.py    # WKB  (~40 min, 375 curves)
```

Tip: to follow the progress bar without opening the final figure, run with
`MPLBACKEND=Agg python3 …` (disables the matplotlib display).

---

## 3. File descriptions

### 3.1 `molecular_potentials.py` — long-range adiabatic potentials
Defines the **four** molecular potentials used by the coupled Hamiltonian in
step 2:

| Function        | Channel        |
|-----------------|----------------|
| `potential_0u`  | 0ᵤ (³Πᵤ + C₃)  |
| `potential_1u`  | 1ᵤ             |
| `potential_0g`  | 0g             |
| `potential_1g`  | 1g             |

Each potential = a short-range part (extended Morse type:
polynomial × exponential) + a damped **Tang-Toennies** dispersion series
(C₆, C₈, C₁₀, C₁₂) + a **resonant dipole** term `±C₃/R³` (the sign depends on the
g/u symmetry). Parameters: Borkowski et al., *PRA* **90**, 032713 (2014). Output
in **GHz**, `R` in Bohr radii `a₀`. Run standalone
(`python3 molecular_potentials.py`) to plot the curves.

### 3.2 `build_cg_tables.py` — coupling coefficients (cache)
**Symbolic** computation (sympy) of the Clebsch-Gordan and Wigner 6j coefficients
of the coupled Hamiltonian. Slow (several minutes), done **once**: the result is
cached in `precomputed_cg_tables.npz`. Parameters at the top of the file
(`J=1, j1=1, i1=i2=9/2, p=-1`).

### 3.3 `compute_adiabatic_curves.py` — curve diagonalization
Diagonalizes the coupled Hamiltonian `H(R)` on a **uniform grid** in `R`
(`H_STEP = 0.1 a₀`, `R ∈ [3, 503] a₀` → 5001 points), with **physical sorting** of
the curves done on the fly:

- **LOBPCG** diagonalization of the sparse matrix `H(R)`, with a "warm start"
  (eigenvectors from step `R` reused at step `R+ΔR`);
- at each step, the **Hungarian algorithm** matches consecutive curves by
  eigenvector overlap `O[j,k] = |⟨ψⱼ(Rᵢ)|ψₖ(Rᵢ₊₁)⟩|`, which correctly tracks both
  real and avoided crossings.

Output `adiabatic_curves_physical.npz`:
- `r_tab` — uniform `R` grid (5001,), in `a₀`;
- `eig_physical` — sorted eigenvalues (5001, 980), in **GHz**.

### 3.4 Origin of the DVR / WKB methods
The DVR and WKB routines were originally ported from a ⁸⁸Sr₂ Mathematica notebook
(Borkowski 2014 / Tiemann 2010). They have since been **copied verbatim** into
scripts 3a/3b, which no longer depend on any external module (they are
self-contained). The copied functions are:

| Function                  | Copied into | Role |
|---------------------------|-------------|------|
| `make_pot_env`            | DVR | builds the **smoothed envelope** of the potential that defines the mapping |
| `build_mapped_grid`       | DVR | builds the non-uniform **mapped grid** `x(R)`, the Jacobian `J(x)` and its derivatives |
| `build_kinetic`           | DVR | Fourier-grid **kinetic energy** matrix (symmetric Willner form) |
| `solve_single_channel`    | DVR | assembles `H = T + V`, **diagonalizes** (`eigh`), selects the bound states |
| `wkb_vibrational_number`  | WKB | **WKB quantum number** `v(E)` via the Bohr-Sommerfeld action integral |

Internal units: **hartree / Bohr / atomic mass unit**. `solve_single_channel`
already returns energies in **MHz**. The physical constants (CODATA 2018) are also
copied at the top of each script.

> ⚠️ **Non-obvious point.** What the project calls "**DVR**" is really the
> **mapped Fourier grid method** (Kokoouline, *JCP* **110**, 9865 (1999);
> Willner, *JCP* **120**, 548 (2004)), **not** a constant-step sinc-DVR. The mesh
> is **non-uniform**: dense at short range (deep well, fast oscillations) and
> sparse at long range. This is essential here: the wells are ~112 GHz deep and
> support ~110 levels, the highest of which extend out to several hundred `a₀`; a
> uniform sinc-DVR does not converge (the number of levels depends on the step).

### 3.5 `compute_binding_energies.py` — bound states by **DVR** (step 3a)
See §4. Output `binding_energies_dvr.npz`.

### 3.6 `compute_binding_energies_wkb.py` — bound states by **WKB** (step 3b)
See §5. Output `binding_energies_wkb.npz`.

### 3.7 `compute_pa_linewidths.py` — Franck-Condon factors & PA linewidths (step 4)
See §7. Reads the ground-state potential from `data/`. Output `pa_linewidths.npz`.

### 3.8 `visualization/plot_potentials_levels.py` — figures
Reloads the precomputed `.npz` files (no recomputation) and overlays the retained
adiabatic potentials with their bound levels drawn as horizontal lines. Supports
`--source {wkb,dvr,both}` and `--rmin/--rmax/--emin/--emax/--save` options.

---

## 4. DVR method (mapped Fourier grid) — `compute_binding_energies.py`

### 4.1 Curve selection
Only curves **near the dissociation threshold** are treated: asymptote
`E∞ < 1 GHz` (`ASYM_THRESH`), where the asymptote is estimated as the **mean of
the last 50 points** in `R` (`N_TAIL`). → **375 curves** retained (many are
degenerate in their hyperfine asymptote to ~10⁻⁴ GHz — this is expected, they are
all treated).

### 4.2 Potential interpolation — **brick shared with WKB**
For **each** curve `k`, `make_PotFun` builds a continuous potential function
`V(R)`:

1. **shift the asymptote to 0**: `V = (eig_physical[:,k] − E∞)` — the mapped
   method expects `V → 0` as `R → ∞` and keeps only eigenvalues in
   `[−|depth|, 0]`;
2. GHz → hartree conversion;
3. **interpolation** with `PchipInterpolator` (piecewise monotone, no spurious
   oscillation) on the tabulated grid;
4. **strictly flat tail** `V(R) = 0` beyond the last grid point
   (`R > r_tab[-1]`).

> It is **this single interpolated function** that is passed to **both** the DVR
> and WKB solvers. Neither reads the raw tabulated points: both evaluate `V(R)` at
> their own abscissas. Guarantee: the two methods see an **identical** potential.

### 4.3 Solving (`dvr_levels`)
For a curve's potential `V(R)`:

```
make_pot_env([V], …)        → smoothed envelope + well depth + inner cutoff
build_mapped_grid(…)        → mapped grid x(R), Jacobian J(x), J', J''
build_kinetic(…)            → kinetic matrix T (Fourier-grid)
solve_single_channel(V, …)  → eigh(T + diag(V)) ; bound states = eigenvalues < 0
```

`solve_single_channel` keeps the eigenvectors whose energy is in `[−|depth|, 0]`
**and** whose amplitude at the outer grid edge is negligible (a genuinely bound
state, not a discretized continuum state).

Since the asymptote was shifted to 0, the **binding energy** of a level is
`E_b = −E_level` (positive). They are sorted `v = 0` (deepest) first.

### 4.4 Key parameters (at the top of the file)

| Parameter        | Value | Role |
|------------------|--------|------|
| `ASYM_THRESH`    | 1.0 GHz | curve selection threshold |
| `N_TAIL`         | 50      | points used to estimate the asymptote |
| `LTOT`           | 503 a₀  | outer edge of the mapped grid (= grid `R_max`) |
| `EFREE_PART_NK`  | 8.0e6 (≈ 8 mK) | "free particle" energy setting the mesh (see §6) |
| `DELTA_X_RESOL`  | 0.45    | grid resolution factor |
| `R_DELTA_PREC`   | (0.02, 0.2, 2.0) | fine steps of the auxiliary `x(R)` grid per slice |
| `STOPS`          | (60, 500) | slice boundaries of the auxiliary grid |
| `TURN_POINT_MIN` | 6.5 a₀  | inner turning-point estimate |
| `WELL_MAX`       | 30 a₀   | upper bound of the **well-bottom** search |

### 4.5 Output `binding_energies_dvr.npz`
- `selected_idx` — indices of the 375 retained curves;
- `curve_{k}_Eb_MHz` — binding energies of curve `k`, in **MHz**, `v = 0`
  (deepest) first;
- `curve_{k}_Easym_GHz` — asymptote of curve `k`, in GHz;
- `meta_method`, `meta_mu_au`, `meta_Ltot_aB`, `meta_DeltaxResol` — metadata.

Only curves with at least one bound state are saved.

---

## 5. WKB method (Bohr-Sommerfeld) — `compute_binding_energies_wkb.py`

### 5.1 Principle
Semi-classical quantization with **two soft turning points** `R₁ < R₂` (where
`V(R) = E`). The continuous **quantum number** is (atomic units, `ħ = 1`):

```
v(E) = (factAux / π) · ∫_{R₁}^{R₂} √( |E − V(R)| ) dR  −  1/2
```

— this is exactly the `wkb_vibrational_number` function (copied into
`compute_binding_energies_wkb.py` from the 88Sr2 module). A bound level `v`
(integer) satisfies `v(E_v) = v`, i.e. the action `∮ p dR = (v + ½)·2πħ`.

### 5.2 Getting the full spectrum (`wkb_levels`)
`wkb_vibrational_number` computes `v` for **one** given energy. To obtain **all**
levels, we **invert** the (monotone) relation `v(E)`:

1. ceiling `v_max = ⌊ v(E → 0⁻) ⌋`;
2. for each integer `v` from 0 to `v_max`, solve `v(E_v) = v` with **`brentq`** on
   `[V_min, E_ceil]`;
3. `E_b = −E_v` (MHz), sorted `v = 0` first.

The potential `V(R)` is the **same interpolated `make_PotFun`** as for the DVR
(imported from `compute_binding_energies.py`).

### 5.3 Key parameters and numerical subtleties

| Parameter        | Value | Role |
|------------------|--------|------|
| `TURN_POINT_MIN` | 6.5 a₀ | inner turning-point estimate |
| `WELL_R_MAX`     | 25 a₀  | upper bound to **locate the well bottom** (see ⚠️) |
| `E_CEIL_MHz`     | 1.0    | binding-energy ceiling for the inversion (see ⚠️) |

> ⚠️ **`WELL_R_MAX` ≠ grid edge.** This parameter is passed as `TurnPointMaxInit`
> to `wkb_vibrational_number`, which uses it as the **upper bound of the
> `minimize_scalar` that finds the well bottom**. The well is **narrow and
> short-range** (minimum near `R ≈ 8 a₀`); with a value as large as the grid edge
> (503 a₀) the bounded minimizer **misses the well** and returns the flat tail,
> breaking the turning-point search (failure "f(a) and f(b) must have different
> signs"). The routine's inner loop extends the **outer** turning point to large
> `R` on its own anyway.

> ⚠️ **`E_CEIL_MHz` > tail noise.** Near the threshold, the outer turning point
> runs off to infinity and the interpolated tail carries **~100 kHz numerical
> noise**. We therefore cap the inversion at a binding energy of 1 MHz: we drop
> the levels bound by **less than ~1 MHz** — precisely those where WKB is least
> reliable anyway (the DVR keeps them).

### 5.4 Output `binding_energies_wkb.npz`
Same schema as the DVR: `selected_idx`, `curve_{k}_Eb_MHz`,
`curve_{k}_Easym_GHz`, plus `meta_method`, `meta_mu_au`, `meta_Eceil_MHz`.

---

## 6. Robustness — pitfalls encountered and fixed

These methods (from the 88Sr2 module) were written for **smooth analytic**
potentials. Applied to **interpolated** curves (noisy, asymptote sometimes
slightly positive), three problems appeared; all are fixed and documented in the
code:

1. **Forced flat tail** (`make_PotFun`). Free spline extrapolation beyond the grid
   can dip below 0 because of the tail noise, making the WKB outer turning point
   undefined. → we impose `V(R) = 0` for `R > r_tab[-1]`.

2. **`EFreePart` raised to ~8 mK** (DVR). Building the mapped grid integrates
   `√(EFree − V_envelope)`. The tail noise (~MHz ≈ 10⁻⁸ hartree) exceeds the
   notebook's `EFreePart` (2.5 mK ≈ 8×10⁻⁹ hartree) on 9 curves: the argument of
   the square root goes negative, `x(R)` stops being strictly increasing, and
   `build_mapped_grid` raises. → `EFreePart = 8 mK` clears the worst noise
   (~1.6 MHz) with margin, without changing the grid resolution
   (`NptX ≈ 4800-4900`).

3. **Bounded well-bottom search** (WKB): see ⚠️ `WELL_R_MAX` above.

Both scripts additionally wrap each curve in a `try/except`: a curve that fails is
**counted** (`fail=`) and **skipped** without interrupting the run.

---

## 7. Photoassociation linewidths — `compute_pa_linewidths.py`

The strength of a photoassociation line is governed by the bound-bound
Franck-Condon factor (FCF) between a ground-state X¹Σg⁺ vibrational level and an
excited-state vibrational level:

```
FCF(v_g, v_e) = |∫ ψ_g(R) ψ_e(R) R² dR|²
                / ( ∫ ψ_g² R² dR  ·  ∫ ψ_e² R² dR )
```

and the **partial radiative linewidth** of each excited level:

```
Γ(v_e) = γ_at · Σ_{v_g} FCF(v_g, v_e)
```

where `γ_at` is the atomic ³P₁ → ¹S₀ spontaneous-emission rate of Sr (689 nm
intercombination line, τ ≈ 21.3 µs). The excited wavefunctions come from the Sr87
adiabatic curves (mapped-grid DVR); the ground potential is the isotope-
independent ⁸⁸Sr X¹Σg⁺ potential (`data/PotGroundShapeStein2010Version2016.tsv`),
solved here with `μ(⁸⁷Sr)`. Only the curves listed in `CURVE_INDICES` are treated.
Output `pa_linewidths.npz`.

---

## 8. DVR vs WKB cross-check

On a deep reference curve (~112 GHz, ~113 levels):

- **same number of levels** (to within ±1);
- **~1–2 %** agreement on the intermediate and deep levels;
- the least-bound **halo state** differs the most (typically 0.16 MHz in DVR vs
  ~1 MHz in WKB) — this is the **expected** behavior: the semi-classical method is
  poor for the state closest to threshold (long tail, distant outer turning
  point). This is exactly what a DVR/WKB comparison should reveal.

To compare afterwards, load the two `.npz` files and overlay `E_b(v)`:

```python
import numpy as np, matplotlib.pyplot as plt
dvr = np.load("npz_data/binding_energies_dvr.npz")
wkb = np.load("npz_data/binding_energies_wkb.npz")
k = 0  # a curve present in both
ebd = np.sort(dvr[f"curve_{k}_Eb_MHz"])   # ascending
ebw = np.sort(wkb[f"curve_{k}_Eb_MHz"])
plt.semilogy(ebd, "o-", label="DVR")
plt.semilogy(ebw, "s-", label="WKB")
plt.xlabel("index (least bound to most bound)"); plt.ylabel("E_b (MHz)")
plt.legend(); plt.show()
```

---

## 9. Data formats (summary)

| File                             | Main keys | Units |
|----------------------------------|-----------|-------|
| `adiabatic_curves_physical.npz`  | `r_tab` (5001,), `eig_physical` (5001, 980) | a₀ ; GHz |
| `binding_energies_dvr.npz`       | `selected_idx`, `curve_{k}_Eb_MHz`, `curve_{k}_Easym_GHz`, `meta_*` | MHz ; GHz |
| `binding_energies_wkb.npz`       | `selected_idx`, `curve_{k}_Eb_MHz`, `curve_{k}_Easym_GHz`, `meta_*` | MHz ; GHz |
| `pa_linewidths.npz`              | `E_ground_MHz`, `curve_{k}_FCF`, `curve_{k}_Gamma_Hz`, `meta_*` | MHz ; Hz |

Common conventions: `R` in Bohr radii `a₀`; binding energies in **MHz**,
**positive**, level `v = 0` (deepest) first in each array. Reduced mass
⁸⁷Sr-⁸⁷Sr: `μ = 86.90888 / 2` in atomic mass units.

---

## 10. Dependencies

```
numpy
scipy
matplotlib
sympy          # only for build_cg_tables.py
mpmath         # only for build_cg_tables.py
```

```bash
pip install numpy scipy matplotlib sympy mpmath
```

---

## 11. References

- P. S. Julienne & F. H. Mies, *J. Opt. Soc. Am. B* **6**, 2257 (1989)
- M. Borkowski et al., *Phys. Rev. A* **90**, 032713 (2014) — 0u/1u potentials, mass scaling
- E. Tiemann et al., *Eur. Phys. J. D* **57**, 171 (2010) — X ¹Σg⁺ ground state
- K. T. Tang & J. P. Toennies, *J. Chem. Phys.* **80**, 3726 (1984) — dispersion damping
- V. Kokoouline et al., *J. Chem. Phys.* **110**, 9865 (1999) — mapped Fourier grid
- K. Willner, O. Dulieu & F. Masnou-Seeuws, *J. Chem. Phys.* **120**, 548 (2004) — symmetric mapped kinetic form
- D. T. Colbert & W. H. Miller, *J. Chem. Phys.* **96**, 1982 (1992) — sinc-DVR (general DVR reference)
```
