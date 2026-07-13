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

> **Steps 3a, 3b and 4 (bound-state and linewidth scripts) are not published in
> this repository.** They are kept in a private companion repository and shared
> on request — see §3.4.

> **All `.npz` files live in the `npz_data/` sub-folder.** Each script reads its
> inputs from and writes its outputs to that folder, which is created
> automatically when needed. (The ground-state potential read by step 4 is a
> `.tsv` file that lives in `data/`.)

- `molecular_potentials.py` is a **library imported** by step 2, not a script to
  run (except to visually check the potentials).
- The DVR / WKB / FCF methods (steps 3a/3b/4) were originally ported from a
  ⁸⁸Sr₂ Mathematica notebook; those functions were then **copied into** each
  script, which is therefore **self-contained** (no external module import).

```bash
# Prerequisites (once) — this repository only
python3 build_cg_tables.py
python3 compute_adiabatic_curves.py
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

### 3.4 Bound-state & linewidth scripts (steps 3a/3b/4) — private

`compute_binding_energies.py` (DVR, mapped Fourier grid), `compute_binding_energies_wkb.py`
(WKB, Bohr-Sommerfeld) and `compute_pa_linewidths.py` (Franck-Condon factors and
photoassociation linewidths) are **not published in this repository**. They live
in a private companion repository (`sr87-photoassociation-private`) and are
shared on request — please contact the author to request access.

The DVR and WKB routines were originally ported from a ⁸⁸Sr₂ Mathematica
notebook (Borkowski 2014 / Tiemann 2010) and then copied into each script,
which is self-contained (no external module import).

> ⚠️ **Non-obvious point, for context.** What the project calls "**DVR**" is
> really the **mapped Fourier grid method** (Kokoouline, *JCP* **110**, 9865
> (1999); Willner, *JCP* **120**, 548 (2004)), **not** a constant-step
> sinc-DVR. The mesh is **non-uniform**: dense at short range (deep well, fast
> oscillations) and sparse at long range. This is essential here: the wells
> are ~112 GHz deep and support ~110 levels, the highest of which extend out
> to several hundred `a₀`; a uniform sinc-DVR does not converge (the number of
> levels depends on the step).

### 3.5 `visualization/plot_potentials_levels.py` — figures
Reloads the precomputed `.npz` files (no recomputation) and overlays the retained
adiabatic potentials with their bound levels drawn as horizontal lines. Supports
`--source {wkb,dvr,both}` and `--rmin/--rmax/--emin/--emax/--save` options.

---

## 4. Data formats — this repository's output

| File                             | Main keys | Units |
|----------------------------------|-----------|-------|
| `adiabatic_curves_physical.npz`  | `r_tab` (5001,), `eig_physical` (5001, 980) | a₀ ; GHz |

The private companion scripts additionally produce `binding_energies_dvr.npz`,
`binding_energies_wkb.npz` and `pa_linewidths.npz` (binding energies in MHz,
FCF and linewidths in Hz) — documented in that repository.

Common conventions: `R` in Bohr radii `a₀`; binding energies in **MHz**,
**positive**, level `v = 0` (deepest) first in each array. Reduced mass
⁸⁷Sr-⁸⁷Sr: `μ = 86.90888 / 2` in atomic mass units.

---

## 5. Dependencies

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

## 6. References

- P. S. Julienne & F. H. Mies, *J. Opt. Soc. Am. B* **6**, 2257 (1989)
- M. Borkowski et al., *Phys. Rev. A* **90**, 032713 (2014) — 0u/1u potentials, mass scaling
- E. Tiemann et al., *Eur. Phys. J. D* **57**, 171 (2010) — X ¹Σg⁺ ground state
- K. T. Tang & J. P. Toennies, *J. Chem. Phys.* **80**, 3726 (1984) — dispersion damping
- V. Kokoouline et al., *J. Chem. Phys.* **110**, 9865 (1999) — mapped Fourier grid
- K. Willner, O. Dulieu & F. Masnou-Seeuws, *J. Chem. Phys.* **120**, 548 (2004) — symmetric mapped kinetic form
- D. T. Colbert & W. H. Miller, *J. Chem. Phys.* **96**, 1982 (1992) — sinc-DVR (general DVR reference)

---

## 7. License & citation

This project is released under the **MIT License** (see `LICENSE`) — free to use,
modify, and redistribute, provided the copyright notice is retained.

**If you use this code in scientific work, please cite it.** GitHub's "Cite this
repository" button reads the machine-readable `CITATION.cff`; the human-readable
form is:

> A. Meyroneinc, *Sr87 adiabatic curves and bound-state pipeline for
> photoassociation*, 2026. https://github.com/halebanc/sr87-photoassociation

Please also cite the underlying physics references in §6 as appropriate (in
particular Borkowski 2014 for the potentials and Kokoouline 1999 / Willner 2004
for the mapped-grid method).
