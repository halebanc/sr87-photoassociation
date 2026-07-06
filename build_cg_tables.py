"""
build_cg_tables.py — Precompute Clebsch-Gordan and coupling tables for Sr87.

This is the expensive one-time setup step.  It computes three lists of
quantum-number tuples (ε, γ, π) and two coupling matrices (εγ, επ) using
exact symbolic arithmetic (sympy CG / Wigner 6j).

Output
------
precomputed_cg_tables.npz
    epsilons  : (N_ε, 5)   — [f1, F, R, T, mT]
    gammalist : (N_γ, 6)   — [Ω, i, ℓ, T, mT, Φ]
    pilist    : (N_π, 5)   — [f1, mf1, mi2, R, mR]
    eg_table  : (N_ε, N_γ) — ⟨ε|γ⟩ coupling matrix
    ep_table  : (N_ε, N_π) — ⟨ε|π⟩ coupling matrix

Runtime: several minutes (sympy CG evaluation).
Result is cached: re-running the script reloads the cache unless the
quantum-number parameters change.
"""

import numpy as np
from pathlib import Path
from sympy.physics.quantum.cg import CG, Wigner6j
from tqdm import tqdm

# ── system parameters ─────────────────────────────────────────────────────────
J  = 1       # excited electronic angular momentum
j1 = 1       # same (j1 = J for the excited atom)
i1 = 9/2     # nuclear spin of atom 1 (Sr87)
i2 = 9/2     # nuclear spin of atom 2
p  = -1      # parity

# all Sr87 .npz live in the npz_data/ sub-folder
NPZ_DIR = Path(__file__).with_name("npz_data")
NPZ_DIR.mkdir(exist_ok=True)
_CACHE = NPZ_DIR / "precomputed_cg_tables.npz"
_META  = {"i1": float(i1), "i2": float(i2), "j1": float(j1),
          "J":  float(J),  "p":  float(p)}


# ── coupling functions ────────────────────────────────────────────────────────
def _gamma_epsilon(f1, F, R, T, mT, Ω, i, ℓ, Tp, mTp, Φ):
    cg1   = CG(F, Φ, R, 0, T, Φ)
    cg2   = CG(J, Ω, i, ℓ, F, Φ)
    w6j   = Wigner6j(j1, i1, f1, i2, F, i)
    kron  = (1 if T == Tp else 0) * (1 if mT == mTp else 0)
    coeff = np.sqrt((2*i + 1) * (2*f1 + 1) * (2*R + 1) / (2*T + 1)) * (-1)
    return float(cg1.doit() * cg2.doit() * w6j.doit() * kron * coeff)


def _epsilon_pi(f1, F, R, T, mT, f1p, mf1, mi2, Rp, mR):
    cg1 = CG(F, mf1 + mi2, R, mR, T, mT)
    cg2 = CG(f1, mf1, i2, mi2, F, mf1 + mi2)
    return float(cg1.doit() * cg2.doit()
                 * (1 if f1 == f1p else 0)
                 * (1 if R  == Rp  else 0))


# ── build or load tables ──────────────────────────────────────────────────────
def build_tables(force_rebuild: bool = False):
    """Return (epsilons, gammalist, pilist, eg_table, ep_table).

    Loads from cache if available and parameters match.
    Set force_rebuild=True to recompute from scratch.
    """
    if not force_rebuild and _CACHE.exists():
        try:
            data = np.load(_CACHE, allow_pickle=True)
            if data["meta"].tolist() == _META:
                print(f"Loaded precomputed tables from {_CACHE.name}")
                return (data["epsilons"], data["gammalist"], data["pilist"],
                        data["eg_table"], data["ep_table"])
            print("Cache metadata mismatch — rebuilding.")
        except Exception as e:
            print(f"Cache unreadable ({e}) — rebuilding.")

    print("Building ε list…")
    eps = []
    for f1_a in np.arange(abs(i1 - J), i1 + J + 1):
        for F_a in np.arange(abs(f1_a - i1), f1_a + i1 + 1):
            for R_a in np.arange(0, 2*(2*i1 + J) + 1, 2):
                for T_a in np.arange(abs(F_a - R_a), F_a + R_a + 1):
                    if 0 < T_a <= 2*i1:
                        eps.append([float(f1_a), float(F_a), float(R_a),
                                    float(T_a), 0.0])
    print(f"  N_ε = {len(eps)}")

    print("Building γ list…")
    gam = []
    for i_a in np.arange(0, 2*i1 + 1):
        for T_a in range(1, int(2*i1) + 1):
            for Ω_a in np.arange(-J, J + 1):
                for ℓ_a in np.arange(-i_a, i_a + 1):
                    for Φ_a in np.arange(-(i1*2 + j1), i1*2 + j1 + 1):
                        if Ω_a + ℓ_a == Φ_a:
                            gam.append([float(Ω_a), float(i_a), float(ℓ_a),
                                        float(T_a), 0.0, float(Φ_a)])
    print(f"  N_γ = {len(gam)}")

    print("Building π list…")
    pi = []
    for f1_a in np.arange(abs(i1 - J), i1 + J + 1):
        for mf1_a in np.arange(-f1_a, f1_a + 1):
            for mi2_a in np.arange(-i1, i1 + 1):
                for R_a in np.arange(0, i1 + J + 1, 2):
                    for mR_a in np.arange(-R_a, R_a + 1):
                        pi.append([float(f1_a), float(mf1_a), float(mi2_a),
                                   float(R_a),  float(mR_a)])
    print(f"  N_π = {len(pi)}")

    print("Computing εγ coupling matrix…")
    eg = [[_gamma_epsilon(*eps[i], *gam[j])
           for j in range(len(gam))]
          for i in tqdm(range(len(eps)), desc="  εγ", unit="ε")]
    print(f"  εγ table: {len(eg)} × {len(eg[0])}")

    print("Computing επ coupling matrix…")
    ep = [[_epsilon_pi(*eps[i], *pi[j])
           for j in range(len(pi))]
          for i in tqdm(range(len(eps)), desc="  επ", unit="ε")]
    print(f"  επ table: {len(ep)} × {len(ep[0])}")

    eps_arr = np.array(eps, dtype=float)
    gam_arr = np.array(gam, dtype=float)
    pi_arr  = np.array(pi,  dtype=float)
    eg_arr  = np.array(eg,  dtype=float)
    ep_arr  = np.array(ep,  dtype=float)

    np.savez_compressed(_CACHE,
                        epsilons=eps_arr,
                        gammalist=gam_arr,
                        pilist=pi_arr,
                        eg_table=eg_arr,
                        ep_table=ep_arr,
                        meta=np.array(_META, dtype=object))
    print(f"Saved → {_CACHE.name}")
    return eps_arr, gam_arr, pi_arr, eg_arr, ep_arr


if __name__ == "__main__":
    epsilons, gammalist, pilist, eg_table, ep_table = build_tables()
    print(f"\nSummary: N_ε={len(epsilons)}, N_γ={len(gammalist)}, "
          f"N_π={len(pilist)}, "
          f"eg_table shape={eg_table.shape}, ep_table shape={ep_table.shape}")
