"""
molecular_potentials.py — Long-range molecular potentials for Sr2.

Provides the four adiabatic potentials needed by the coupled-channel
Hamiltonian:  0u, 1u, 0g, 1g.

Each potential is built from:
  - A short-range part  : sum of exponential + polynomial terms (Morse-like)
  - A long-range part   : damped C6 / C8 / C10 / C12 dispersion series
    (Tang–Toennies damping function)
  - A resonant dipole term : ±C3 / r³  (sign depends on g/u symmetry)

Parameters are fitted to ab initio data for 3Σ_u and 3Π_u states of Sr88.
The C3 constant encodes the resonant dipole interaction at the
5s²→5s5p ³P₁ asymptote.

Run this file directly to plot all four potentials.
Come from the article Mass scaling and non-adiabatic effects in photoassociation spectroscopy of ultracold strontium, by Borkowski et al. (2014).
"""

import math
import numpy as np

# ── unit conversion ───────────────────────────────────────────────────────────
a0      = 5.29177e-11                                      # Bohr radius, m
Hartree = 27.2114 * 2.41799e14 / (1e6 * 1e3)              # Hartree → GHz

# ── short-range parameters for 3Π_u ──────────────────────────────────────────
A0_u_Π =  5.78723038e6
A1_u_Π = -3.46113235e6
A2_u_Π =  7.79019763e5
A3_u_Π = -7.85317879e4
A4_u_Π =  3.01833743e3
γ_u_Π  =  1.34967817e-3
β_u_Π  =  1.03238202
α_u_Π  =  1.99168225    # fitted for Sr88
C12_u_Π = -1.06415514e10
C10_u_Π =  5.24064e7
C8_u_Π  =  3.4156471e5
C6_u_Π  =  3.8683912e3

# ── short-range parameters for 3Σ_u ──────────────────────────────────────────
A0_u_Σ =  1.29406314e2
A1_u_Σ = -7.90551852e1
A2_u_Σ =  1.87863441e1
A3_u_Σ = -1.96979418e0
A4_u_Σ =  7.88636443e-2
γ_u_Σ  =  7.61382806e-2
β_u_Σ  =  1.0
α_u_Σ  =  0.045647282   # fitted for Sr88
C12_u_Σ = -5.31841848e9
C10_u_Σ =  2.20495e8
C8_u_Σ  =  2.3574797e6
C6_u_Σ  =  4.3015063e3

# ── resonant dipole coupling constants ───────────────────────────────────────
C3_0u =  1.52356615e-2   # 0u channel
C3_1u = -C3_0u / 2       # 1u channel  (C3_0g = -C3_0u, C3_1g = -C3_1u)


# ── Tang–Toennies damping ─────────────────────────────────────────────────────
def _tang_toennies(r, b, n):
    """1 - exp(-b*r) * sum_{k=0}^{n} (b*r)^k / k!"""
    s = sum((b * r)**k / math.factorial(k) for k in range(n + 1))
    return 1.0 - math.exp(-b * r) * s


# ── electronic state potentials (in units of C6 / Hartree — before conversion) ─
def _potential_3sigma_u(r):
    exp_term = math.exp(-α_u_Σ * r - γ_u_Σ * r**2)
    V  = (A0_u_Σ + A1_u_Σ*r + A2_u_Σ*r**2 + A3_u_Σ*r**3 + A4_u_Σ*r**4) * exp_term
    V -= C6_u_Σ  * _tang_toennies(r, β_u_Σ,  6) / r**6
    V -= C8_u_Σ  * _tang_toennies(r, β_u_Σ,  8) / r**8
    V -= C10_u_Σ * _tang_toennies(r, β_u_Σ, 10) / r**10
    V -= C12_u_Σ * _tang_toennies(r, β_u_Σ, 12) / r**12
    return V


def _potential_3pi_u(r):
    exp_term = math.exp(-α_u_Π * r - γ_u_Π * r**2)
    V  = (A0_u_Π + A1_u_Π*r + A2_u_Π*r**2 + A3_u_Π*r**3 + A4_u_Π*r**4) * exp_term
    V -= C6_u_Π  * _tang_toennies(r, β_u_Π,  6) / r**6
    V -= C8_u_Π  * _tang_toennies(r, β_u_Π,  8) / r**8
    V -= C10_u_Π * _tang_toennies(r, β_u_Π, 10) / r**10
    V -= C12_u_Π * _tang_toennies(r, β_u_Π, 12) / r**12
    return V


# ── public API : four adiabatic potentials (output in GHz) ───────────────────
def potential_0u(r):
    """0u adiabatic potential at internuclear distance r (in a₀). Returns GHz."""
    return (_potential_3pi_u(r) - C3_0u / r**3) * Hartree


def potential_1u(r):
    """1u adiabatic potential at internuclear distance r (in a₀). Returns GHz."""
    return (0.5 * (_potential_3pi_u(r) + _potential_3sigma_u(r)) - C3_1u / r**3) * Hartree


def potential_0g(r):
    """0g adiabatic potential at internuclear distance r (in a₀). Returns GHz."""
    return (_potential_3pi_u(r) + C3_0u / r**3) * Hartree


def potential_1g(r):
    """1g adiabatic potential at internuclear distance r (in a₀). Returns GHz."""
    return (0.5 * (_potential_3pi_u(r) + _potential_3sigma_u(r)) + C3_1u / r**3) * Hartree


# ── Hund (a) electronic potentials (output in GHz) ───────────────────────────
def potential_sigma_u(r):
    """³Σ_u⁺ Hund (a) potential at r (a₀). Returns GHz."""
    return _potential_3sigma_u(r) * Hartree


def potential_pi_u(r):
    """³Π_u Hund (a) potential at r (a₀). Returns GHz."""
    return _potential_3pi_u(r) * Hartree


# ── vectorised wrappers (accept numpy arrays) ─────────────────────────────────
potential_0u_vec    = np.vectorize(potential_0u)
potential_1u_vec    = np.vectorize(potential_1u)
potential_0g_vec    = np.vectorize(potential_0g)
potential_1g_vec    = np.vectorize(potential_1g)
potential_sigma_u_vec = np.vectorize(potential_sigma_u)
potential_pi_u_vec    = np.vectorize(potential_pi_u)


# ── quick visual check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    r = np.linspace(5, 250, 10000)

    fig, axes = plt.subplots(1, 1, figsize=(13, 5))

    axes.plot(r, potential_0u_vec(r), label="0u")
    axes.plot(r, potential_1u_vec(r), label="1u")
    #axes[0].plot(r, potential_0g_vec(r), label="0g")
    #axes[0].plot(r, potential_1g_vec(r), label="1g")
    #axes.plot(r, potential_sigma_u_vec(r), lw=2, label="³Σ_u⁺")
    #axes.plot(r, potential_pi_u_vec(r),  lw=2, label="³Π_u")
    #axes[0].set_ylim(-5, 5)
    axes.set_xlabel("R (a₀)", fontsize=20)
    axes.set_ylabel("V (GHz)", fontsize=20)
    axes.set_title("Adiabatic + Hund (a) potentials — full range", fontsize=22)
    axes.tick_params(axis="both", labelsize=16)
    axes.legend(fontsize=18)
    axes.grid(True, alpha=0.3)
    """
    #axes[1].plot(r, potential_0u_vec(r), label="0u")
    #axes[1].plot(r, potential_1u_vec(r), label="1u")
    #axes[1].plot(r, potential_0g_vec(r), label="0g")
    #axes[1].plot(r, potential_1g_vec(r), label="1g")
    axes[1].plot(r, potential_sigma_u_vec(r), "k--", lw=1, label="³Σ_u⁺")
    axes[1].plot(r, potential_pi_u_vec(r),    "k:",  lw=1, label="³Π_u")
    axes[1].set_xlim(5, 30)
    axes[1].set_ylim(-50, 10)
    axes[1].set_xlabel("R (a₀)")
    axes[1].set_title("Short range")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    """
    plt.tight_layout()
    plt.show()
