"""
plot_adiabatic_curves.py — visualise the Sr87 adiabatic potential curves
(same plot as produced at the end of compute_adiabatic_curves.py).

Reloads the precomputed .npz file (no recomputation) from the npz_data/
folder of the parent directory:
    ../npz_data/adiabatic_curves_physical.npz   (required)

Every curve (out of ~980) is drawn, clipped to the requested energy window,
so only the segments actually inside the window are plotted.

Selecting one dissociation limit
--------------------------------
--f1 keeps only the curves that dissociate to one hyperfine limit of the
excited atom, f1 = 11/2, 9/2 or 7/2.  The limits are found from the data, not
from hard-coded energies: see _asymptote_groups.  Without the option nothing
is computed or printed, so the default plot is unchanged.

Usage
-----
    python3 plot_adiabatic_curves.py                    # default window
    python3 plot_adiabatic_curves.py --emin -10 --emax 10
    python3 plot_adiabatic_curves.py --rmin 5 --rmax 200
    python3 plot_adiabatic_curves.py --f1 11/2          # lowest limit only
    python3 plot_adiabatic_curves.py --save figure.png   # save instead of show
"""

import os
import argparse
import numpy as np


HERE    = os.path.dirname(os.path.abspath(__file__))
PARENT  = os.path.dirname(HERE)
NPZ_DIR = os.path.join(PARENT, "npz_data")   # all Sr87 .npz live here

N_TAIL   = 50         # points at large R used to estimate the asymptote
# Gap in the sorted asymptotes that separates two hyperfine limits.  The limits
# are 1.1-1.5 GHz apart, while the channels of a single limit are spread over
# at most ~60 MHz by the centrifugal term, so anything in between works.
ASYM_GAP = 0.25       # GHz
F1_LABELS = ("11/2", "9/2", "7/2")   # hyperfine limits of 3P1, lowest first


def _asymptote_groups(asym, gap=ASYM_GAP):
    """Split the curves into their hyperfine dissociation limits.

    A limit is a *band*, not a single energy: the centrifugal term
    hbar^2 R(R+1)/(2 mu r^2) spreads the channels of one limit over ~60 MHz at
    the end of the R grid, while two limits sit 1.1-1.5 GHz apart.  Cutting the
    sorted asymptotes wherever they jump by more than `gap` therefore recovers
    the limits from the data alone, without duplicating the hyperfine constants
    of compute_adiabatic_curves.py here.

    Returns a list of index arrays, lowest limit first.  For 87Sr 3P1 the
    magnetic-dipole constant A is negative, so the lowest limit is f1 = 11/2.

    (Same helper as in plot_potentials_levels.py — the repo keeps its scripts
    self-contained rather than sharing a module.)
    """
    order = np.argsort(asym)
    cuts = np.where(np.diff(asym[order]) > gap)[0] + 1
    return [np.sort(g) for g in np.split(order, cuts)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rmin", type=float, default=None,  help="R window min (a0)")
    ap.add_argument("--rmax", type=float, default=None,  help="R window max (a0)")
    ap.add_argument("--emin", type=float, default=-5.0,  help="energy window min (GHz)")
    ap.add_argument("--emax", type=float, default=5.0,   help="energy window max (GHz)")
    ap.add_argument("--f1", choices=["all", *F1_LABELS], default="all",
                    help="keep only the curves dissociating to this hyperfine "
                         "limit of the excited atom (default: all)")
    ap.add_argument("--save", default=None,
                    help="save figure to this path instead of showing it")
    args = ap.parse_args()

    path = os.path.join(NPZ_DIR, "adiabatic_curves_physical.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run compute_adiabatic_curves.py first.")
    data = np.load(path)
    r_grid   = data["r_tab"]
    eig_phys = data["eig_physical"]
    n_eig    = eig_phys.shape[1]

    rmask = np.ones_like(r_grid, dtype=bool)
    if args.rmin is not None:
        rmask &= r_grid >= args.rmin
    if args.rmax is not None:
        rmask &= r_grid <= args.rmax

    # nothing below runs unless --f1 is given, so the default plot is untouched
    curves = range(n_eig)
    if args.f1 != "all":
        asym = eig_phys[-N_TAIL:, :].mean(axis=0)
        groups = _asymptote_groups(asym)
        print(f"  {len(groups)} dissociation limit(s) found in the data:")
        for lab, g in zip(F1_LABELS, groups):
            print(f"    f1 = {lab:>4} : {len(g):3d} curves, E_inf in "
                  f"[{asym[g].min():.4f}, {asym[g].max():.4f}] GHz")
        if len(groups) > len(F1_LABELS):
            print(f"    WARNING: {len(groups)} groups for {len(F1_LABELS)} "
                  f"expected limits — a limit may be split; check ASYM_GAP.")
        idx = F1_LABELS.index(args.f1)
        if idx >= len(groups):
            raise SystemExit(f"limit f1 = {args.f1} is not present in the data "
                             f"(only {len(groups)} group(s) found).")
        curves = groups[idx]
        print(f"  --f1 {args.f1}: {len(curves)} curves kept")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in curves:
        col = eig_phys[:, k]
        mask = rmask & (col >= args.emin) & (col <= args.emax)
        if mask.any():
            ax.plot(r_grid[mask], col[mask], lw=0.6)

    ax.set_xlabel("R (a₀)")
    ax.set_ylabel("Energy (GHz)")
    title = "Physical adiabatic curves — Sr87"
    if args.f1 != "all":
        title += f"  (limit f1 = {args.f1}, {len(curves)} curves)"
    ax.set_title(title)
    if args.rmin is not None or args.rmax is not None:
        ax.set_xlim(args.rmin, args.rmax)
    ax.set_ylim(args.emin, args.emax)
    ax.axhline(0, color="k", lw=0.4, ls="--")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=120)
        print(f"  saved -> {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
