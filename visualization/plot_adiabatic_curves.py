"""
plot_adiabatic_curves.py — visualise the Sr87 adiabatic potential curves
(same plot as produced at the end of compute_adiabatic_curves.py).

Reloads the precomputed .npz file (no recomputation) from the npz_data/
folder of the parent directory:
    ../npz_data/adiabatic_curves_physical.npz   (required)

Every curve (out of ~980) is drawn, clipped to the requested energy window,
so only the segments actually inside the window are plotted.

Usage
-----
    python3 plot_adiabatic_curves.py                    # default window
    python3 plot_adiabatic_curves.py --emin -10 --emax 10
    python3 plot_adiabatic_curves.py --rmin 5 --rmax 200
    python3 plot_adiabatic_curves.py --save figure.png   # save instead of show
"""

import os
import argparse
import numpy as np


HERE    = os.path.dirname(os.path.abspath(__file__))
PARENT  = os.path.dirname(HERE)
NPZ_DIR = os.path.join(PARENT, "npz_data")   # all Sr87 .npz live here


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rmin", type=float, default=None,  help="R window min (a0)")
    ap.add_argument("--rmax", type=float, default=None,  help="R window max (a0)")
    ap.add_argument("--emin", type=float, default=-5.0,  help="energy window min (GHz)")
    ap.add_argument("--emax", type=float, default=5.0,   help="energy window max (GHz)")
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

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(n_eig):
        col = eig_phys[:, k]
        mask = rmask & (col >= args.emin) & (col <= args.emax)
        if mask.any():
            ax.plot(r_grid[mask], col[mask], lw=0.6)

    ax.set_xlabel("R (a₀)")
    ax.set_ylabel("Energy (GHz)")
    ax.set_title("Physical adiabatic curves — Sr87")
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
