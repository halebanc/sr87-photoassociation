"""
plot_potentials_levels.py — visualise the Sr87 adiabatic potentials together
with their bound vibrational levels drawn as horizontal lines (same style as
the plot produced by compute_binding_energies_wkb.py).

Reloads the precomputed .npz files (no recomputation) from the npz_data/ folder
of the parent directory:
    ../npz_data/adiabatic_curves_physical.npz   (potentials, required)
    ../npz_data/binding_energies_wkb.npz        (WKB levels)
    ../npz_data/binding_energies_dvr.npz        (DVR levels)

The potentials are the retained adiabatic curves (asymptote E_inf < 1 GHz),
shifted so that each asymptote sits at 0 GHz — exactly the potential fed to the
DVR / WKB solvers.  Each bound level is drawn as a full horizontal line at
E = -E_b (binding energies are stored positive).

Selecting one dissociation limit
--------------------------------
--f1 keeps only the curves that dissociate to one hyperfine limit of the
excited atom, f1 = 11/2, 9/2 or 7/2.  The limits are found from the data, not
from hard-coded energies: see _asymptote_groups.  `--f1 11/2` reproduces
exactly the 375 curves that compute_binding_energies.py feeds to its solvers,
so the bound levels drawn then belong to the potentials drawn.

Usage
-----
    python3 plot_potentials_levels.py                  # both sources (default)
    python3 plot_potentials_levels.py --source wkb     # WKB levels only
    python3 plot_potentials_levels.py --source dvr     # DVR levels only
    python3 plot_potentials_levels.py --f1 11/2        # lowest limit only
    python3 plot_potentials_levels.py --emin -10 --emax 0.5 --rmin 5 --rmax 200
    python3 plot_potentials_levels.py --save figure.png   # save instead of show

If a requested .npz is missing (e.g. the DVR run has not been done yet) the
script warns and continues with whatever is available.
"""

import os
import argparse
import numpy as np


HERE       = os.path.dirname(os.path.abspath(__file__))
PARENT     = os.path.dirname(HERE)
NPZ_DIR    = os.path.join(PARENT, "npz_data")   # all Sr87 .npz live here
ASYM_THRESH = 100.0   # GHz — same selection threshold as the solvers
N_TAIL      = 50    # points used to estimate the asymptote

# Gap in the sorted asymptotes that separates two hyperfine limits.  The limits
# are 1.1-1.5 GHz apart, while the channels of a single limit are spread over
# at most ~60 MHz by the centrifugal term, so anything in between works.
ASYM_GAP = 0.25       # GHz
F1_LABELS = ("11/2", "9/2", "7/2")   # hyperfine limits of 3P1, lowest first

# colour per level source
LEVEL_COLOR = {"wkb": "C0", "dvr": "C3"}


def _load(name):
    """Load an .npz from the npz_data/ folder, or return None if absent."""
    path = os.path.join(NPZ_DIR, name)
    if not os.path.exists(path):
        return None
    return np.load(path)


def _curve_keys(data):
    """Indices k for which the file stores levels (curve_{k}_Eb_MHz)."""
    return sorted(int(key.split("_")[1])
                  for key in data.files
                  if key.startswith("curve_") and key.endswith("_Eb_MHz"))


def _asymptote_groups(asym, gap=ASYM_GAP):
    """Split the curves into their hyperfine dissociation limits.

    A limit is a *band*, not a single energy: the centrifugal term
    hbar^2 R(R+1)/(2 mu r^2) spreads the channels of one limit over ~60 MHz at
    the end of the R grid (0 to 57 MHz for R = 0 to 18 at 503 a0), while two
    limits sit 1.1-1.5 GHz apart.  Cutting the sorted asymptotes wherever they
    jump by more than `gap` therefore recovers the limits from the data alone,
    without duplicating the hyperfine constants of compute_adiabatic_curves.py
    here — so this keeps working if those constants are revised.

    Returns a list of index arrays, lowest limit first.  For 87Sr 3P1 the
    magnetic-dipole constant A is negative, so the lowest limit is f1 = 11/2.
    """
    order = np.argsort(asym)
    cuts = np.where(np.diff(asym[order]) > gap)[0] + 1
    return [np.sort(g) for g in np.split(order, cuts)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["wkb", "dvr", "both"], default="both",
                    help="which bound-level set(s) to overlay (default: both)")
    ap.add_argument("--f1", choices=["all", *F1_LABELS], default="all",
                    help="keep only the curves dissociating to this hyperfine "
                         "limit of the excited atom (default: all). '11/2' is "
                         "the lowest limit, i.e. the 375 curves the binding-"
                         "energy solvers actually use")
    ap.add_argument("--rmin", type=float, default=5.0,   help="R window min (aB)")
    ap.add_argument("--rmax", type=float, default=150.0, help="R window max (aB)")
    ap.add_argument("--emin", type=float, default=-5.0,  help="energy window min (GHz)")
    ap.add_argument("--emax", type=float, default=0.5,   help="energy window max (GHz)")
    ap.add_argument("--save", default=None,
                    help="save figure to this path instead of showing it")
    args = ap.parse_args()

    # ── potentials (required) ─────────────────────────────────────────────────
    adia = _load("adiabatic_curves_physical.npz")
    if adia is None:
        raise FileNotFoundError(
            "adiabatic_curves_physical.npz not found in the parent directory — "
            "run compute_adiabatic_curves.py first.")
    r_tab    = adia["r_tab"]
    eig_phys = adia["eig_physical"]
    asym     = eig_phys[-N_TAIL:, :].mean(axis=0)
    selected = np.where(asym < ASYM_THRESH)[0]

    groups = _asymptote_groups(asym)
    print(f"  {len(groups)} dissociation limit(s) found in the data:")
    for lab, g in zip(F1_LABELS, groups):
        print(f"    f1 = {lab:>4} : {len(g):3d} curves, E_inf in "
              f"[{asym[g].min():.4f}, {asym[g].max():.4f}] GHz")
    if len(groups) > len(F1_LABELS):
        print(f"    WARNING: {len(groups)} groups for {len(F1_LABELS)} expected "
              f"limits — a limit may be split; check ASYM_GAP.")

    if args.f1 != "all":
        idx = F1_LABELS.index(args.f1)
        if idx >= len(groups):
            raise SystemExit(f"limit f1 = {args.f1} is not present in the data "
                             f"(only {len(groups)} group(s) found).")
        selected = np.intersect1d(selected, groups[idx])
        print(f"  --f1 {args.f1}: {len(selected)} curves kept")

    # ── level sources ─────────────────────────────────────────────────────────
    wanted = ["wkb", "dvr"] if args.source == "both" else [args.source]
    sources = {}
    for src in wanted:
        data = _load(f"binding_energies_{src}.npz")
        if data is None:
            print(f"  WARNING: binding_energies_{src}.npz not found — "
                  f"skipping {src.upper()} levels.")
            continue
        sources[src] = data
    if not sources:
        print("  No level file available; plotting bare potentials only.")

    # ── plot ──────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(11, 6))

    # potentials: each retained curve, shifted so its asymptote is 0 (GHz)
    rmask = (r_tab >= args.rmin) & (r_tab <= args.rmax)
    for k in selected:
        V_GHz = eig_phys[:, k] - asym[k]
        ax.plot(r_tab[rmask], V_GHz[rmask], color="0.6", lw=0.4, alpha=0.5)

    # bound levels as full horizontal lines (E_level = -E_b, in GHz)
    # the curve_{k} keys are column indices of eig_physical, so the same
    # selection filters the levels and the potentials they belong to
    sel_set = set(selected.tolist())
    handles = []
    for src, data in sources.items():
        col = LEVEL_COLOR[src]
        n_lines = 0
        for k in _curve_keys(data):
            if k not in sel_set:
                continue
            for Eb_MHz in data[f"curve_{k}_Eb_MHz"]:
                E_level = -Eb_MHz / 1e3
                if args.emin <= E_level <= args.emax:
                    ax.axhline(E_level, color=col, lw=0.3, alpha=0.25)
                    n_lines += 1
        handles.append(plt.Line2D([], [], color=col, lw=1.5,
                                   label=f"{src.upper()} levels ({n_lines})"))
        print(f"  {src.upper()}: {n_lines} level-lines in the energy window")

    ax.axhline(0.0, color="k", lw=0.8, ls="--", label="dissociation threshold")
    ax.set_xlim(args.rmin, args.rmax)
    ax.set_ylim(args.emin, args.emax)
    ax.set_xlabel("R (a0)")
    ax.set_ylabel("Energy (GHz)  [asymptote at 0]")
    which = (f"E_inf < {ASYM_THRESH} GHz" if args.f1 == "all"
             else f"limit f1 = {args.f1}")
    ax.set_title(f"Adiabatic potentials and bound levels — Sr87  "
                 f"({len(selected)} curves, {which})")
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=9)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=120)
        print(f"  saved -> {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
