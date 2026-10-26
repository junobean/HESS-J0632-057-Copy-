"""Orbits of HESS J0632+057 for the Casares et al. (2012) and Moritani et al. (2018)
solutions (values from Moritani et al. 2018, Table 1), with the four HAWC phase bins.

Phase zero is JD 2454857.5 (MJD 54857.0) in both papers and in this analysis.
omega is the argument of periastron of the Be star; the compact object has omega + 180 deg.
The orbit is drawn in the orbital plane with the observer towards -y.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORBITS = {
    "Casares et al. (2012)": {"e": 0.83, "omega_be": 129.0, "phi_peri": 0.967},
    "Moritani et al. (2018)": {"e": 0.64, "omega_be": 271.0, "phi_peri": 0.663},
}
STATES = [  # (label, phi_min, phi_max, color)
    ("First high", 0.2, 0.4, "#d62728"),
    ("Low", 0.4, 0.6, "#1f77b4"),
    ("Second high", 0.6, 0.8, "#ff7f0e"),
    ("Medium", 0.8, 1.2, "#2ca02c"),
]


MARKERS = {"Periastron": "^", "Apastron": "v", "Superior conjunction": "s", "Inferior conjunction": "D"}


def true_anomaly(phi, e, phi_peri):
    m = 2 * np.pi * ((np.asarray(phi) - phi_peri) % 1.0)
    ecc = m.copy()
    for _ in range(100):
        ecc -= (ecc - e * np.sin(ecc) - m) / (1 - e * np.cos(ecc))
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(ecc / 2), np.sqrt(1 - e) * np.cos(ecc / 2))
    return nu, 1 - e * np.cos(ecc)


def position(phi, p):
    nu, r = true_anomaly(phi, p["e"], p["phi_peri"])
    theta = nu + np.radians(p["omega_be"] + 180.0)
    return r * np.cos(theta), r * np.sin(theta), r, nu


def phase_of_nu(nu_target, p):
    phi = np.linspace(0, 1, 200001)
    nu, _ = true_anomaly(phi, p["e"], p["phi_peri"])
    d = np.angle(np.exp(1j * (nu - nu_target)))
    return phi[np.argmin(np.abs(d))]


def key_phases(p):
    w_co = np.radians(p["omega_be"] + 180.0)
    return {
        "Periastron": p["phi_peri"] % 1,
        "Apastron": phase_of_nu(np.pi, p),
        "Superior conjunction": phase_of_nu(np.pi / 2 - w_co, p),
        "Inferior conjunction": phase_of_nu(3 * np.pi / 2 - w_co, p),
    }


fig, axes = plt.subplots(1, 2, figsize=(11, 6.4))
for ax, (name, p) in zip(axes, ORBITS.items()):
    print(f"== {name}")
    full = np.linspace(0, 1, 2000)
    x, y, _, _ = position(full, p)
    ax.plot(x, y, color="0.75", lw=1, zorder=1)
    for label, lo, hi, c in STATES:
        ph = np.linspace(lo, hi, 400)
        x, y, r, _ = position(ph % 1, p)
        ax.plot(x, y, color=c, lw=4, label=f"{label} ($\\phi$={lo:.1f}–{hi % 1:.1f})", zorder=2)
        print(f"  {label:12s} separation r/a = {r.min():.2f}-{r.max():.2f}")
    for ph in np.arange(0, 1, 0.1):
        x, y, _, _ = position(ph, p)
        ax.plot(x, y, "k+", ms=7, zorder=3)
        u = np.hypot(x, y)
        ax.annotate(f"{ph:.1f}", (x, y), textcoords="offset points", xytext=(14 * x / u - 6, 14 * y / u - 4), fontsize=8)
    for k, ph in key_phases(p).items():
        x, y, _, _ = position(ph, p)
        ax.plot(x, y, MARKERS[k], color="k", mfc="white", ms=9, mew=1.3, zorder=4, label=k)
        print(f"  {k:12s} phase = {ph:.3f}")
    ax.plot(0, 0, "*", color="gold", mec="k", ms=16, zorder=5)
    ax.annotate("", xy=(0, -2.05), xytext=(0, -1.75), arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(0.05, -1.95, "To observer", fontsize=8)
    ax.set_title(f"{name}\n$e$={p['e']}, $\\omega$={p['omega_be']:.0f}$^\\circ$, $\\phi_{{\\rm peri}}$={p['phi_peri']}")
    ax.set_aspect("equal")
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.set_xlabel("$x$ / $a$")
    ax.set_ylabel("$y$ / $a$")
    ax.grid(alpha=0.3)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(sys.argv[1], dpi=200)
