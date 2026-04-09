import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

# ---- individual h values for each file ----
files = {
    r'$\Lambda_{\omega_s}$CDM': ("/home/carloni/class_public-master3/output/default00_pk.dat", 0.7151),
    r'$\Lambda$CDM':           ("/home/carloni/class_public-master3/output/default01_pk.dat", 0.688),
}

# ---- output figure (PDF) ----
output_fig = "Pk.pdf"

# ---- plotting style ----
plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.0,
    "mathtext.default": "it",
    "mathtext.fontset": "dejavuserif",
})

styles = {
    r'$\Lambda_{\omega_s}$CDM': dict(color='red', linestyle='-', linewidth=1.8),
    r'$\Lambda$CDM':            dict(color='blue', linestyle=':', linewidth=2.2),
}

def load_pk(path, hval, pk_col=1):
    """
    Load CLASS P(k) and convert:
      k : h/Mpc      ->  1/Mpc
      P : (Mpc/h)^3  ->  Mpc^3
    """
    data = np.loadtxt(path, comments="#")
    k = data[:, 0]
    Pk = data[:, pk_col]

    idx = np.argsort(k)
    k = k[idx]
    Pk = Pk[idx]

    k_phys = k * hval
    P_phys = Pk / hval**3

    return k_phys, P_phys

# ---- plot ----
fig, ax = plt.subplots(figsize=(6.2, 3.3))

for label, (path, hval) in files.items():
    k, Pk = load_pk(path, hval)
    ax.semilogx(k, Pk, label=label, **styles[label])
    print(f"{Path(path).name}: h = {hval}, max P(k) = {Pk.max():.3g}")

# ---- axes ----
ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
ax.set_ylabel(r"$P(k)\ [{\rm Mpc}^{3}]$")
ax.set_xlim(1e-4, 1e0)
ax.set_ylim(0, 9e4)

# ---- scientific notation on y-axis: 1,2,3,... × 10^4 ----
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((4, 4))  # force ×10^4
ax.yaxis.set_major_formatter(formatter)

ax.tick_params(which="both", direction="in", top=True, right=True)
ax.minorticks_on()

leg = ax.legend(frameon=True, loc="upper right")
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(0.8)

plt.tight_layout()

# ---- save figure (PDF, vector) ----
plt.savefig(output_fig, format="pdf", bbox_inches="tight")
print(f"Figure saved to: {output_fig}")

plt.show()

