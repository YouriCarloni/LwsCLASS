import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

files = {
    r'$\Lambda_{\omega_s}$CDM': ("/home/carloni/class_public-master3/output/default00_pk.dat", 0.7151),
    r'$\Lambda$CDM':           ("/home/carloni/class_public-master3/output/default01_pk.dat", 0.688),
}

output_fig = "Pk2.pdf"

plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.0,
    "mathtext.default": "it",
    "mathtext.fontset": "dejavuserif",
})

styles = {
    r'$\Lambda_{\omega_s}$CDM': dict(color='black', linestyle='-', linewidth=1.8),
    r'$\Lambda$CDM':            dict(color='C0', linestyle=':', linewidth=2.2),
}

def load_pk(path, hval, pk_col=1):
    data = np.loadtxt(path, comments="#")
    k = data[:, 0]
    Pk = data[:, pk_col]

    idx = np.argsort(k)
    k = k[idx]
    Pk = Pk[idx]

    k_phys = k * hval
    P_phys = Pk / hval**3

    return k_phys, P_phys

def top_hat_window(x):
    return 3.0 * (np.sin(x) - x * np.cos(x)) / x**3

def sigma8_from_pk(k_phys, P_phys, hval):
    """
    sigma_8 is defined on R = 8 Mpc/h.
    Since we are using physical units (Mpc, not Mpc/h), use R = 8/h Mpc.
    """
    R8 = 8.0 / hval  # Mpc
    x = k_phys * R8
    W = top_hat_window(x)
    integrand = k_phys**2 * P_phys * W**2
    sigma2 = np.trapz(integrand, k_phys) / (2.0 * np.pi**2)
    return np.sqrt(sigma2)

fig, ax = plt.subplots(figsize=(6.2, 3.3))

for label, (path, hval) in files.items():
    k, Pk = load_pk(path, hval)
    sigma8 = sigma8_from_pk(k, Pk, hval)

    ax.semilogx(k, Pk, label=label, **styles[label])

    print(f"{Path(path).name}")
    print(f"  h       = {hval}")
    print(f"  sigma_8 = {sigma8:.4f}")
    print(f"  max P(k)= {Pk.max():.3g}\n")

ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
ax.set_ylabel(r"$P(k)\ [{\rm Mpc}^{3}]$")
ax.set_xlim(1e-4, 1e0)
ax.set_ylim(0, 9e4)

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((4, 4))
ax.yaxis.set_major_formatter(formatter)

ax.tick_params(which="both", direction="in", top=True, right=True)
ax.minorticks_on()

leg = ax.legend(frameon=True, loc="upper right")
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(0.8)

plt.tight_layout()
plt.savefig(output_fig, format="pdf", bbox_inches="tight")
print(f"Figure saved to: {output_fig}")
plt.show()

