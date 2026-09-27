"""
Analisis de los neutrinos de SN1987A (poster FNPE, Universidad de Oviedo)
=========================================================================
Dos calculos:
  (1) Energia total y temperatura del estallido de antineutrinos electronicos, ajustando
      un espectro termico a los eventos de Kamiokande-II e IMB (likelihood de Poisson sin
      binear). Formalismo simplificado de Vissani, J. Phys. G 42 (2015) 013001.
  (2) Cota a la masa del neutrino por tiempo de vuelo.

Datos: sn1987a_eventos.csv (Hirata+87, Bionta+87, recopilados en Vissani 2015, Tab. 1).
"""
import numpy as np, pandas as pd
from scipy.special import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- constantes
MeV_erg = 1.602176634e-6          # 1 MeV en erg
D_kpc   = 50.0                    # distancia a SN1987A (kpc), incertidumbre ~5 %
kpc_cm  = 3.0857e21
D_cm    = D_kpc * kpc_cm
Delta   = 1.293                   # m_n - m_p (MeV)
m_e, m_p = 0.511, 938.272         # MeV

# ----------------------------------------------------------------------------- datos
ev = pd.read_csv("sn1987a_eventos.csv", comment="#")
KII = ev[(ev.detector == "KII") & (ev.E_MeV >= 7.5)].copy()   # 11 eventos "canonicos"
IMB = ev[ev.detector == "IMB"].copy()                          # 8 eventos

# Parametros de cada detector (Vissani 2015, Tab. 2)
DET = {
    "KII": dict(Np=1.4e32, Emin=7.5, s_stat=1.27, s_syst=1.0, B30=0.55, data=KII),
    "IMB": dict(Np=4.6e32, Emin=15.0, s_stat=3.0, s_syst=0.4, B30=0.01, data=IMB),
}
T_WINDOW = 30.0   # s, ventana temporal de analisis (para el fondo)

# ----------------------------------------------------------------------------- fisica
def fluence(Enu, Etot_erg, T):
    """dF/dE_nu [cm^-2 MeV^-1]: espectro termico normalizado a energia total Etot en nu_e-bar.
    <E> = 3T.  (Vissani 2015, ec. 1)"""
    Etot = Etot_erg / MeV_erg
    return Etot / (4 * np.pi * D_cm**2) * Enu**2 * np.exp(-Enu / T) / (6 * T**4)

def sigma_ibd(Enu):
    """Seccion eficaz nu_e-bar + p -> e+ + n  [cm^2]  (Strumia & Vissani 2003, ec. 24-25)"""
    Enu = np.asarray(Enu, float)
    Ee = Enu - Delta
    pe = np.sqrt(np.clip(Ee**2 - m_e**2, 0, None))
    x = np.log(np.where(Enu > 0, Enu, 1e-9))
    kappa = np.exp(-0.07056 * x + 0.02018 * x**2 - 0.001953 * x**4) * 1e-43
    return np.where(Ee > m_e, kappa * pe * Ee, 0.0)

def Enu_of_Ee(Ee):
    return (Ee + Delta) / (1 - Ee / m_p)

def jacobian(Enu):
    return (1 + Enu / m_p) ** 2 / (1 + Delta / m_p)

def eta(det, Ee):
    """Eficiencia intrinseca del detector (Vissani 2015, ec. 14 y 16)"""
    Ee = np.asarray(Ee, float)
    if det == "KII":
        return np.where(Ee > 2.6, 0.93 * (1 - 0.2 / Ee - (2.5 / Ee) ** 2), 0.0).clip(0, 1)
    if det == "IMB":
        y = Ee / 15.0 - 1
        val = 0.369 * y - 6e-4 * y**4 + 1e-4 * y**5
        return np.where(Ee > 15.0, val, 0.0).clip(0, 1)

def sigmaE(det, Ee):
    p = DET[det]
    return p["s_stat"] * np.sqrt(Ee / 10) + p["s_syst"] * (Ee / 10)

def g_thr(det, Ee):
    """Probabilidad de que un e+ de energia real Ee se reconstruya por encima del umbral"""
    return 0.5 * (1 + erf((Ee - DET[det]["Emin"]) / (np.sqrt(2) * sigmaE(det, Ee))))

def dS_dEe(det, Ee, Etot, T):
    """Espectro de positrones producidos [MeV^-1] (Vissani 2015, ec. 23)"""
    Enu = Enu_of_Ee(Ee)
    return DET[det]["Np"] * fluence(Enu, Etot, T) * sigma_ibd(Enu) * jacobian(Enu)

Ee_grid = np.linspace(1.0, 90.0, 1800)

def n_expected(det, Etot, T):
    """Numero esperado de eventos de senal por encima del umbral"""
    integrand = eta(det, Ee_grid) * g_thr(det, Ee_grid) * dS_dEe(det, Ee_grid, Etot, T)
    return np.trapezoid(integrand, Ee_grid)

def dN_obs(det, Ei, Etot, T):
    """Densidad de eventos esperada a la energia observada Ei (senal convolucionada con la resolucion)"""
    s = sigmaE(det, Ee_grid)
    G = np.exp(-(Ee_grid - Ei) ** 2 / (2 * s**2)) / (np.sqrt(2 * np.pi) * s)
    return np.trapezoid(eta(det, Ee_grid) * G * dS_dEe(det, Ee_grid, Etot, T), Ee_grid)

def minus2lnL(Etot, T, background=True):
    tot = 0.0
    for det, p in DET.items():
        d = p["data"]
        Ntot = n_expected(det, Etot, T) + (p["B30"] if background else 0.0)
        tot += Ntot
        for Ei, Bi in zip(d.E_MeV, d.fondo_Hz_MeV):
            b = (Bi * T_WINDOW if (background and np.isfinite(Bi)) else 0.0)
            tot -= np.log(dN_obs(det, Ei, Etot, T) + b)
    return 2 * tot

# ----------------------------------------------------------------------------- (1) ajuste
E_grid = np.linspace(1e52, 2.0e53, 120)      # erg (solo nu_e-bar)
T_grid = np.linspace(2.0, 8.0, 121)          # MeV
chi2 = np.array([[minus2lnL(E, T) for T in T_grid] for E in E_grid])
iE, iT = np.unravel_index(np.argmin(chi2), chi2.shape)
E_best, T_best, chi2_min = E_grid[iE], T_grid[iT], chi2[iE, iT]
dchi2 = chi2 - chi2_min

def interval(axis_vals, prof):
    ok = axis_vals[prof <= 1.0]
    return ok.min(), ok.max()
E_lo, E_hi = interval(E_grid, dchi2.min(axis=1))
T_lo, T_hi = interval(T_grid, dchi2.min(axis=0))

print("=== (1) Ajuste del espectro termico (KII>=7.5 MeV + IMB, con fondo) ===")
print(f"E(nu_e-bar) = {E_best/1e52:.1f} (+{(E_hi-E_best)/1e52:.1f} -{(E_best-E_lo)/1e52:.1f}) x10^52 erg")
print(f"T           = {T_best:.2f} (+{T_hi-T_best:.2f} -{T_best-T_lo:.2f}) MeV   -> <E_nu> = 3T = {3*T_best:.1f} MeV")
for det in DET:
    print(f"   {det}: observados {len(DET[det]['data'])}, esperados {n_expected(det, E_best, T_best):.1f} (+fondo {DET[det]['B30']})")
E_tot6 = 6 * E_best
print(f"Energia total (6 especies, equiparticion) = {E_tot6/1e53:.2f} x10^53 erg")
N_nue = E_best / MeV_erg / (3 * T_best)
print(f"Numero de nu_e-bar emitidos ~ {N_nue:.2e};  fluencia en la Tierra ~ {N_nue/(4*np.pi*D_cm**2):.2e} cm^-2")

# Energia de ligadura gravitatoria de la estrella de neutrones (Lattimer & Prakash 2001)
G, c = 6.674e-8, 2.998e10
Msun = 1.989e33
for M, R in [(1.4, 12e5), (1.4, 10e5), (1.6, 12e5)]:
    beta = G * M * Msun / (R * c**2)
    Eb = 0.6 * beta / (1 - 0.5 * beta) * M * Msun * c**2
    Eb_newt = 0.6 * G * (M * Msun) ** 2 / R
    print(f"E_b(M={M} Msun, R={R/1e5:.0f} km): relativista {Eb/1e53:.2f}e53 erg, newtoniana 3GM^2/5R = {Eb_newt/1e53:.2f}e53 erg")

# ----------------------------------------------------------------------------- (2) masa del neutrino
# Retraso por masa respecto a un neutrino sin masa: dt = (L/2c) (m c^2/E)^2
L_over_c = D_cm / c                      # s
K = L_over_c / 2 * 1e-12                 # s * MeV^2 / eV^2   (dt = K * m[eV]^2 / E[MeV]^2)
print(f"\n=== (2) Masa del neutrino ===  L/c = {L_over_c:.3e} s ({L_over_c/3.156e7:.0f} anos);  dt = {K:.2f} s x (m/eV)^2 / (E/MeV)^2")

# (a) cota "de libro": todos los eventos de KII (7.5-35 MeV) llegan en <= 12.4 s
dt_burst = KII.t_rel_ms.max() / 1000
E1, E2 = KII.E_MeV.min(), KII.E_MeV.max()
m_a = np.sqrt(dt_burst / (K * (1 / E1**2 - 1 / E2**2)))
print(f"(a) Duracion observada {dt_burst:.1f} s con E entre {E1} y {E2} MeV  ->  m < {m_a:.1f} eV")

# (b) dispersion de los tiempos de emision inferidos en funcion de m
def t_emision(m, d):
    return d.t_rel_ms.values / 1000 - K * m**2 / d.E_MeV.values**2
ms = np.linspace(0, 30, 301)
spread = np.array([np.ptp(t_emision(m, KII)) for m in ms])      # duracion (max-min)
std    = np.array([np.std(t_emision(m, KII)) for m in ms])
m_b = ms[np.argmax(spread > 2 * spread[0])]
print(f"(b) Duracion inferida de la emision: {spread[0]:.1f} s para m=0; se duplica para m = {m_b:.1f} eV")

# ----------------------------------------------------------------------------- figuras
# Estilo "revista": serif STIX, ticks hacia dentro, sin rejilla, paleta del poster
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14.5, "legend.fontsize": 11.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "axes.grid": False, "savefig.dpi": 220,
})
NAVY, AMBER, TEAL, GREY = "#0f1b3d", "#d9902a", "#2a7f8f", "#7a7f8c"

# --- Figura 1: los 24 eventos
fig, ax = plt.subplots(figsize=(8, 4.4))
for det, mk, c, lab in [("KII", "o", NAVY, "Kamiokande-II"), ("IMB", "s", AMBER, "IMB"), ("Baksan", "^", TEAL, "Baksan")]:
    d = ev[(ev.detector == det) & (ev.E_MeV >= 7.5)]
    ax.errorbar(d.t_rel_ms / 1000, d.E_MeV, yerr=d.dE_MeV, fmt=mk, ms=6.5, color=c, mec="k", mew=0.5,
                ecolor=c, elinewidth=0.9, capsize=2, label=f"{lab} ({len(d)} ev.)")
ax.axhline(7.5, color=NAVY, ls=":", lw=1); ax.text(3.2, 8.3, "umbral KII (7.5 MeV)", color=NAVY, fontsize=10.5)
ax.axhline(15, color=AMBER, ls=":", lw=1); ax.text(3.2, 15.8, "umbral IMB (15 MeV)", color=AMBER, fontsize=10.5)
ax.set_xlim(-0.5, 14); ax.set_ylim(0, 50)
ax.set_xlabel("tiempo desde el primer evento de cada detector (s)")
ax.set_ylabel(r"energía visible del e$^+$ (MeV)")
ax.set_title("SN1987A, 23-II-1987, 7:35 UT: 24 antineutrinos en 13 s")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig("fig1_eventos.png")

# --- Figura 2: region permitida (E,T) + espectros ajustados, un panel por detector
fig = plt.figure(figsize=(10.5, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], hspace=0.08, wspace=0.28)
a1 = fig.add_subplot(gs[:, 0]); a2 = fig.add_subplot(gs[0, 1]); a3 = fig.add_subplot(gs[1, 1], sharex=a2)
a1.contourf(T_grid, E_grid / 1e52, dchi2, levels=[0, 2.30, 6.18], colors=["#7fb3d5", "#d6eaf8"])
a1.contour(T_grid, E_grid / 1e52, dchi2, levels=[2.30, 6.18], colors="k", linewidths=0.8)
a1.plot(T_best, E_best / 1e52, "r*", ms=13, label="mejor ajuste")
a1.set_xlabel(r"$T$ (MeV)"); a1.set_ylabel(r"$W$ ($10^{52}$ erg)")
a1.set_title("Región permitida (68 % y 95 % C.L.)"); a1.legend(); a1.grid(alpha=0.3)
a1.text(0.03, 0.95, "(a)", transform=a1.transAxes, va="top", fontweight="bold")

Eo = np.linspace(5, 70, 300)
bins = np.arange(5, 50, 5)
for ax, det, col, lab, tag in [(a2, "KII", NAVY, "Kamiokande-II", "(b)"), (a3, "IMB", AMBER, "IMB", "(c)")]:
    d = DET[det]["data"]
    spec = np.array([dN_obs(det, e, E_best, T_best) for e in Eo])
    ax.hist(d.E_MeV, bins=bins, histtype="stepfilled", color=col, alpha=0.22, weights=np.full(len(d), 1 / 5))
    ax.hist(d.E_MeV, bins=bins, histtype="step", color=col, lw=1.4, weights=np.full(len(d), 1 / 5), label=f"observado ({len(d)} ev.)")
    ax.plot(Eo, spec, color=col, lw=1.8, label="ajuste")
    ax.set_xlim(5, 70); ax.set_ylim(0, 1.15)
    ax.set_ylabel("eventos / MeV"); ax.legend(loc="upper right", title=lab, title_fontsize=11.5, fontsize=11, handlelength=1.4, borderaxespad=0.3)
    ax.text(0.03, 0.93, tag, transform=ax.transAxes, va="top", fontweight="bold")
a2.set_title("Espectro observado vs. ajuste"); a2.tick_params(labelbottom=False)
a3.set_xlabel(r"energía visible del e$^+$ (MeV)")
fig.savefig("fig2_ajuste_espectro.png", bbox_inches="tight")

# --- Figura 3: masa del neutrino
fig, (b1, b2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
Ecurve = np.linspace(6, 45, 300)
for m, ls in [(0, "-"), (10, "--"), (20, ":")]:
    b1.plot(K * m**2 / Ecurve**2, Ecurve, color=GREY, ls=ls, lw=1.3, label=f"$m$ = {m} eV")
b1.errorbar(KII.t_rel_ms / 1000, KII.E_MeV, yerr=KII.dE_MeV, fmt="o", ms=6, color=NAVY, mec="k", mew=0.5,
            ecolor=NAVY, elinewidth=0.9, capsize=2, label="Kamiokande-II")
b1.errorbar(IMB.t_rel_ms / 1000, IMB.E_MeV, yerr=IMB.dE_MeV, fmt="s", ms=6, color=AMBER, mec="k", mew=0.5,
            ecolor=AMBER, elinewidth=0.9, capsize=2, label="IMB")
b1.set_xlim(-0.5, 14); b1.set_ylim(0, 50)
b1.set_xlabel("tiempo de llegada (s)"); b1.set_ylabel(r"$E$ (MeV)")
b1.set_title("Retraso esperado si la emisión fuese instantánea")
b1.legend(loc="upper right", ncol=2, columnspacing=1.0)
b1.text(0.03, 0.95, "(a)", transform=b1.transAxes, va="top", fontweight="bold")

b2.fill_between(ms, 0, 45, where=ms >= m_a, color="#e8e2d6", lw=0)
b2.plot(ms, spread, color=NAVY, lw=1.8, label="duración inferida (máx − mín)")
b2.plot(ms, std, color=AMBER, lw=1.8, label="desviación típica")
b2.axhline(dt_burst, color=GREY, ls="--", lw=1.1)
b2.text(0.5, dt_burst - 2.6, f"duración observada, {dt_burst:.1f} s", color=GREY, fontsize=11)
b2.text(m_a + 0.7, 40, f"excluido:\n$m$ > {m_a:.0f} eV", color=NAVY, fontsize=11.5, va="top")
b2.set_xlim(0, 30); b2.set_ylim(0, 45)
b2.set_xlabel(r"masa del neutrino $m$ (eV)"); b2.set_ylabel("dispersión de los tiempos de emisión (s)")
b2.set_title("Kamiokande-II: emisión corregida por tiempo de vuelo")
b2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.78))
b2.text(0.03, 0.95, "(b)", transform=b2.transAxes, va="top", fontweight="bold")
fig.tight_layout(w_pad=2.5); fig.savefig("fig3_masa_neutrino.png")

print("\nFiguras: fig1_eventos.png, fig2_ajuste_espectro.png, fig3_masa_neutrino.png")
