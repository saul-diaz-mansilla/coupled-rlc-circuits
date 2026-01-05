import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Parámetros del experimento
R = 1700
L = 1
C = 1e-7
C3_val = np.linspace(1e-8, 1e-7, 10)
V0 = 10.3
w1 = np.sqrt(1 / (L * C))
beta = R / (2 * L)
f0 = 14.1184

def G(w,wn,beta):
    return 1/np.sqrt((w**2-wn**2)**2+(2*beta*w)**2)
def phi(w,wn,beta):
    val = np.arctan2(-2*beta*w,w**2-wn**2)
    return val + np.pi*(val<0)

def A1(w,w1,w2,beta,V0):
    G1 = G(w,w1,beta)
    G2 = G(w,w2,beta)
    phi1 = phi(w,w1,beta)
    phi2 = phi(w,w2,beta)
    return V0/2/L/C*np.sqrt(G1**2+G2**2+2*G1*G2*np.cos(phi1-phi2))

def A2(w,w1,w2,beta,V0):
    G1 = G(w,w1,beta)
    G2 = G(w,w2,beta)
    phi1 = phi(w,w1,beta)
    phi2 = phi(w,w2,beta)
    return V0/2/L/C*np.sqrt(G1**2+G2**2-2*G1*G2*np.cos(phi1-phi2))

def fit_1(f, a, w1, w2, beta, V0):
    n = f / f0
    bound = -50
    val = 20*np.log10(A1(2*np.pi*f,w1,w2,beta,V0) / n) + a
    return val * (val > bound) + bound * (val <= bound)

def fit_2(f, a, w1, w2, beta, V0):
    n = f / f0
    bound = -50
    val = 20*np.log10(A2(2*np.pi*f,w1,w2,beta,V0) / n) + a
    return val * (val > bound) + bound * (val <= bound)

w1_c1 = []
dw1_c1 = []
w2_c1 = []
dw2_c1 = []
w1_c2 = []
dw1_c2 = []
w2_c2 = []
dw2_c2 = []
for i in range(10): # range(10)
    C3 = C3_val[i]
    w2 = np.sqrt(1 / (L * C) + 2 / (L * C3))
    number = str(i).zfill(4)
    # Data import
    df = pd.read_csv(r"C:\Users\Usuario\Documents\Documentos\Universidad\Tecnicas experimentales II\Proyecto\furier_c1\ALL"+number+r"\F"+number+r"FFT.CSV", header=None)
    df.columns = ["Metadata1", "Metadata2", "Empty", "X_Values", "Y_Values", "Empty2"]

    f1_exp = df["X_Values"].to_numpy()
    db1_exp = df["Y_Values"].to_numpy()
    
    df = pd.read_csv(r"C:\Users\Usuario\Documents\Documentos\Universidad\Tecnicas experimentales II\Proyecto\furier_c2\ALL"+number+r"\F"+number+r"FFT.CSV", header=None)
    df.columns = ["Metadata1", "Metadata2", "Empty", "X_Values", "Y_Values", "Empty2"]

    f2_exp = df["X_Values"].to_numpy()
    db2_exp = df["Y_Values"].to_numpy()

    n_points = 2**12
    f1max = f1_exp[-1] # 5045.56906
    f2max = f2_exp[-1] # 5045.56906
    f = np.linspace(0, f1max, n_points//2)

    peaks = find_peaks(db1_exp, distance=10)[0]
    f1_peaks = f1_exp[peaks]
    V1_peaks = db1_exp[peaks]

    mask = f1_peaks < (2/3 * f1max)
    f1_peaks = f1_peaks[mask]
    V1_peaks = V1_peaks[mask]

    peaks = find_peaks(db2_exp, distance=10)[0]
    f2_peaks = f2_exp[peaks]
    V2_peaks = db2_exp[peaks]

    mask = f2_peaks < (2/3 * f2max)
    f2_peaks = f2_peaks[mask]
    V2_peaks = V2_peaks[mask]

    plt.plot(f1_peaks, V1_peaks, 'o', label='Peaks')
    # plt.plot(f, fit_1(f, a_fit, w1_fit, w2_fit, beta_fit, V0_fit), label='Fit')
    plt.plot(f, fit_1(f, -5, w1, w2, beta, V0), label='Teorico')
    plt.title("Ajuste")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Ampliitud (dB)")
    plt.grid()
    plt.legend()
    plt.show()

    parameters, covariance = curve_fit(fit_1, f1_peaks, V1_peaks, p0=[-5, w1, w2, beta, V0])
    a_fit, w1_fit, w2_fit, beta_fit, V0_fit = parameters
    a_err, w1_err, w2_err, beta_err, V0_err = np.sqrt(np.diag(covariance))

    w1_c1.append(w1_fit)
    dw1_c1.append(w1_err)
    w2_c1.append(w2_fit)
    dw2_c1.append(w2_err)

    parameters, covariance = curve_fit(fit_2, f2_peaks, V2_peaks, p0=[-5, w1, w2, beta, V0])
    a_fit, w1_fit, w2_fit, beta_fit, V0_fit = parameters
    a_err, w1_err, w2_err, beta_err, V0_err = np.sqrt(np.diag(covariance))

    w1_c2.append(w1_fit)
    dw1_c2.append(w1_err)
    w2_c2.append(w2_fit)
    dw2_c2.append(w2_err)

    # plt.plot(f_peaks, V_peaks, 'o', label='Peaks')
    # plt.plot(f, fit_2(f, a_fit, w1_fit, w2_fit, beta_fit, V0_fit), label='Fit')
    # plt.plot(f, fit_2(f, -5, w1, w2, beta, V0), label='Teorico')
    # plt.title("Ajuste")
    # plt.xlabel("Frecuencia (Hz)")
    # plt.ylabel("Ampliitud (dB)")
    # plt.grid()
    # plt.legend()
    # plt.show()

plt.figure(figsize=(12,5))

plt.subplot(121)
plt.errorbar(C3_val, w1_c1, yerr=dw1_c1, fmt='o', label=r'Datos $C_1$')
plt.errorbar(C3_val, w1_c2, yerr=dw1_c2, fmt='o', label=r'Datos $C_2$')
# plt.plot(C3_val, w1_c2, 'o', label='Experimental')
plt.plot(C3_val, np.sqrt(1/L/C)*np.ones_like(C3_val), label='Teórico')
plt.xlabel(r"$C_3$ (F)")
plt.ylabel(r"$\omega_1 (rad/s)$")
plt.title(r"Resonancia $\omega_1$")
plt.grid()
plt.legend()

C3_plot = np.linspace(1e-8, 1e-7, 100)
plt.subplot(122)
plt.errorbar(C3_val, w2_c1, yerr=dw2_c1, fmt='o', label=r'Datos $C_1$')
plt.errorbar(C3_val, w2_c2, yerr=dw2_c2, fmt='o', label=r'Datos $C_2$')
# plt.plot(C3_val, w2_c2, 'o', label='Experimental')
plt.plot(C3_plot, np.sqrt(1/L/C + 2/L/C3_plot), label='Teórico')
plt.xlabel(r"$C_3$ (F)")
plt.ylabel(r"$\omega_2 (rad/s)$")
plt.title(r"Resonancia $\omega_2$")
plt.grid()
plt.legend()

plt.show()
