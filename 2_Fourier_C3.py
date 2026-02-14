import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Experiment parameters
R = 1700
L = 1
C = 1e-7
C3_val = np.linspace(1e-8, 1e-7, 10)
V0 = 10.3
w1 = np.sqrt(1 / (L * C))
beta = R / (2 * L)
f0 = 14.1184

# Gain function for RLC circuit
def G(w,wn,beta):
    return 1/np.sqrt((w**2-wn**2)**2+(2*beta*w)**2)

# Phase function for RLC circuit
def phi(w,wn,beta):
    val = np.arctan2(-2*beta*w,w**2-wn**2)
    return val + np.pi*(val<0)

# Amplitude response for capacitor 1 in coupled circuit
def A1(w,w1,w2,beta,V0):
    G1 = G(w,w1,beta)
    G2 = G(w,w2,beta)
    phi1 = phi(w,w1,beta)
    phi2 = phi(w,w2,beta)
    return V0/2/L/C*np.sqrt(G1**2+G2**2+2*G1*G2*np.cos(phi1-phi2))

# Amplitude response for capacitor 2 in coupled circuit
def A2(w,w1,w2,beta,V0):
    G1 = G(w,w1,beta)
    G2 = G(w,w2,beta)
    phi1 = phi(w,w1,beta)
    phi2 = phi(w,w2,beta)
    return V0/2/L/C*np.sqrt(G1**2+G2**2-2*G1*G2*np.cos(phi1-phi2))

# Fit function for capacitor 1 response in dB
def fit_1(f, a, w1, w2, beta, V0):
    n = f / f0
    bound = np.min(decibels)
    val = 20*np.log10(A1(2*np.pi*f,w1,w2,beta,V0) / n) + a
    return val * (val > bound) + bound * (val <= bound)

# Fit function for capacitor 2 response in dB
def fit_2(f, a, w1, w2, beta, V0):
    n = f / f0
    bound = -50
    val = 20*np.log10(A2(2*np.pi*f,w1,w2,beta,V0) / n) + a
    return val * (val > bound) + bound * (val <= bound)

w1_val = []
dw1 = []
w2_val = []
dw2 = []
# Loop over different coupling capacitance values
for i in range(10):
    C3 = C3_val[i]
    w2 = np.sqrt(1 / (L * C) + 2 / (L * C3))
    number = str(i).zfill(4)
    
    # Import data from CSV files
    df = pd.read_csv(r"RLCC_FFT_C2\ALL"+number+r"\F"+number+r"FFT.CSV", header=None)
    df.columns = ["Metadata1", "Metadata2", "Empty", "X_Values", "Y_Values", "Empty2"]

    frequencies = df["X_Values"].to_numpy()
    decibels = df["Y_Values"].to_numpy()

    n_points = 2**12
    fmax = frequencies[-1] # 5045.56906
    f = np.linspace(0, fmax, n_points//2)

    # Find peaks in the frequency spectrum
    peaks = find_peaks(decibels, distance=10)[0]
    f_peaks = frequencies[peaks]
    V_peaks = decibels[peaks]

    # Filter peaks below a certain frequency threshold
    mask = f_peaks < (2/3 * fmax)
    f_peaks = f_peaks[mask]
    V_peaks = V_peaks[mask]

    # Fit the theoretical model to the experimental peaks
    parameters, covariance = curve_fit(fit_2, f_peaks, V_peaks, p0=[-5, w1, w2, beta, V0])
    a_fit, w1_fit, w2_fit, beta_fit, V0_fit = parameters
    a_err, w1_err, w2_err, beta_err, V0_err = np.sqrt(np.diag(covariance))

    print(a_fit)
    
    w1_val.append(w1_fit)
    dw1.append(w1_err)
    w2_val.append(w2_fit)
    dw2.append(w2_err)

    if i==0:
        plt.plot(f_peaks, V_peaks, 'o', label='Máximos')
        plt.plot(f, fit_2(f, a_fit, w1_fit, w2_fit, beta_fit, V0_fit), label='Envolvente')
        # plt.plot(f, fit_2(f, -5, w1, w2, beta, V0), label='Teorico')
        plt.title("Espectro de frecuencias")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Ampliitud (dB)")
        plt.grid()
        plt.legend()
        plt.show()

plt.figure(figsize=(12,5))

# Plot resonance frequency w1 vs coupling capacitance
plt.subplot(121)
plt.errorbar(C3_val, w1_val, yerr=dw1, fmt='o', label='Experimental')
# plt.plot(C3_val, w1_val, 'o', label='Experimental')
plt.plot(C3_val, np.sqrt(1/L/C)*np.ones_like(C3_val), label='Teórico')
plt.xlabel(r"$C_3$ (F)")
plt.ylabel(r"$\omega_1 (rad/s)$")
plt.title(r"Resonancia $\omega_1$")
plt.grid()
plt.legend()

# Plot resonance frequency w2 vs coupling capacitance
C3_plot = np.linspace(1e-8, 1e-7, 100)
plt.subplot(122)
plt.errorbar(C3_val, w2_val, yerr=dw1, fmt='o', label='Experimental')
# plt.plot(C3_val, w2_val, 'o', label='Experimental')
plt.plot(C3_plot, np.sqrt(1/L/C + 2/L/C3_plot), label='Teórico')
plt.xlabel(r"$C_3$ (F)")
plt.ylabel(r"$\omega_2 (rad/s)$")
plt.title(r"Resonancia $\omega_2$")
plt.grid()
plt.legend()

plt.show()
