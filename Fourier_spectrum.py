import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Parámetros del experimento
R = 1700
L = 1
C = 1e-7
C3 = 1e-8
V0 = 10.3
w1 = np.sqrt(1 / (L * C))
w2 = np.sqrt(1 / (L * C) + 2 / (L * C3))
beta = R / (2 * L)
f0 = 14.1184

# Data import
df = pd.read_csv(r"RLCC_FFT_C2\ALL0000\F0000FFT.CSV", header=None)
df.columns = ["Metadata1", "Metadata2", "Empty", "X_Values", "Y_Values", "Empty2"]

frequencies = df["X_Values"].to_numpy()
decibels = df["Y_Values"].to_numpy()

def G(w,wn,beta):
    return 1/np.sqrt((w**2-wn**2)**2+(2*beta*w)**2)
def phi(w,wn,beta):
    val = np.arctan2(-2*beta*w,w**2-wn**2)
    return val + np.pi*(val<0)

def A1(w,beta,V0):
    G1 = G(w,w1,beta)
    G2 = G(w,w2,beta)
    phi1 = phi(w,w1,beta)
    phi2 = phi(w,w2,beta)
    return np.sqrt(G1**2+G2**2+2*G1*G2*np.cos(phi1-phi2))

def A2(w,beta,V0):
    G1 = G(w,w1,beta)
    G2 = G(w,w2,beta)
    phi1 = phi(w,w1,beta)
    phi2 = phi(w,w2,beta)
    return np.sqrt(G1**2+G2**2-2*G1*G2*np.cos(phi1-phi2))

def theta1(w,R,L,C,C3,V0):
    G1 = G(w,np.sqrt(1/L/C),R/2/L)
    G2 = G(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    phi1 = phi(w,np.sqrt(1/L/C),R/2/L)
    phi2 = phi(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    val = np.arctan2(G1*np.sin(phi1)+G2*np.sin(phi2),G1*np.cos(phi1)+G2*np.cos(phi2))
    return val % (2 * np.pi)

def theta2(w,R,L,C,C3,V0):
    G1 = G(w,np.sqrt(1/L/C),R/2/L)
    G2 = G(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    phi1 = phi(w,np.sqrt(1/L/C),R/2/L)
    phi2 = phi(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    val = np.arctan2(-G1*np.sin(phi1)+G2*np.sin(phi2),-G1*np.cos(phi1)+G2*np.cos(phi2))
    return val % (2 * np.pi)

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

sigma = 1e0
f_dc = 1.0

n_points = 2**12
# dt = tf / n_points
# dt = 4.932134e-6
# t = np.arange(0, n_points) * dt

# df = 1 / dt / n_points
# df = 1 / tf
# f = np.arange(0, n_points//2) * df
fmax = frequencies[-1] # 5045.56906
f = np.linspace(0, fmax, n_points//2)
N = len(f)

V_1 = np.zeros(N, dtype=np.complex128)
V_2 = np.zeros(N, dtype=np.complex128)
n_terms = 250
for n in range(n_terms):
    fn = (2*n + 1) * f0
    V_1 += A1(2 * np.pi * fn, beta, V0) / (2*n + 1) * (np.exp(1j * theta1(2 * np.pi * fn, R, L, C, C3, V0)) * gaussian(f, fn, sigma) - np.exp(-1j * theta1(2 * np.pi * fn, R, L, C, C3, V0)) * gaussian(f, -fn, sigma))
    V_2 += A2(2 * np.pi * fn, beta, V0) / (2*n + 1) * (np.exp(1j * theta2(2 * np.pi * fn, R, L, C, C3, V0)) * gaussian(f, fn, sigma) - np.exp(-1j * theta2(2 * np.pi * fn, R, L, C, C3, V0)) * gaussian(f, -fn, sigma))
    # V_1 += A1(2 * np.pi * f_dc, beta, V0) * gaussian(f, f_dc, sigma)
    # V_2 += A2(2 * np.pi * f_dc, beta, V0) * gaussian(f, f_dc, sigma)
V_1 *= -1j * V0 / np.pi / L / C
V_2 *= -1j * V0 / np.pi / L / C

V_1 = 2 * np.abs(V_1)**2 / N
V_2 = 2 * np.abs(V_2)**2 / N

V_1 = 20 * np.log10(V_1 + 1e-10*np.ones_like(V_1))
V_2 = 20 * np.log10(V_2 + 1e-10*np.ones_like(V_2))

# V_2 -= (np.min(V_2) - np.min(decibels))

V_2 = (V_2 - (np.max(V_2) - np.min(V_2)) ) * (np.max(decibels) - np.min(decibels)) / (np.max(V_2) - np.min(V_2))
V_2 -= (np.min(V_2) - np.min(decibels))

# V_1 = 20 * np.log10(V_1 / np.max(V_1) + 1e-10*np.ones_like(V_1))
# V_2 = 20 * np.log10(V_2 / np.max(V_2) + 1e-10*np.ones_like(V_2))

# plt.plot(f, V_2, 'o', label='V2')
plt.plot(f, V_2, label='V2')
plt.plot(frequencies, decibels, 'o', label='FFT Data')
# plt.plot(f, 20*np.log10(f0/f), label='1/n')
# plt.plot(f, V_1, label='V1')
plt.title('Fourier Transform of Capacitive Circuit')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.legend()
plt.grid()
plt.show()

peaks, = find_peaks(decibels, distance=10)
f_peaks = frequencies[peaks]
V_peaks = decibels[peaks]



plt.plot(f_peaks, V_peaks, 'o', label='Peaks')

