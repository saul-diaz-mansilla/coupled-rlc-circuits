# Bode Capacitivo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Parámetros del experimento
R = 1700
L = 1
C = 1e-7
C3 = 5e-8
V0 = 10.3
w1 = np.sqrt(1 / (L * C))
w2 = np.sqrt(1 / (L * C) + 2 / (L * C3))
min_distance = 1.5e-4  # Minimum time separation for peaks and zero crossings

def seno(t, A, omega, phase, const):
    phase = phase % (2 * np.pi)
    return A * np.sin(omega * t - phase) + const

def robust_maximum_indices(array, time_array, min_distance):
    """
    Finds peaks that are sufficiently far apart in a noisy signal.

    Parameters:
        array (numpy array): Voltage signal.
        time_array (numpy array): Corresponding time values.
        min_distance (float): Minimum allowed time separation between consecutive peaks.

    Returns:
        numpy array: Indices of valid peaks.
    """
    time_step = np.mean(np.diff(time_array))  # Approximate time step
    min_samples = int(min_distance / time_step)

    # Find peaks with a minimum separation of `min_samples` indices
    peaks, _ = find_peaks(array, distance=min_samples, prominence=np.ptp(array) * 0.1)
    
    return peaks

def filtered_zero_crossings(array, time_array, min_distance):
    """
    Finds zero crossings that are sufficiently spaced apart.

    Parameters:
        array (numpy array): Voltage signal.
        time_array (numpy array): Corresponding time values.
        min_distance (float): Minimum allowed time separation between consecutive zero crossings.

    Returns:
        numpy array: Indices of valid zero crossings.
    """
    crossings = np.where(np.diff(np.sign(array)))[0]
    valid_crossings = []

    last_time = -np.inf
    for idx in crossings:
        if time_array[idx] - last_time >= min_distance:
            valid_crossings.append(idx)
            last_time = time_array[idx]

    return np.array(valid_crossings)

def G(w,wn,beta):
    return 1/np.sqrt((w**2-wn**2)**2+(2*beta*w)**2)
def phi(w,wn,beta):
    val = np.arctan2(-2*beta*w,w**2-wn**2)
    return val + np.pi*(val<0)

# def A1(w,beta,V0):
#     G1 = G(w,w1,beta)
#     G2 = G(w,w2,beta)
#     phi1 = phi(w,w1,beta)
#     phi2 = phi(w,w2,beta)
#     return V0/2/L/C*np.sqrt(G1**2+G2**2+2*G1*G2*np.cos(phi1-phi2))

def A1(w,R,L,C,C3,V0):
    G1 = G(w,np.sqrt(1/L/C),R/2/L)
    G2 = G(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    phi1 = phi(w,np.sqrt(1/L/C),R/2/L)
    phi2 = phi(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    return V0/2/L/C*np.sqrt(G1**2+G2**2+2*G1*G2*np.cos(phi1-phi2))

def A2(w,beta,V0):
    G1 = G(w,w1,beta)
    G2 = G(w,w2,beta)
    phi1 = phi(w,w1,beta)
    phi2 = phi(w,w2,beta)
    return V0/2/L/C*np.sqrt(G1**2+G2**2-2*G1*G2*np.cos(phi1-phi2))

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

# Lists to store extracted parameters
V_0 = []
V_C1 = []
w = []
ph_1 = []
ph_2 = []
DC_0 = []
DC_C1 = []

for i in range(46):  # 46
    number = str(i).zfill(4)

    CH1 = pd.read_csv(r"C:\Users\Usuario\Documents\Documentos\Universidad\Tecnicas experimentales II\Proyecto\Democrito_alfa\1_Platon\ALL" + number + r"\F" + number + r"CH1.CSV", header=None)
    CH2 = pd.read_csv(r"C:\Users\Usuario\Documents\Documentos\Universidad\Tecnicas experimentales II\Proyecto\Democrito_alfa\1_Platon\ALL" + number + r"\F" + number + r"CH2.CSV", header=None)

    CH1.columns = ["Metadata1", "Metadata2", "Empty", "X_Values", "Y_Values", "Empty2"]
    CH2.columns = ["Metadata1", "Metadata2", "Empty", "X_Values", "Y_Values", "Empty2"]

    # Extract scale and offset values
    vertical_scale_1 = float(CH1.at[8, "Metadata2"])
    vertical_offset_1 = float(CH1.at[9, "Metadata2"])
    vertical_scale_2 = float(CH2.at[8, "Metadata2"])
    vertical_offset_2 = float(CH2.at[9, "Metadata2"])
    horizontal_scale_1 = float(CH1.at[11, "Metadata2"])
    horizontal_scale_2 = float(CH2.at[11, "Metadata2"])

    # Convert data to numpy arrays
    time_1 = CH1["X_Values"].to_numpy()
    volt_1 = CH1["Y_Values"].to_numpy() - vertical_offset_1
    time_2 = CH2["X_Values"].to_numpy()
    volt_2 = CH2["Y_Values"].to_numpy() - vertical_offset_2

    # Detect peaks with minimum separation
    max_indices_1 = robust_maximum_indices(volt_1, time_1, min_distance)
    
    # Estimate frequency from maxima
    if len(max_indices_1) > 1:
        period = np.mean(np.diff(time_1[max_indices_1]))  # Average period
        omega = 2 * np.pi / period
    else:
        omega = 1200  # Fallback frequency if not enough peaks are detected

    # Detect zero crossings with minimum separation
    zero_crossings_1 = filtered_zero_crossings(volt_1, time_1, min_distance)
    zero_crossings_2 = filtered_zero_crossings(volt_2, time_2, min_distance)

    # Compute phase based on average zero-crossing time
    if len(zero_crossings_1) > 0:
        t_1_0 = np.mean(time_1[zero_crossings_1])
    else:
        t_1_0 = 0  # Fallback value

    if len(zero_crossings_2) > 0:
        t_2_0 = np.mean(time_2[zero_crossings_2])
    else:
        t_2_0 = 0  # Fallback value

    phase_1 = (omega * t_1_0) % (2 * np.pi)
    phase_2 = (omega * t_2_0) % (2 * np.pi)

    # Fit sinusoidal curves to voltage data
    parameters_1, _ = curve_fit(seno, time_1, volt_1, p0=[(max(volt_1)-min(volt_1))/2, omega, phase_1, np.mean(volt_1)])
    parameters_2, _ = curve_fit(seno, time_2, volt_2, p0=[(max(volt_2)-min(volt_2))/2, omega, phase_2, np.mean(volt_2)])

    V_0.append(parameters_1[0])
    V_C1.append(parameters_2[0])
    w.append(parameters_1[1])
    ph_1.append(parameters_1[2])
    ph_2.append(parameters_2[2])
    DC_0.append(parameters_1[3])
    DC_C1.append(parameters_2[3])

    # # Print extracted parameters for debugging
    # print(f"Amplitude (Expected vs. Fitted): {((max(volt_1)-min(volt_1))/2)} vs. {parameters_1[0]}")
    # print(f"Frequency (Expected vs. Fitted): {omega} vs. {parameters_1[1]}")
    # print(f"Phase (Expected vs. Fitted): {phase_1} vs. {parameters_1[2]}")

    # # Plot results
    # plt.plot(time_1, volt_1, '.', label="Fuente")
    # plt.plot(time_1, seno(time_1, parameters_1[0], parameters_1[1], parameters_1[2], parameters_1[3]), label="Ajuste fuente")
    # plt.plot(time_2, volt_2, '.', label="VC1")
    # plt.plot(time_2, seno(time_2, parameters_2[0], parameters_2[1], parameters_2[2], parameters_2[3]), label="Ajuste VC1")
    
    # plt.xlabel('t (s)')
    # plt.ylabel('V (V)')
    # plt.legend()
    # plt.grid()
    # plt.show()
w_fit = np.linspace(w[0],w[-1],1000)
plt.plot(w, np.abs(V_C1), '.', label="Datos experimentales")
plt.plot(w_fit, A1(w_fit, R, L, C, C3, 10.3), label="Curva teórica")
plt.title("Ajuste")
plt.xlabel(r"$\omega$ (rad/s)")
plt.ylabel(r"$V_{C1}$ (V)")
plt.legend()
plt.grid()
plt.show()

ph_1 = np.unwrap(np.array(ph_1))
ph_2 = np.unwrap(np.array(ph_2))
dph = (ph_2-ph_1) % (2*np.pi)
plt.plot(w, np.unwrap(dph) % (np.pi), '.', label="Experimental")
plt.plot(w_fit, theta1(w_fit, R, L, C, C3, V0))
plt.title("Desfase en C1")
plt.xlabel(r"$\omega$ (rad/s)")
plt.ylabel(r"$\phi$ (rad)")
plt.legend()
plt.grid()
plt.show()