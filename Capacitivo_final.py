# Bode Capacitivo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.odr import ODR, Model, RealData

# Parámetros del experimento
R = 1700
L = 1
C = 1e-7
C3 = 5e-8
V0 = 10.3
w1 = np.sqrt(1 / (L * C))
w2 = np.sqrt(1 / (L * C) + 2 / (L * C3))
min_distance = 2.5e-4  # Minimum time separation for peaks and zero crossings

def seno(t, A, omega, phase, const):
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

def A1(w,R,L,C,C3,V0):
    G1 = G(w,np.sqrt(1/L/C),R/2/L)
    G2 = G(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    phi1 = phi(w,np.sqrt(1/L/C),R/2/L)
    phi2 = phi(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    return V0/2/L/C*np.sqrt(G1**2+G2**2+2*G1*G2*np.cos(phi1-phi2))

def A2(w,R,L,C,C3,V0):
    G1 = G(w,np.sqrt(1/L/C),R/2/L)
    G2 = G(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    phi1 = phi(w,np.sqrt(1/L/C),R/2/L)
    phi2 = phi(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    return V0/2/L/C*np.sqrt(G1**2+G2**2-2*G1*G2*np.cos(phi1-phi2))

def theta1(w,R,L,C,C3,V0):
    G1 = G(w,np.sqrt(1/L/C),R/2/L)
    G2 = G(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    phi1 = phi(w,np.sqrt(1/L/C),R/2/L)
    phi2 = phi(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    val = np.arctan2(G1*np.sin(phi1)+G2*np.sin(phi2),G1*np.cos(phi1)+G2*np.cos(phi2))
    return val

def theta2(w,R,L,C,C3,V0):
    G1 = G(w,np.sqrt(1/L/C),R/2/L)
    G2 = G(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    phi1 = phi(w,np.sqrt(1/L/C),R/2/L)
    phi2 = phi(w,np.sqrt(1/L/C+2/L/C3),R/2/L)
    val = np.arctan2(-G1*np.sin(phi1)+G2*np.sin(phi2),-G1*np.cos(phi1)+G2*np.cos(phi2))
    return val % (np.pi)

# Lists to store extracted parameters
V_0 = []
dV_0 = []
V_C1 = []
dV_C1 = []
w = []
d_w = []
ph_in_1 = []
d_ph_in_1 = []
ph_out_1 = []
d_ph_out_1 = []
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

    phase_1 = omega * t_1_0
    phase_2 = omega * t_2_0

    # Fit sinusoidal curves to voltage data
    parameters_1, covariance_1 = curve_fit(seno, time_1, volt_1, p0=[(max(volt_1)-min(volt_1))/2, omega, phase_1 % (2*np.pi), np.mean(volt_1)], bounds=([-np.inf, omega*0.9, 0, -np.inf], [np.inf, omega*1.1, 2*np.pi, np.inf]))
    parameters_2, covariance_2 = curve_fit(seno, time_2, volt_2, p0=[(max(volt_2)-min(volt_2))/2, omega, phase_2 % (2*np.pi), np.mean(volt_2)], bounds=([-np.inf, omega*0.9, 0, -np.inf], [np.inf, omega*1.1, 2*np.pi, np.inf]))
    err_1 = np.sqrt(np.diag(covariance_1))
    err_2 = np.sqrt(np.diag(covariance_2))

    V_0.append(parameters_1[0])
    dV_0.append(err_1[0])
    V_C1.append(parameters_2[0])
    dV_C1.append(err_2[0])
    w.append(parameters_1[1])
    d_w.append(err_1[1])
    ph_in_1.append(parameters_1[2])
    d_ph_in_1.append(err_1[2])
    ph_out_1.append(parameters_2[2])
    d_ph_out_1.append(err_2[2])
    DC_0.append(parameters_1[3])
    DC_C1.append(parameters_2[3])

w_fit = np.linspace(w[0],w[-1],1000)
parameters_C1, covariance_C1 = curve_fit(A1, w, np.abs(V_C1), p0=[R,L,C,C3,V0])

figure1 = plt.figure()
fig1 = figure1.add_subplot(111)
figure2 = plt.figure()
fig2 = figure2.add_subplot(111)

fig1.errorbar(w, np.abs(V_C1), xerr=d_w, yerr=dV_C1, fmt='.', label="Datos experimentales")
fig1.plot(w_fit, A1(w_fit, parameters_C1[0], parameters_C1[1], parameters_C1[2], parameters_C1[3], parameters_C1[4]), label="Curva ajustada")
plt.show()
ph_in_1 = np.unwrap(np.array(ph_in_1))
d_dph_1 = d_ph_in_1 + d_ph_out_1
ph_out_1 = np.unwrap(np.array(ph_out_1))
dph_1 = (ph_out_1 - ph_in_1) % (2*np.pi)
dph_1 = np.unwrap(dph_1) % (np.pi)

parameters_ph1, _ = curve_fit(theta1, w, dph_1, p0=[R,L,C,C3,V0])

fig2.errorbar(w, dph_1, xerr=d_w, yerr=d_dph_1, fmt='o', label="Datos experimentales")
fig2.plot(w_fit, theta1(w_fit, parameters_ph1[0], parameters_ph1[1], parameters_ph1[2], parameters_ph1[3], parameters_ph1[4]), label="Ajuste")

min_distance = 5.5e-4
# Lists to store extracted parameters
V_0 = []
V_C2 = []
w = []
ph_in_2 = []
ph_out_2 = []
DC_0 = []
DC_C2 = []
for i in [i for i in range(55) if i != 5]:  # 55
    number = str(i).zfill(4)

    CH1 = pd.read_csv(r"C:\Users\Usuario\Documents\Documentos\Universidad\Tecnicas experimentales II\Proyecto\Democrito_alfa\2_Aristoteles\ALL" + number + r"\F" + number + r"CH1.CSV", header=None)
    CH2 = pd.read_csv(r"C:\Users\Usuario\Documents\Documentos\Universidad\Tecnicas experimentales II\Proyecto\Democrito_alfa\2_Aristoteles\ALL" + number + r"\F" + number + r"CH2.CSV", header=None)

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

    phase_1 = omega * t_1_0
    phase_2 = omega * t_2_0

    # Fit sinusoidal curves to voltage data
    parameters_1, _ = curve_fit(seno, time_1, volt_1, p0=[(max(volt_1)-min(volt_1))/2, omega, phase_1, np.mean(volt_1)])
    parameters_2, _ = curve_fit(seno, time_2, volt_2, p0=[(max(volt_2)-min(volt_2))/2, omega, phase_2, np.mean(volt_2)])

    V_0.append(parameters_1[0])
    V_C2.append(parameters_2[0])
    w.append(parameters_1[1])
    ph_in_2.append(parameters_1[2])
    ph_out_2.append(parameters_2[2])
    DC_0.append(parameters_1[3])
    DC_C2.append(parameters_2[3])

parameters_C2, covariance_C2 = curve_fit(A2, w, np.abs(V_C2), p0=[R,L,C,C3,V0])

fig1.plot(w, np.abs(V_C2), 'g.', label="Datos experimentales 2")
fig1.plot(w_fit, A2(w_fit, parameters_C2[0], parameters_C2[1], parameters_C2[2], parameters_C2[3], parameters_C2[4]), label="Curva ajustada 2")

ph_in_2 = np.unwrap(np.array(ph_in_2))
ph_out_2 = np.unwrap(np.array(ph_out_2))
dph_2 = (ph_out_2 - ph_in_2) % (2*np.pi)
dph_2 = np.unwrap(dph_2) % (np.pi)

parameters_ph2, _ = curve_fit(theta1, w, dph_2, p0=[R,L,C,C3,V0])
fig2.plot(w, dph_2, '.', label="Datos experimentales 2")
fig2.plot(w_fit, theta2(w_fit, parameters_ph2[0], parameters_ph2[1], parameters_ph2[2], parameters_ph2[3], parameters_ph2[4]), label="Ajuste 2")

fig1.set_title("Amplitudes en los condensadores")
fig1.set_xlabel(r"$\omega$ (rad/s)")
fig1.set_ylabel(r"$V_{C2}$ (V)")
fig1.legend()
fig1.grid()

fig2.set_title("Amplitudes en los condensadores")
fig2.set_xlabel(r"$\omega$ (rad/s)")
fig2.set_ylabel(r"$V_{C2}$ (V)")
fig2.legend()
fig2.grid()

plt.show()
