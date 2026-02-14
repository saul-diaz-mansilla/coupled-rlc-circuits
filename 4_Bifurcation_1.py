# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

f_plot = []
V_plot = []

# f_values = np.loadtxt(r"RL_diode_run_1\frequencies.txt")
# Open the file and read all lines
with open(r"RL_diode_run_1\frequencies.txt", "r") as file:
    data = file.read().splitlines()  # Reads all lines and removes "\n"

# Convert to numbers (int or float as needed)
f_values = [float(x) for x in data]  # Use int(x) if values are integers

for i in range(63):
    number = str(i).zfill(4)

    CH1 = pd.read_csv(r"RL_diode_run_1\ALL"+number+r"\F"+number+r"CH1.CSV", header=None)
    CH2 = pd.read_csv(r"RL_diode_run_1\ALL"+number+r"\F"+number+r"CH2.CSV", header=None)

    CH1.columns = ["Metadata1", "Metadata2", "Empty", "X_Values", "Y_Values", "Empty2"]
    CH2.columns = ["Metadata1", "Metadata2", "Empty", "X_Values", "Y_Values", "Empty2"]

    vertical_scale_1 = float(CH1.at[8,"Metadata2"])
    vertical_offset_1 = float(CH1.at[9,"Metadata2"])
    vertical_scale_2 = float(CH2.at[8,"Metadata2"])
    vertical_offset_2 = float(CH2.at[9,"Metadata2"])
    horizontal_scale_1 = float(CH1.at[11,"Metadata2"])
    horizontal_scale_2 = float(CH2.at[11,"Metadata2"])

    time_1 = CH1["X_Values"].to_numpy() #* horizontal_scale_1
    volt_1 = CH1["Y_Values"].to_numpy() #* vertical_scale_1
    time_2 = CH2["X_Values"].to_numpy() #* horizontal_scale_2
    volt_2 = CH2["Y_Values"].to_numpy() #* vertical_scale_2

    volt_1 -= vertical_offset_1 * np.ones_like(volt_1)
    volt_2 -= vertical_offset_2 * np.ones_like(volt_2)

    # f = (110 + 10*i)*1e3 # Hz
    f = f_values[i] * 1e3
    T = 1/f

    indeces, = np.where(((time_1 + T/4*np.ones_like(time_1)) % T) < .005 * horizontal_scale_1)
    # tol: .005 * hs
    for j in indeces:
        f_plot.append(f)
        V_plot.append(volt_2[j])

plt.plot(f_plot[:len(f_plot)//3],V_plot[:len(f_plot)//3],'.')
plt.xlabel(r"$f\ (Hz)$")
plt.ylabel(r"$V_R\ (V)$")
plt.title("Diagrama de bifurcación")
plt.grid()
plt.show()

# %%
