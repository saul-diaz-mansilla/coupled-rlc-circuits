# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Open the file and read all lines
with open(r"C:\Users\Usuario\Documents\Documentos\Universidad\Tecnicas experimentales II\Proyecto\Frecuencias_Betas.txt", "r") as file:
    data = file.read().splitlines()  # Reads all lines and removes "\n"

# Convert to numbers (int or float as needed)
f_values = [float(x) for x in data]  # Use int(x) if values are integers
i = 11
number = str(i).zfill(4)
CH1 = pd.read_csv(r"C:\Users\Usuario\Documents\Documentos\Universidad\Tecnicas experimentales II\Proyecto\IFT_Betas\ALL"+number+r"\F"+number+r"CH1.CSV", header=None)
CH2 = pd.read_csv(r"C:\Users\Usuario\Documents\Documentos\Universidad\Tecnicas experimentales II\Proyecto\IFT_Betas\ALL"+number+r"\F"+number+r"CH2.CSV", header=None)

CH1.columns = ["Metadata1", "Metadata2", "Empty", "X_Values", "Y_Values", "Empty2"]
CH2.columns = ["Metadata1", "Metadata2", "Empty", "X_Values", "Y_Values", "Empty2"]

vertical_scale_1 = float(CH1.at[8,"Metadata2"])
vertical_offset_1 = float(CH1.at[9,"Metadata2"])
vertical_scale_2 = float(CH2.at[8,"Metadata2"])
vertical_offset_2 = float(CH2.at[9,"Metadata2"])
horizontal_scale_1 = float(CH1.at[11,"Metadata2"])
horizontal_scale_2 = float(CH2.at[11,"Metadata2"])

time_1 = CH1["X_Values"].to_numpy()
volt_1 = CH1["Y_Values"].to_numpy()
time_2 = CH2["X_Values"].to_numpy()
volt_2 = CH2["Y_Values"].to_numpy()

volt_1 -= vertical_offset_1 * np.ones_like(volt_1)
volt_2 -= vertical_offset_2 * np.ones_like(volt_2)

# time_1 = time_1 * horizontal_scale_1
# time_2 = time_2 * horizontal_scale_2
# volt_1 = volt_1 * vertical_scale_1 + vertical_offset_1 * np.ones_like(volt_1)
# volt_2 = volt_2 * vertical_scale_2 + vertical_offset_2 * np.ones_like(volt_2)

f = 110*1e3 # Hz
T = 1/f

indeces, = np.where((time_1 % T) < .005 * horizontal_scale_1)

plt.figure(figsize=(12,5))

plt.subplot(121)
plt.plot(time_1,volt_1,label=r"$V_{in}\ (V)$")
plt.plot(time_2,volt_2,label=r"$V_R\ (V)$")
plt.xlabel(r"$t\ (s)$")
plt.ylabel(r"$V\ (V)$")
plt.title(r"Osciloscopio con $f = $"+str(f_values[i])+r"$\ kHz$")
plt.legend()
plt.grid()

plt.subplot(122)
plt.plot(volt_1,volt_2,'.', label="Diagrama de fase")
plt.plot(volt_1[indeces],volt_2[indeces],'.', label="Sección de Poincaré")
plt.xlabel(r"$V_{in}\ (V)$")
plt.ylabel(r"$V_{R}\ (V)$")
plt.xlim([-12,12])
plt.ylim([-4.5,3.5])
plt.title(r"Diagrama de fases con $f = $"+str(f_values[i])+r"$\ kHz$")
plt.legend(loc="upper left")
plt.grid()

plt.show()

