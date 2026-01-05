import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

CH1 = pd.read_csv(r"C:\Users\Usuario\Documents\Documentos\Universidad\Tecnicas experimentales II\Proyecto\CSVs\DS0000.CSV", header=None)
vertical_scale_1 = float(CH1.at[5,1])
vertical_offset_1 = float(CH1.at[6,1])
horizontal_scale_1 = float(CH1.at[8,1])

CH2 = pd.read_csv(r"C:\Users\Usuario\Documents\Documentos\Universidad\Tecnicas experimentales II\Proyecto\CSVs\DS0001.CSV", header=None)

vertical_scale_2 = float(CH2.at[5,1])
vertical_offset_2 = float(CH2.at[6,1])
horizontal_scale_2 = float(CH2.at[8,1])

n_data = int(CH2.at[0,1])
t = np.linspace(0, horizontal_scale_1*(n_data-1), n_data)

V_in = CH1[0].to_numpy()  # Convert the column to a NumPy array
V_in = V_in[16:]  # Slice the array starting from index 16
V_in = np.array([float(i) for i in V_in])  # Convert each element to float and create a NumPy array
# V_in = V_in * vertical_scale_1 + vertical_offset_1 * np.ones_like(V_in)  # Perform element-wise operations

V_out = CH2[0].to_numpy()
V_out = V_out[16:]
V_out = np.array([float(i) for i in V_out])


# print(CH1)