# coupled-rlc-circuits
An experimental project that studies the behavior of RLC circuits as harmonic oscillators as part of a lab module. The project explores resonance in simple RLC circuits, capacitive coupling between two RLC circuits in both time and frequency domains, and chaotic behavior in RL-diode circuits.

Co-author: Álvaro Castillo Fernández: alvaro.castillof@estudiante.uam.es

Read the final lab report in "0_TE2_RLC_Acoplados.pdf".

Folders:
- CSVs: Raw experimental data exported from the oscilloscope in .csv format.
- bombardeen_tektronix_c1 / c2: Waveform data sets for the primary and coupled capacitors.
- furier_c1 / c2: Oscilloscope data containing the FFT of the input signal and the signal measured at C1 / C2.
- z_Figures: Final figures generated for the report.
- Democrito_alfa / IFT_Betas / moooooodle: Data subsets and working directories for specific experimental runs.

Main analysis:
Section 2: Simple RLC Circuit
- "1_data.xlsx": Empirical data taken manually of a simple RLC circuit varying different parameters
- "1_Amplitude.py": Amplitude of output signal as a function of the input frequency. Empirical data vs theory
- "1_Frequency.py": Resonance frequency varying the inductance L. Empirical data vs theory.
- "1_Phase.py": Phase difference between input and output signals as a function of the input frequency. Empirical data vs theory.

Section 3: Coupled RLC Circuits
- "Datos_semana_2.xlsx": Data log for the second phase of experimental measurements.
- "2_Fourier_C3.py": Extraction of resonance frequencies with variable C3.
- "2_Fourier_C.py": Extraction of resonance frequencies with variable C.

Section 4: RL-Diode Circuit and Chaos
- "4_Phase_diagrams.py": Generates phase space plots (V_source vs V_resistance) to visualize periodic vs. chaotic orbits.
- "4_Bifurcation_1.py": Main script for generating the bifurcation diagram to show period-doubling as frequency increases.
- "4_Bifurcation_2.py": Bifurcation analysis for data in second run (unused in final report).

Miscelaneous:
- "Fourier.py": Theoretical model for the oscilloscope input and output signals for a simple RLC circuit.
- "Fourier_coupled.py": Theoretical model for the capacitive coupling.
- "Fourier_spectrum-py": Theoretical spectrum observed in capacitive coupling.

Work in progress cleaning up the rest of the codes