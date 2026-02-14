# coupled-rlc-circuits
An experimental project that studies the behavior of RLC circuits as harmonic oscillators as part of a lab module. The project explores resonance in simple RLC circuits, capacitive coupling between two RLC circuits in both time and frequency domains, and chaotic behavior in RL-diode circuits.

Co-author: Álvaro Castillo Fernández: alvaro.castillof@estudiante.uam.es

Read the final lab report in "0_Report.pdf".

Folders:
- RLCC_C1 / C2: Data from capacitive coupling for various input frequencies.
- RLCC_FFT_C1 / C2: Data from capacitive coupling varying C3 and C in the frequency spectrum. Measured at C1 / C2.
- RL_diode_run_1 / 2: Runs of data varying input frequency in RL-diode circuit.
- Unused_data: Unused data in final report.
- Figures: Final figures generated for the report.

Main analysis:
Section 2: Simple RLC Circuit
- "2_data.xlsx": Empirical data taken manually of a simple RLC circuit varying different parameters
- "2_Amplitude.py": Amplitude of output signal as a function of the input frequency. Empirical data vs theory
- "2_Frequency.py": Resonance frequency varying the inductance L. Empirical data vs theory.
- "2_Phase.py": Phase difference between input and output signals as a function of the input frequency. Empirical data vs theory.

Section 3: Coupled RLC Circuits
- "3_data.xlsx": Data for capacitive and inductive coupling taken manually.
- "3_Main.py": Main script for observed amplitudes and phase differences in C1 and C2.
- "3_Fourier_C3.py": Extraction of resonance frequencies with variable C3.
- "3_Fourier_C.py": Extraction of resonance frequencies with variable C.

Section 4: RL-Diode Circuit and Chaos
- "4_Phase_diagrams.py": Generates phase space plots (V_source vs V_resistance) to visualize periodic vs. chaotic orbits.
- "4_Bifurcation_1.py": Main script for generating the bifurcation diagram to show period-doubling as frequency increases.
- "4_Bifurcation_2.py": Bifurcation analysis for data in second run (unused in final report).

Miscelaneous:
- "Capacitive_theory.py": Theoretical model for expected Bode diagram for capacitive coupling.
- "Capacitive_analysis.py": Analyisis and fit of first capacitive coupling data. Failed due to oscilloscope impedance.
- "Fourier_simple.py": Theoretical model for the oscilloscope input and output signals for a simple RLC circuit.
- "Fourier_coupled.py": Theoretical model for the capacitive coupling.
- "Fourier_spectrum-py": Theoretical spectrum observed in capacitive coupling.
- "Inductive.py": Attempted study of inductive coupling (unused in report). Failed calculation of mutual inductance.