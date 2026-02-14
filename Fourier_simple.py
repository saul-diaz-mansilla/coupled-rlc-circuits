import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

L = 1
C = 1e-7
R = 100
w0 = 2*np.pi*144.5/25
q0 = 23.2/2*C
V0 = 20.8/2
wn = 1/np.sqrt(L*C)
beta = R/(2*L)

def G(w):
    return 1/np.sqrt((w**2-wn**2)**2+(2*beta*w)**2)
def phi(w):
    return np.arctan2(-2*beta*w,(w**2-wn**2))

# t = np.linspace(-4*np.pi/w0, 4*np.pi/w0, 1000)
# t = np.linspace(0, 2*np.pi/w0, 1000)
t = np.linspace(10*2*np.pi/w0+0, 10*2*np.pi/w0+.1, 1000)
nterms = 1000
f = np.zeros_like(t)
q = np.zeros_like(t)
for i in range(nterms):
    W = (2*i+1)*w0
    f += 4*V0/np.pi/L/(2*i+1)*np.sin(W*t)
    q += 4*V0/np.pi/L/(2*i+1)*G(W)*np.sin(W*t-phi(W))
V = q/C

plt.plot(t, f,label="Onda impulsora")
plt.plot(t, V,label="Respuesta estacionaria")
plt.title(f"Solución estacionaria a la onda cuadrada ({nterms} términos)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.legend()
plt.grid()
plt.show()

qf = fft(q)
plt.plot(np.abs(qf))
plt.title("Transformada de Fourier de la carga")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.show()