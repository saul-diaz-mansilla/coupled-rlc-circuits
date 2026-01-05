import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

L = 1
C = 1e-7
R = 100
C3 = 2e-7
w0 = 2*np.pi*144.5/25
V0 = 20.8/2

wn1 = 1/np.sqrt(L*C)
wn2 = np.sqrt(1/L/C+2/L/C3)
beta1 = R/(2*L)
beta2 = R/(2*L)

def G(w,wn,beta):
    return 1/np.sqrt((w**2-wn**2)**2+(2*beta*w)**2)
def phi(w,wn,beta):
    return np.arctan2(-2*beta*w,(w**2-wn**2))
def Amp(A,B,alpha,beta):
    return np.sqrt(A**2+B**2+2*A*B*np.cos(alpha-beta))
def theta(A,B,alpha,beta):
    return np.arctan2(A*np.sin(alpha)+B*np.sin(beta),A*np.cos(alpha)+B*np.cos(beta))

# t = np.linspace(-4*np.pi/w0, 4*np.pi/w0, 1000)
# t = np.linspace(0, 2*np.pi/w0, 1000)
t = np.linspace(10*2*np.pi/w0+0, 10*2*np.pi/w0+.1, 1000)
nterms = 1000
f = np.zeros_like(t)
q1 = np.zeros_like(t)
q2 = np.zeros_like(t)
for i in range(nterms):
    W = (2*i+1)*w0
    f += 4*V0/np.pi/L/(2*i+1)*np.sin(W*t)
    q1 += 2*V0/np.pi/L/(2*i+1)*Amp(G(W,wn1,beta1),-G(W,wn2,beta2),phi(W,wn1,beta1),phi(W,wn2,beta2))*np.sin(W*t-theta(G(W,wn1,beta1),-G(W,wn2,beta2),phi(W,wn1,beta1),phi(W,wn2,beta2)))
    q2 += 2*V0/np.pi/L/(2*i+1)*Amp(G(W,wn1,beta1),G(W,wn2,beta2),phi(W,wn1,beta1),phi(W,wn2,beta2))*np.sin(W*t-theta(G(W,wn1,beta1),G(W,wn2,beta2),phi(W,wn1,beta1),phi(W,wn2,beta2)))
V1 = q1/C
V2 = q2/C

plt.plot(t, f,label="Onda impulsora")
plt.plot(t, V1,label="Respuesta en el condensador 1")
plt.plot(t, V2,label="Respuesta en el condensador 2")
plt.title(f"Solución estacionaria a la onda cuadrada ({nterms} términos)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.legend()
plt.grid()
plt.show()

qf1 = fft(q1)
qf2 = fft(q2)
plt.plot(np.abs(qf1),label="Condensador 1")
plt.plot(np.abs(qf2),label="Condensador 2")
plt.title("Transformada de Fourier de la carga")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.legend()
plt.show()