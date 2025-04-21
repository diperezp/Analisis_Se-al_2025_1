import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square
from scipy.signal import butter



#caracteristicas de la señal
frecuency=1 #frecuencia
n=500       #numero de muestras

#tiempo
t=np.linspace(0,5,500)
# Crear señal cuadrada
signal_square = square(2 * np.pi * frecuency * t)

#filtros a aplicar
h_bordes_1=np.array([-1,0,1])
h_smoth_1 =np.ones(10)/10
h_bordes_2=np.arange(-2,2,1)
h_smoth_2=np.ones(20)/20
h_lowpass_1=0.09**np.arange(10)
a,b=butter(4,0.2,btype='low')
h_lowpass_2=a
print(a)
print(b)

signal_convolve_1=np.convolve(signal_square,h_bordes_1,mode='same')
signal_convolve_2=np.convolve(signal_square,h_bordes_2,mode='same')
signal_convolve_3=np.convolve(signal_square,h_smoth_1,mode='same')
signal_convolve_4=np.convolve(signal_square,h_smoth_2,mode='same')
signal_convolve_5=np.convolve(signal_square,h_lowpass_1,mode='same')
signal_convolve_6=np.convolve(signal_square,h_lowpass_2,mode='same')

#graficamos la señal cuadrada
plt.figure(figsize=(5,5))
plt.subplot(4,2,1)
plt.plot(t,signal_square,label="Señal cuadrada")
plt.title("Señal Cuadrada")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()


plt.subplot(4,2,2)
plt.plot(t,signal_convolve_1,label="Señal cuadrada")
plt.title("Señal Cuadrada (filtro 1)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()


plt.subplot(4,2,3)
plt.plot(t,signal_convolve_2,label="Señal cuadrada")
plt.title("Señal Cuadrada (filtro 2)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()

plt.subplot(4,2,4)
plt.plot(t,signal_convolve_3,label="Señal cuadrada")
plt.title("Señal Cuadrada (filtro 3)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()

plt.subplot(4,2,5)
plt.plot(t,signal_convolve_4,label="Señal cuadrada")
plt.title("Señal Cuadrada (filtro 4)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()

plt.subplot(4,2,6)
plt.plot(t,signal_convolve_5,label="Señal cuadrada")
plt.title("Señal Cuadrada (filtro 5)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()

plt.subplot(4,2,7)
plt.plot(t,signal_convolve_6,label="Señal cuadrada")
plt.title("Señal Cuadrada (filtro 6)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()


plt.show()








