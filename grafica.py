import numpy as np
import matplotlib.pyplot as plt

# Leer el archivo
filename = "Resultados/Resultados_N=64_pMc=100000_M=0.txt"

# Cargar los datos, omitiendo la primera línea (cabecera)
data = np.loadtxt(filename, delimiter=",", skiprows=1)

# Extraer las columnas necesarias
temperatura = data[:, 0]  # Primera columna (Temperatura)
calor_especifico = data[:, 4]  # Quinta columna (Calor específico)

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(temperatura, calor_especifico, marker="o", label="Calor específico")
plt.xlabel("Temperatura")
plt.ylabel("Calor específico")
plt.title("Calor específico en función de la Temperatura")
plt.grid()
plt.legend()
plt.show()