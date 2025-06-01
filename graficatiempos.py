import numpy as np
import matplotlib.pyplot as plt
import os

# Crear la carpeta "Graficas" si no existe
if not os.path.exists("Graficas"):
    os.makedirs("Graficas")

# ================================================================================
# Parámetros
pasos = 100000 # Cambia este valor para graficar diferentes variables
N = 128  # Lista de tamaños N para las simulaciones
Guardar = True  # Si es True, guarda la gráfica en un archivo
# ================================================================================

file1name = f"Tiempos/JOEL_N{N}_PMc_{pasos}.txt"
if not os.path.exists(file1name):
    raise FileNotFoundError(f"El archivo {file1name} no existe. Asegúrate de que el archivo esté en la carpeta Tiempos.")
file2name = f"Tiempos/PC_N{N}_PMc_{pasos}.txt"
if not os.path.exists(file2name):
    raise FileNotFoundError(f"El archivo {file2name} no existe. Asegúrate de que el archivo esté en la carpeta Tiempos.")
# Cargar los datos de los archivos
data1 = np.loadtxt(file1name, delimiter=",", skiprows=1)
data2 = np.loadtxt(file2name, delimiter=",", skiprows=1)
# Extraer los datos
threads1 = data1[:, 0]  # Primera columna (Número de hilos)
tiempos1 = data1[:, 1]  # Segunda columna (Tiempos en segundos)
tiempos2 = data2[:, 1]  # Segunda columna (Tiempos en segundos)
l = len(threads1) - len(tiempos2)
if l > 0:
    # Si hay más hilos en tiempos1, agregar ceros a tiempos2
    for _ in range(l):
        tiempos2 = np.append(tiempos2, None)  # Agregar NaN para alinear las longitudes

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(threads1, tiempos1, label='Joel')
plt.plot(threads1, tiempos2, label='Mi Pc')
plt.xlabel("Número de threads")
plt.ylabel("Tiempo (segundos)")
plt.title("Tiempo de ejecución vs Número de threads")
plt.grid()
plt.legend()
if Guardar:
    # Guardar el gráfico
    output_filename = f"Graficas/Tiempo_vs_Threads_N{N}_PMc_{pasos}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
plt.show()