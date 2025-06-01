import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import os

# Crear la carpeta "Graficas" si no existe
if not os.path.exists("Graficas/Densidades"):
    os.makedirs("Graficas/Densidades")


# ================================================================================
# Parámetros
pasos = 100000 # Cambia este valor para graficar diferentes variables
N = 128  # Lista de tamaños N para las simulaciones
T = 6  # Temperatura del sistema, valores posibles: 0.5, 1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6
Guardar = True  # Si es True, guarda la gráfica en un archivo
# ================================================================================


@nb.njit(parallel=True)
def dist_densidad(spines):
    """Calcula la densidad de espines en el sistema."""
    N = spines.shape[0]
    densidad = np.zeros(N)
    for i in nb.prange(N):
        for j in nb.prange(N):
            if spines[j, i] == 1:
                densidad[i] += 1
    """Calcula la distribución de densidades."""
    dist = np.zeros(N)
    for i in nb.prange(N):
        for j in nb.prange(N):
            if i == densidad[j]:
                dist[i] += 1
    for i in nb.prange(N):
        dist[i] = dist[i] / N
    return dist, densidad

def leer_ultimos_spines(filename, N):
    """
    Lee las últimas N filas del archivo y las guarda en una matriz N x N.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        # Filtra líneas vacías y elimina saltos de línea
        lines = [line.strip() for line in lines if line.strip()]
        # Toma las últimas N filas
        ultimas_filas = lines[-N:]
        # Convierte cada fila en una lista de enteros
        matriz = np.array([list(map(int, fila.split(','))) for fila in ultimas_filas])
    return matriz

# Ejemplo de uso:
filename = "spines/spinesN_" + str(N)+ "_T=" + str(T) + ".txt"
spines = leer_ultimos_spines(filename, N)

# Calcular la distribución de densidades
dist, densidad = dist_densidad(spines)
# Graficar la distribución de densidades
plt.figure(figsize=(8, 5))
plt.bar(range(len(dist)), dist, color='blue', alpha=0.7)
plt.xlabel("Número de Espines hacia Arriba")
plt.ylabel("Distribución")
plt.title("Distribución de Densidades de Espines")
if Guardar:
    # Guardar el gráfico
    output_filename = f"Graficas/Densidades/Distribucion_Densidades_N{N}_T{T}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
plt.show()