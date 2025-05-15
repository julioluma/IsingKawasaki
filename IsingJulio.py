import numpy as np
import matplotlib.pyplot as plt
import numba as nb

@nb.njit
def inicializar_spines(N):
    """Inicializa el spines con espines aleatorios (+1 o -1)."""
    spines = np.empty((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            spines[i, j] = 1 if np.random.rand() > 0.5 else -1
    return spines

def guardar_spines_txt(spines, paso, filename="estados.txt"):
    """Guarda el estado de los spines en un archivo de texto en el formato especificado."""
    with open(filename, "a") as f:  # Abre el archivo en modo de adición
        np.savetxt(f, spines, fmt="%d", delimiter=",")  # Guarda la matriz de espines
        f.write("\n")  # Agrega una línea en blanco para separar los pasos

@nb.njit(parallel=True)  # Decorador para compilar la función con Numba
def energia_promedio(spines):
    """Calcula la energía promedio del sistema."""
    N = spines.shape[0]
    energia = 0
    for i in range(N):
        for j in range(N):
            S = spines[i, j]
            # Interacción con los vecinos
            vecinos = spines[(i+1)%N, j] + spines[i, (j+1)%N] + spines[(i-1)%N, j] + spines[i, (j-1)%N]
            energia += -S * vecinos
                    
    return energia/(2* N*N)  # Dividir por 2 para evitar contar cada interacción dos veces


@nb.njit  # Decorador para compilar la función con Numba
def magnetizacion_promedio(spines):
    """Calcula la magnetización promedio del sistema."""
    N = spines.shape[0]
    return np.abs(np.sum(spines))/(N *N)

@nb.njit  # Decorador para compilar la función con Numba
def metropolis(spines, beta):
    """Implementa el algoritmo de Metropolis para actualizar el spines."""
    N = spines.shape[0]
    for _ in range(N * N):  # N^2 intentos de actualización por paso
        i, j = np.random.randint(0, N, size=2)
        S = spines[i, j]
        # Interacción con los vecinos
        vecinos = spines[(i+1)%N, j] + spines[i, (j+1)%N] + spines[(i-1)%N, j] + spines[i, (j-1)%N]
        dE = 2 * S * vecinos
        p= min(1, np.exp(-beta * dE))  # Probabilidad de aceptación
        # Aceptar o rechazar el cambio
        if np.random.rand() < p:
            spines[i, j] *= -1
    return spines

def simular_ising(N, T, pasos):
    """Simula el modelo de Ising."""
    beta = 1 / T
    spines = inicializar_spines(N)
    energias = []
    magnetizaciones = []

    # Limpia el archivo de salida antes de comenzar
    open("estados.txt", "w").close()

    for paso in range(pasos):
        spines = metropolis(spines, beta)
        energia = energia_promedio(spines)
        magnetizacion = magnetizacion_promedio(spines)
        energias.append(energia)
        magnetizaciones.append(magnetizacion)

        # Guardar el estado de los spines en cada paso
        guardar_spines_txt(spines, paso)

    return spines, energias, magnetizaciones

# Parámetros
N = 8  # Tamaño del spines (N x N)
T = 0.5  # Temperatura
pasos = 100  # Número de pasos de Monte Carlo

# Simulación
spines_final, energias, magnetizaciones = simular_ising(N, T, pasos)

# Graficar resultados
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(energias, label="Energía promedio")
plt.xlabel("Paso")
plt.ylabel("Energía promedio")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(magnetizaciones, label="Magnetización promedio")
plt.xlabel("Paso")
plt.ylabel("Magnetización promedio")
plt.legend()


plt.suptitle(f"Simulación de Ising para N={N} y T={T}")

plt.tight_layout()
plt.show()

# Mostrar el spines final
plt.imshow(spines_final, cmap="Greys", interpolation="nearest")
plt.title(f"Configuración final del spines para T={T}")
plt.axis("off")
plt.colorbar(label="Spin")
plt.show()