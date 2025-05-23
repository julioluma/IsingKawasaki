import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import time
import os

# Parámetros
# ================================================================================
# ================================================================================
filename = "estado_inicial.txt" # Nombre del archivo de entrada, debe contener
                                # el estado inicial del sistema
                                # el formato debe ser el siguiente:
                                #   s(1,1), s(1,2), ..., s(1,N)
                                #   s(2,1), s(2,2), ..., s(2,N)
                                #   (...)
                                #   s(N,1), s(N,2), ..., s(N,N)
                                
Temperaturas = [1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.5, 4] # Lista de temperaturas a simular
pasos = 10**6  # Número de pasos de Monte Carlo
Guardar_spines = False  # Guardar el estado de los spines para animar
pasos_almacenamiento = 100  # Pasos para almacenar el estado de los spines
pasos_promediar = 100  # Pasos para promediar la energía y la magnetización

# ================================================================================
# ================================================================================


"""def inicializar_spines(N, M):
    spines = np.empty((N, N), dtype=np.int32)
    
    # Configurar la primera fila como -1 y la última fila como 1
    spines[0, :] = -1
    spines[N-1, :] = 1
    
    # Inicializar las filas intermedias con valores aleatorios
    for i in range(1, N-1):
        for j in range(N):
            spines[i, j] = 1 if np.random.rand() > 0.5 else -1
    
    # Ajustar la suma para que sea 0
    while np.sum(spines) != int(M*N*N):
        # Seleccionar una posición aleatoria en las filas intermedias
        i = np.random.randint(1, N-1)
        j = np.random.randint(0, N)
        
        # Cambiar el signo del elemento seleccionado
        spines[i, j] *= -1

    return spines"""

def inicializar_spines(file="estado_inicial.txt"):
    # Lectura del fichero de datos
    with open(file, "r") as f:
        spines = np.loadtxt(f, delimiter=",", dtype=np.int32)  # Carga la matriz de espines
    N = spines.shape[0]  # Obtener el número de filas (N)
    M = np.sum(spines) / (N * N)  # Calcular M como la suma de espines dividido por el número total de espines
    return N, M, spines

@nb.njit  # Decorador para compilar la función con Numba
def energia_total(spines):
    """Calcula la energía promedio del sistema."""
    N = spines.shape[0]
    energia = 0
    for i in range(N):
        for j in range(N):
            S = spines[i, j]
            # Interacción con los vecinos
            vecinos = spines[(i+1)%N, j] + spines[i, (j+1)%N] + spines[(i-1)%N, j] + spines[i, (j-1)%N]
            energia += -S * vecinos
                    
    return energia/(2)  # Dividir por 2 para evitar contar cada interacción dos veces


@nb.njit  
def magnetizacion_promedio(spines, M):
    """Calcula la magnetización promedio del sistema, arriba y abajo"""
    N= spines.shape[0]
    suma1 = 0
    suma2 = 0
    N1 = int(N*(1+M)/2)
    N2 = int(N*(1-M)/2)
    for i in range(0, N1):
        for j in range(0, N):
            suma1 += spines[i, j]
    for i in range(0, N2):
        for j in range(0, N):
            suma2 += spines[N1+i, j]
    return suma1/(N1*N), suma2/(N2*N)



    

def guardar_spines_txt(spines, paso, filename="estados.txt"):
    """Guarda el estado de los spines en un archivo de texto en el formato especificado."""
    with open(filename, "a") as f:  # Abre el archivo en modo de adición
        np.savetxt(f, spines, fmt="%d", delimiter=",")  # Guarda la matriz de espines
        f.write("\n")  # Agrega una línea en blanco para separar los pasos

@nb.njit
def Kawasaki(spines, beta):
    """Implementa el algoritmo de Metropolis para actualizar el spines."""
    N = spines.shape[0]
    for _ in range((N-2) * N):  # N^2 intentos de actualización por paso
        i = np.random.randint(1, N-1)  # Seleccionar una fila intermedia
        j = np.random.randint(0, N)  # Seleccionar una columna
        S1 = spines[i, j]
        vecinos1 = spines[(i+1)%N, j] + spines[i, (j+1)%N] + spines[(i-1)%N, j] + spines[i, (j-1)%N] #+ spines[(i+1)%N, (j+1)%N] + spines[(i-1)%N, (j-1)%N] + spines[(i+1)%N, (j-1)%N] + spines[(i-1)%N, (j+1)%N]
        for _ in range(4):  # intentos de intercambio
            if 4 * S1 == vecinos1:
                break  # Salir si no hay vecinos para intercambiar
            a= np.random.randint(-1, 2)  # Seleccionar un vecino
            b= np.random.randint(-1, 2)  # Seleccionar un vecino
            if a == 0 and b == 0:
                continue           
            i2 = (i + a) % N
            j2 = (j + b) % N
            #i2 = np.random.randint(1, N-1)  # Seleccionar una fila intermedia
            #j2 = np.random.randint(0, N)  # Seleccionar una columna
            if S1 == spines[i2, j2]:
                continue
            if i2 == 0 or i2 == N-1 :
                continue
            # Interacción con los vecinos
            S2 = spines[i2, j2]
            vecinos2 = spines[(i2+1)%N, j2] + spines[i2, (j2+1)%N] + spines[(i2-1)%N, j2] + spines[i2, (j2-1)%N] #+ spines[(i2+1)%N, (j2+1)%N] + spines[(i2-1)%N, (j2-1)%N] + spines[(i2+1)%N, (j2-1)%N] + spines[(i2-1)%N, (j2+1)%N]
            dE = (S1-S2)*(vecinos1 - vecinos2 - S2 + S1)
            if dE < 0 or np.random.rand() < np.exp(-beta * dE):
                aux = spines[i, j]
                spines[i, j] = spines[i2, j2]
                spines[i2, j2] = aux
                break  # Salir del bucle de intentos de intercambio
            

    return spines

def simular_ising(filename, T, pasos, pasos_almacenamiento=100, pasos_promediar=100):
    """Simula el modelo de Ising."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"El archivo {filename} no existe.")
    else:

        beta = 1 / T
        N, M, spines = inicializar_spines(filename)

        if Guardar_spines:
            # Limpia el archivo de salida antes de comenzar
            open("estados.txt", "w").close()
            # Guarda el estado inicial
            guardar_spines_txt(spines, 0)

        energias = []
        magnetizaciones1 = []
        magnetizaciones2 = []
        M1 = []
        M2 = []
        E = []
        Cv = []
        Sus = []
        N1 = int(N*(1+M)/2)

        for paso in range(pasos):
            # Actualizar el porcentaje completado
            porcentaje = (paso + 1) / pasos * 100
            print(f"Progreso: {porcentaje:.2f}%", end="\r")  # Sobrescribe la línea anterior


            spines = Kawasaki(spines, beta)
            
            if paso % pasos_promediar == 0:
                energia = energia_total(spines)
                magnetizacion1, magnetizacion2 = magnetizacion_promedio(spines, M)
                energias.append(energia)
                magnetizaciones1.append(magnetizacion1)
                magnetizaciones2.append(magnetizacion2)

            if Guardar_spines:
                if paso % pasos_almacenamiento == 0:
                    # Guardar el estado de los spines en cada x pasos
                    guardar_spines_txt(spines, paso)

        M1 = np.mean(magnetizaciones1)
        M2 = np.mean(magnetizaciones2)
        Sus = np.var(magnetizaciones1)/(N1*N*T)
        E = np.mean(energias)/N**2
        Cv = np.var(energias)/N**2*T**2


    return N, M, spines, E, M1, M2, Cv, Sus



@nb.njit
def dist_densidad(spines):
    """Calcula la densidad de espines en el sistema."""
    N = spines.shape[0]
    densidad = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if spines[j, i] == 1:
                densidad[i] += 1
    """Calcula la distribución de densidades."""
    dist = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i == densidad[j]:
                dist[i] += 1
    for i in range(N):
        dist[i] = dist[i] / N
    return dist, densidad
# ================================================================================
inicio = time.time()
EVec = []
M1Vec = []
M2Vec = []
CvVec = []
SusVec = []


for T in Temperaturas:
    # Ejecutar la simulación
    print(f"Simulando para T={T}...")
    

    N, M, spines_final, E, M1, M2, Cv, Sus = simular_ising(filename, T, pasos, pasos_almacenamiento, pasos_promediar)
    distribucion_densidad, densidad = dist_densidad(spines_final)
    Nvector = np.arange(0, N)

    EVec.append(E)
    M1Vec.append(M1)
    M2Vec.append(M2)
    CvVec.append(Cv)
    SusVec.append(Sus)


fin = time.time()
# Mostrar el tiempo de ejecución
print(f"Tiempo de ejecución: {fin - inicio:.2f} s")
# Graficar resultados
ax = plt.figure(figsize=(12, 5))
ax.suptitle(f"N={N}, T={T}, pMc={pasos}, M={M}\n Tiempo de ejecución: {fin - inicio:.2f} s")
plt.subplot(2, 2, 1)
plt.plot(Temperaturas, M1Vec, label="Magnetización promedio superior")
plt.plot(Temperaturas, M2Vec, label="Magnetización promedio inferior")
plt.xlabel("Temperatura")
plt.ylabel("Magnetización")
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(Temperaturas, EVec, label="Energía promedio de una partícula")
plt.xlabel("Temperatura")
plt.ylabel(r"$\varepsilon$")
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(Temperaturas, SusVec, label="Susceptibilidad magnética")
plt.xlabel("Temperatura")
plt.ylabel(r"$\chi_N$")
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(Temperaturas, CvVec, label="Calor específico")
plt.xlabel("Temperatura")
plt.ylabel(r"$c_v$")
plt.legend()
# Crear la carpeta "Figuras" si no existe
if not os.path.exists("Figuras"):
    os.makedirs("Figuras")
# Guardar el gráfico en la carpeta "Figuras"
plt.tight_layout()
plt.savefig(f"Figuras/Ordenado_N={N}_pMc={pasos}_M={M}.png", dpi=300, bbox_inches="tight")
plt.show()