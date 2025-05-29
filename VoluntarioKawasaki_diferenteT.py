import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import time
import os

# Crear la carpeta "Resultados" si no existe
if not os.path.exists("Resultados"):
    os.makedirs("Resultados")

#Crear la carpeta "spines" si no existe
if not os.path.exists("spines"):
    os.makedirs("spines")

# Parámetros
# ================================================================================
# ================================================================================
Random = False # True: espines aleatorios, False: espines ordenados
filename = "estado_inicial.txt" # Nombre del archivo de entrada, debe contener
                                # el estado inicial del sistema
                                # el formato debe ser el siguiente:
                                #   s(1,1), s(1,2), ..., s(1,N)
                                #   s(2,1), s(2,2), ..., s(2,N)
                                #   (...)
                                #   s(N,1), s(N,2), ..., s(N,N)
                                
Temperaturas = [0.5, 1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6] # Lista de temperaturas a simular
Ns = [128] # Lista de tamaños de retículo a simular
M = 0
pasos = 10**5  # Número de pasos de Monte Carlo
Guardar_spines = True  # Guardar el estado de los spines para animar
pasos_almacenamiento = 100  # Pasos para almacenar el estado de los spines
pasos_promediar = 100  # Pasos para promediar la energía y la magnetización

# ================================================================================
# ================================================================================
@nb.njit
def inicializar_spines(N, M):
    """Inicializa el spines con espines aleatorios (+1 o -1) y suma total igual a 0."""
    spines = np.empty((N, N), dtype=np.int32)
    
    # Configurar la primera fila como -1 y la última fila como 1
    spines[0, :] = -1
    spines[N-1, :] = 1
    
    if Random:
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
    else:
        # Inicializar las filas intermedias con valores ordenados
        for i in range(1, N-1):
            for j in range(N):
                spines[i, j] = 1 if i >= int(N*(M+1)/2) else -1

    return spines

"""def inicializar_spines(file="estado_inicial.txt"):
    # Lectura del fichero de datos
    with open(file, "r") as f:
        spines = np.loadtxt(f, delimiter=",", dtype=np.int32)  # Carga la matriz de espines
    N = spines.shape[0]  # Obtener el número de filas (N)
    M = np.sum(spines) / (N * N)  # Calcular M como la suma de espines dividido por el número total de espines
    return N, M, spines"""

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
def Kawasaki(spines, beta, N):
    """Implementa el algoritmo de Metropolis para actualizar el spines."""
    for _ in range((N-2) * N):  # N^2 intentos de actualización por paso
        i = np.random.randint(1, N-1)  # Seleccionar una fila intermedia
        j = np.random.randint(0, N)  # Seleccionar una columna
        S1 = spines[i, j]
        vecinos1 = spines[(i+1)%N, j] + spines[i, (j+1)%N] + spines[(i-1)%N, j] + spines[i, (j-1)%N] #+ spines[(i+1)%N, (j+1)%N] + spines[(i-1)%N, (j-1)%N] + spines[(i+1)%N, (j-1)%N] + spines[(i-1)%N, (j+1)%N]
        if 4 * S1 == vecinos1:
            continue  # Salir si no hay vecinos para intercambiar
        nran0= np.random.randint(0, 4)  # Seleccionar un vecino
        for vecino in range(4):
            nran = (nran0 + vecino) % 4
            if nran == 0:
                a, b = 1, 0
            elif nran == 1:
                a, b = 0, 1
            elif nran == 2:
                a, b = -1, 0
            elif nran == 3:
                a, b = 0, -1     
            i2 = (i + a) % N
            j2 = (j + b) % N
            S2 = spines[i2, j2]
            if S1 == S2:
                continue
            if i2 == 0 or i2 == N-1 :
                continue
            # Interacción con los vecinos
            vecinos2 = spines[(i2+1)%N, j2] + spines[i2, (j2+1)%N] + spines[(i2-1)%N, j2] + spines[i2, (j2-1)%N] #+ spines[(i2+1)%N, (j2+1)%N] + spines[(i2-1)%N, (j2-1)%N] + spines[(i2+1)%N, (j2-1)%N] + spines[(i2-1)%N, (j2+1)%N]
            dE = (S1-S2) * (vecinos1 - vecinos2 - S2 + S1)
            if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
                spines[i, j], spines[i2,j2] = spines[i2, j2], spines[i,j]  # Intercambiar los espines
                break  # Salir del bucle de intentos de intercambio
        
            

    return spines

def simular_ising(estadoinicial, T, pasos, pasos_almacenamiento=100, pasos_promediar=100):
    """Simula el modelo de Ising."""
    spines = estadoinicial
    beta = 1 / T
    if Guardar_spines:
        # Limpia el archivo de salida antes de comenzar
        open("spines/spinesN_" + str(N) + "_T=" + str(T) + ".txt", "w").close()
        # Guarda el estado inicial
        guardar_spines_txt(spines, 0, "spines/spinesN_" + str(N) + "_T=" + str(T) + ".txt")
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
        spines = Kawasaki(spines, beta, N)
        
        if paso % pasos_promediar == 0:
            energia = energia_total(spines)
            magnetizacion1, magnetizacion2 = magnetizacion_promedio(spines, M)
            energias.append(energia)
            magnetizaciones1.append(magnetizacion1)
            magnetizaciones2.append(magnetizacion2)
        if Guardar_spines:
            if paso % pasos_almacenamiento == 0:
                # Guardar el estado de los spines en cada x pasos
                guardar_spines_txt(spines, paso, "spines/spinesN_" + str(N) + "_T=" + str(T) + ".txt")
    M1 = np.mean(magnetizaciones1)
    M2 = np.mean(magnetizaciones2)
    Sus = np.var(magnetizaciones1)/(N1*N*T**2)
    E = np.mean(energias)/(N**2)
    Cv = np.var(energias)/(N**2*T**2)


    return spines, E, M1, M2, Cv, Sus



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

for N in Ns:
    inicio = time.time()

    EVec = []
    M1Vec = []
    M2Vec = []
    CvVec = []
    SusVec = []
    estadoinicial = inicializar_spines(N, M)
    print(f"Simulando para N={N}...")
    for T in Temperaturas:
        # Ejecutar la simulación
        print(f"Simulando para T={T}...")
        estadoinicial_copia = estadoinicial.copy()  # Copia del estado inicial para cada temperatura

        spines_final, E, M1, M2, Cv, Sus = simular_ising(estadoinicial_copia, T, pasos, pasos_almacenamiento, pasos_promediar)
        distribucion_densidad, densidad = dist_densidad(spines_final)
        Nvector = np.arange(0, N)

        EVec.append(E)
        M1Vec.append(M1)
        M2Vec.append(M2)
        CvVec.append(Cv)
        SusVec.append(Sus)


    # Guardar los resultados en un archivo de texto
    with open(f"Resultados/Resultados_N={N}_pMc={pasos}_M={M}.txt", "w") as f:
        f.write("Temperatura, Magnetizacion promedio superior, Magnetizacion promedio inferior, Energia promedio, Calor especifico, Susceptibilidad magnetica\n")
        for i in range(len(Temperaturas)):
            f.write(f"{Temperaturas[i]}, {M1Vec[i]}, {M2Vec[i]}, {EVec[i]}, {CvVec[i]}, {SusVec[i]}\n")

    fin = time.time()
    print(f"Simulación para N={N} finalizada en {fin - inicio:.2f} segundos.")