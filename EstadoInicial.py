import numpy as np

#Pramétros
# ================================================================================
N=128 # Número de filas
M=0 # Magnetización promedio
Random=False # True: espines aleatorios, False: espines ordenados
# ================================================================================

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

def guardar_spines_txt(spines, filename="estado_inicial.txt"):
    # Limpia el archivo de salida antes de comenzar
    """Guarda el estado de los spines en un archivo de texto en el formato especificado."""
    with open(filename, "w") as f:  # Abre el archivo en modo de adición
        np.savetxt(f, spines, fmt="%d", delimiter=",")  # Guarda la matriz de espines
        f.write("\n")  # Agrega una línea en blanco para separar los pasos


spines= inicializar_spines(N, M)
guardar_spines_txt(spines, "estado_inicial.txt")
