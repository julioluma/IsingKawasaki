import numpy as np
import matplotlib.pyplot as plt
import os

# Crear la carpeta "Graficas" si no existe
if not os.path.exists("Graficas"):
    os.makedirs("Graficas")


# ================================================================================
# Parámetros
# Escoge qué quieres graficar frente a la temperatura
# 1: Magnetizacion promedio superior
# 2: Magnetizacion promedio inferior
# 3: Energia promedio
# 4: Calor especifico
# 5: Susceptibilidad magnetica

variable = 5 # Cambia este valor para graficar diferentes variables
Ns = [128, 64, 32]  # Lista de tamaños N para las simulaciones
Guardar = True  # Si es True, guarda la gráfica en un archivo
Intentos = 3 # Número de simulaciones realizadas
# ================================================================================


if variable ==1:
    nombre_variable = "Magnetización promedio superior"  
elif variable ==2:
    nombre_variable = "Magnetización promedio inferior"
elif variable ==3:
    nombre_variable = "Energía promedio"
elif variable ==4:
    nombre_variable = "Calor específico"  
elif variable ==5:
    nombre_variable = "Susceptibilidad magnética"
else:
    raise ValueError("Variable no válida. Debe ser un número entre 1 y 5.")

plt.figure(figsize=(8, 5))

for N in Ns:
    # Carga el primer archivo para saber cuántas temperaturas hay
    filename = f"Resultados/Resultados1_N={N}_pMc=100000_M=0.txt"
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    temperatura = data[:, 0]
    variable_data = np.zeros((len(temperatura), Intentos))
    for t in range(len(temperatura)):
        variable_data[t, 0] = data[t, variable]
    # Carga el resto de los intentos
    for intento in range(2, Intentos + 1):
        filename = f"Resultados/Resultados{intento}_N={N}_pMc=100000_M=0.txt"
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
        for t in range(len(temperatura)):
            variable_data[t, intento-1] = data[t, variable]
    s = np.std(variable_data, axis=1)
    mean_variable = np.mean(variable_data, axis=1)
    eb = plt.errorbar(temperatura, mean_variable, yerr=s, fmt='o', label=f'N={N}', capsize=5)
    color = eb.lines[0].get_color()
    plt.fill_between(temperatura, mean_variable - s, mean_variable + s, alpha=0.1, color=color)
    plt.plot(temperatura, mean_variable, linestyle='-', alpha=0.7, color=color)  # Línea que une los puntos
    
# Graficar
plt.xlabel("Temperatura")
plt.ylabel(nombre_variable)
plt.title(nombre_variable + " en función de la Temperatura")
plt.grid()
plt.legend()
if Guardar:
    # Guardar el gráfico
    output_filename = f"Graficas/{nombre_variable.replace(' ', '_')}_vs_Temperatura.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
plt.show()