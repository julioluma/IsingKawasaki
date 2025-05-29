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

plt.figure(figsize=(8, 5))
for N in Ns:
    filename = f"Resultados/Resultados_N={N}_pMc=100000_M=0.txt"
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    
    temperatura = data[:, 0]  # Primera columna (Temperatura)
    
    variable_data = data[:, variable]  # Segunda columna (Magnetización promedio superior)
    
    plt.plot(temperatura, variable_data, marker='o', label=f'N={N}')

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