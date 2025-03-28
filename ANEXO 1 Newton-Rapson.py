import matplotlib.pyplot as plt
import numpy as np

""" 
Anexo 1: Ejercicio 3) d) Newton-Raphson

Nota si no tiene las librerias instaladas ejecutar en consola el siguiente comando:
pip install numpy matplotlib

Simplemente ejecutar el código y se mostrará el gráfico de S(θ) vs θ.
"""
# xi | yi 
tabla_original= [[0.7,3.8],
                [11.3,4.6],
                [2.1,2.1],
                [30.7,5.6],
                [4.6, 10.3],
                [20.2, 2.8],
                [0.3, 1.9],
                [0.9, 1.4],
                [0.7, 0.4],
                [2.3, 0.9],
                [1.1, 2.8],
                [1.9, 3.2],
                [0.5, 8.5],
                [0.8, 14.5],
                [1.2, 14.4],
                [15.2, 8.8],
                [0.2, 7.6],
                [0.7, 1.3],
                [0.4, 2.2],
                [2.3, 4.0]]

# Calculo de los Ri
def actualizar_tabla_ri(tabla_original):
    tabla_ri = []
    for i in range(len(tabla_original)):
        xi = tabla_original[i][0]
        yi = tabla_original[i][1]
        ri = xi/yi
        tabla_ri.append([xi, yi, ri])
    return tabla_ri

# Obtener ri
def obtener_list_ri():
    tabla_ri = actualizar_tabla_ri(tabla_original)
    ri_list = [tabla_ri[i][2] for i in range(len(tabla_ri))]
    return ri_list

# Función s_teta
def s_teta(teta):
    tabla_ri = actualizar_tabla_ri(tabla_original)
    s_teta = 0
    for i in range(len(tabla_ri)):
        ri = tabla_ri[i][2]
        s_teta += (1-(teta*ri))/(teta + (ri*(teta**2)))
    return s_teta

# Función graficar
def graficar_S_vs_theta(min_theta=-50, max_theta=50, step=0.5):

    theta_vals = np.arange(min_theta, max_theta + step, step)
    S_vals = [s_teta(theta) for theta in theta_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(theta_vals, S_vals, color='blue', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$S(\theta)$')
    plt.title(r'Gráfico de $S(\theta)$ vs $\theta$')
    plt.grid(True)
    plt.ylim(-50, 40)
    plt.tight_layout()
    plt.show()
    
# Encontrar corte eje x
def encontrar_cortes_con_eje_x(min_theta=-50, max_theta=50, step=0.5):
    cortes = []
    theta_vals = np.arange(min_theta, max_theta + step, step)
    S_vals = [s_teta(theta) for theta in theta_vals]

    for i in range(1, len(theta_vals)):
        y1, y2 = S_vals[i - 1], S_vals[i]
        if y1 * y2 < 0:
            x1, x2 = theta_vals[i - 1], theta_vals[i]
            x0 = x1 - y1 * (x2 - x1) / (y2 - y1) 
            
            if x0 > 0:
                cortes.append(x0)

    return cortes

def newton_raphson_thetas(r_list, theta0, iteraciones=5):
    thetas = [theta0]

    for k in range(iteraciones):
        theta_k = thetas[-1]
        
        numerador = 0
        denominador = 0

        for i in range(20):
            arriba = (1-(theta_k*r_list[i]))
            abajo = (theta_k + (r_list[i]*(theta_k**2)))
            numerador += arriba/abajo
        
        for i in range(20):
            arriba = ((r_list[i]*(theta_k**2)) + (2*r_list[i]*theta_k) + 1)
            abajo = (theta_k + (r_list[i]*(theta_k**2)))**2
            denominador += arriba/abajo
            
        theta_next = theta_k + (numerador / denominador)
        thetas.append(theta_next)

    return thetas[1:]  

# Calcular los primeros 5 valores de θ
newton = newton_raphson_thetas(obtener_list_ri(), 2, 5)
print(newton)

#Punto razonable para θ0
ejex = encontrar_cortes_con_eje_x()
print(ejex)

#Graficar
graficar_S_vs_theta()
