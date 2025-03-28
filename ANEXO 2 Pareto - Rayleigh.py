import numpy as np
from scipy.stats import pareto, rayleigh
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

""" 
ANEXO 2: Pareto - Rayleigh

Solo tiene que dar clic en el botón de "Ejecutar" para visualizar los resultados. 
En el apartado de ejecución principal, puede ver la división de la parte a y b del ejercicio.

Nota: Si no tienes las librerías ejecutar en consola el siguiente comando:

pip install numpy scipy matplotlib seaborn pandas
"""

# Parte a: Simulación de estimadores
def generar_muestra(modelo: str, theta: float, n: int) -> np.ndarray:
    if modelo == "pareto":
        return pareto.rvs(b=theta, scale=1, size=n)  # f(x) = theta / x^{theta + 1}, x >= 1
    elif modelo == "rayleigh":
        return rayleigh.rvs(scale=theta, size=n)     # f(x) = (x/theta^2) * exp(-x^2/(2 theta^2))
    

def calcular_estimadores(modelo: str, muestra: np.ndarray) -> tuple[float, float]:
    """Calcula los estimadores de momentos (θ1) y EMV (θ2)."""
    if modelo == "pareto":
        # θ1 (momentos)
        mu = np.mean(muestra)
        theta1 = mu / (mu - 1)
        # θ2 (EMV)
        theta2 = len(muestra) / np.sum(np.log(muestra))
        return (theta1, theta2)
    
    elif modelo == "rayleigh":
        # θ1 (momentos)
        mu = np.mean(muestra)
        theta1 = mu / np.sqrt(np.pi / 2)
        # θ2 (EMV)
        theta2 = np.sqrt(np.sum(muestra**2) / (2 * len(muestra)))
        return (theta1, theta2)


def simulacion(modelo: str, theta: float, n_values: list[int], n_simulaciones: int = 500) -> dict[int, np.ndarray]:
    resultados = {}
    for n in n_values:
        estimadores = np.zeros((n_simulaciones, 2))  # Columnas: θ1, θ2
        for i in range(n_simulaciones):
            muestra = generar_muestra(modelo, theta, n)
            theta1, theta2 = calcular_estimadores(modelo, muestra)
            estimadores[i, :] = [theta1, theta2]
        resultados[n] = estimadores
    return resultados

# Parte b: Visualización con boxplots 
def preparar_dataframe(resultados_completos: dict) -> pd.DataFrame:
    datos = []
    for modelo in resultados_completos:
        for theta in resultados_completos[modelo]:
            for n in resultados_completos[modelo][theta]:
                matriz_estimadores = resultados_completos[modelo][theta][n]
                for i in range(matriz_estimadores.shape[0]):
                    datos.append({
                        "modelo": modelo,
                        "theta": theta,
                        "n": n,
                        "estimador": "θ1 (momentos)",
                        "valor": matriz_estimadores[i, 0]
                    })
                    datos.append({
                        "modelo": modelo,
                        "theta": theta,
                        "n": n,
                        "estimador": "θ2 (EMV)",
                        "valor": matriz_estimadores[i, 1]
                    })
    return pd.DataFrame(datos)

def graficar_boxplots(df: pd.DataFrame):
    sns.set(style="whitegrid")
    for modelo in df["modelo"].unique():
        for theta in df["theta"].unique():
            df_filtrado = df[(df["modelo"] == modelo) & (df["theta"] == theta)]
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=df_filtrado,
                x="n",
                y="valor",
                hue="estimador",
                palette={"θ1 (momentos)": "skyblue", "θ2 (EMV)": "salmon"}
            )
            plt.title(f"Distribución de estimadores - {modelo.capitalize()} (θ = {theta})")
            plt.xlabel("Tamaño de muestra (n)")
            plt.ylabel("Valor del estimador")
            plt.axhline(y=theta, color="red", linestyle="--", label="Valor verdadero")
            plt.legend()
            plt.show()

# Parte C: Funciones para calcular sesgo, MSE y eficiencia
def calcular_sesgo_mse(resultados_completos: dict, theta_verdadero: float) -> dict:
   
    metricas = {}
    for modelo in resultados_completos:
        metricas[modelo] = {}
        for theta in resultados_completos[modelo]:
            if theta != theta_verdadero:
                continue  
            
            metricas[modelo][theta] = {}
            for n in resultados_completos[modelo][theta]:
                estimadores = resultados_completos[modelo][theta][n]
                theta1 = estimadores[:, 0]  
                theta2 = estimadores[:, 1]  
                
                # Calcular sesgo y MSE
                sesgo_theta1 = np.mean(theta1) - theta
                sesgo_theta2 = np.mean(theta2) - theta
                mse_theta1 = np.mean((theta1 - theta) ** 2)
                mse_theta2 = np.mean((theta2 - theta) ** 2)
                
                metricas[modelo][theta][n] = {
                    "sesgo_theta1": sesgo_theta1,
                    "sesgo_theta2": sesgo_theta2,
                    "mse_theta1": mse_theta1,
                    "mse_theta2": mse_theta2,
                    "eff": mse_theta2 / mse_theta1  
                }
    return metricas

def graficar_mse_eff(metricas: dict, theta_verdadero: float):
    
    sns.set(style="whitegrid")
    modelos = list(metricas.keys())
    n_values = sorted(list(metricas[modelos[0]][theta_verdadero].keys()))
    
    # Configurar figura
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Graficar MSE
    for modelo in modelos:
        mse_theta1 = [metricas[modelo][theta_verdadero][n]["mse_theta1"] for n in n_values]
        mse_theta2 = [metricas[modelo][theta_verdadero][n]["mse_theta2"] for n in n_values]
        axes[0].plot(n_values, mse_theta1, 'o-', label=f"{modelo} - θ1 (momentos)")
        axes[0].plot(n_values, mse_theta2, 's-', label=f"{modelo} - θ2 (EMV)")
    
    axes[0].set_title(f"MSE de los estimadores (θ = {theta_verdadero})")
    axes[0].set_xlabel("Tamaño de muestra (n)")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True)
    
    # Graficar eficiencia relativa
    for modelo in modelos:
        eff = [metricas[modelo][theta_verdadero][n]["eff"] for n in n_values]
        axes[1].plot(n_values, eff, 'o-', label=modelo)
    
    axes[1].axhline(y=1, color='red', linestyle='--', label="Línea de referencia (Eff = 1)")
    axes[1].set_title(f"Eficiencia relativa Eff(θ1, θ2) (θ = {theta_verdadero})")
    axes[1].set_xlabel("Tamaño de muestra (n)")
    axes[1].set_ylabel("Eff(θ1, θ2)")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    
    
# --- Ejecución principal ---
if __name__ == "__main__":
    modelos = ["pareto", "rayleigh"]
    thetas = [2.3, 8.0]
    n_values = [20, 50, 200, 500, 1000]
    
    # Parte a: Simulación
    resultados_completos = {}
    for modelo in modelos:
        resultados_completos[modelo] = {}
        for theta in thetas:
            print(f"Simulando: modelo={modelo}, theta={theta}")
            resultados_completos[modelo][theta] = simulacion(modelo, theta, n_values)
    
    # Parte b: Graficar
    df = preparar_dataframe(resultados_completos)
    graficar_boxplots(df)
    
    # Parte c: Calcular sesgo, MSE y eficiencia
    metricas_theta2_3 = calcular_sesgo_mse(resultados_completos, theta_verdadero=2.3)
    metricas_theta8 = calcular_sesgo_mse(resultados_completos, theta_verdadero=8.0)
    
    # Graficar MSE y eficiencia relativa
    print("Resultados para θ = 2.3:")
    graficar_mse_eff(metricas_theta2_3, theta_verdadero=2.3)
    
    print("Resultados para θ = 8.0:")
    graficar_mse_eff(metricas_theta8, theta_verdadero=8.0)
    
    # Imprimir ARE teórico (comparación)
    print("\nARE teórico (asintótico):")
    print("Pareto: ARE(θ1, θ2) = (θ^2) / (θ^2 (θ-1)^4 / (θ-2)) = (θ-2) / (θ-1)^4")
    print(f"  Para θ=2.3: ARE = {(2.3 - 2) / (2.3 - 1)**4:.4f}")
    print(f"  Para θ=8.0: ARE = {(8.0 - 2) / (8.0 - 1)**4:.4f}")
    print("\nRayleigh: ARE(θ1, θ2) = (θ^2/2) / ((4-π)θ^2/2) = 1 / (4-π) ≈ 0.655")