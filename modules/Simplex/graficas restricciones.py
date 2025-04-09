import matplotlib.pyplot as plt
import numpy as np

def graficar_restricciones(coeficientes, valores, num_variables, num_restricciones):
    x1_vals = np.linspace(0, 10, 400)
    
    plt.figure(figsize=(15, 10))
    
    for i in range(num_restricciones):
        coef = coeficientes[i]
        valor = valores[i]
        
        if num_variables == 2:
            restriccion = lambda x1: (valor - coef[0] * x1) / coef[1]
            plt.subplot(2, num_restricciones, i+1)
            plt.plot(x1_vals, restriccion(x1_vals), label=f'Restricción {i+1}')
            plt.fill_between(x1_vals, restriccion(x1_vals), alpha=0.3)
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.axhline(0, color='black', linewidth=0.5)
            plt.axvline(0, color='black', linewidth=0.5)
            plt.grid(color='gray', linestyle='--', linewidth=0.5)
            plt.legend()
            plt.title(f'Restricción {i+1}')
        else:
            print("Actualmente solo se pueden graficar restricciones con 2 variables de decisión.")
            return
    
    # Combinamos todas las restricciones en una sola gráfica
    plt.subplot(2, num_restricciones, num_restricciones+1)
    for i in range(num_restricciones):
        coef = coeficientes[i]
        valor = valores[i]
        restriccion = lambda x1: (valor - coef[0] * x1) / coef[1]
        plt.plot(x1_vals, restriccion(x1_vals), label=f'Restricción {i+1}')
    
    # Corregir el error en la función np.minimum.reduce
    restricciones = [lambda x1, coef=coeficientes[j], val=valores[j]: (val - coef[0] * x1) / coef[1] for j in range(num_restricciones)]
    min_restriccion = np.minimum.reduce([restriccion(x1_vals) for restriccion in restricciones])
    
    plt.fill_between(x1_vals, min_restriccion, alpha=0.3)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.title('Región factible combinada')
    
    # Mostramos la gráfica
    plt.tight_layout()
    plt.show()

def main():
    num_variables = int(input("Ingrese la cantidad de variables de decisión: "))
    num_restricciones = int(input("Ingrese la cantidad de restricciones: "))
    
    coeficientes = []
    valores = []
    
    for i in range(num_restricciones):
        coef = list(map(float, input(f"Ingrese los coeficientes de la restricción {i+1} separados por comas: ").split(',')))
        valor = float(input(f"Ingrese el valor de la restricción {i+1}: "))
        coeficientes.append(coef)
        valores.append(valor)
    
    graficar_restricciones(coeficientes, valores, num_variables, num_restricciones)

if __name__ == "__main__":
    main()
