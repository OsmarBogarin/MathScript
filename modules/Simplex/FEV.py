import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import linalg
from scipy.spatial import ConvexHull

def crear_problema():
    """Permite al usuario definir la función objetivo y puntos FEV"""
    print("\n===== CONFIGURACIÓN DEL PROBLEMA DE PROGRAMACIÓN LINEAL =====")
    
    # Elegir tipo de optimización
    tipo = input("\n¿Desea maximizar (max) o minimizar (min)?: ").lower().strip()
    while tipo not in ["max", "min"]:
        tipo = input("\nPor favor, ingrese 'max' para maximizar o 'min' para minimizar: ").lower().strip()
    
    # Número de variables de decisión
    num_vars = int(input("\nIngrese el número de variables de decisión (2-3): "))
    while num_vars < 2 or num_vars > 3:
        num_vars = int(input("\nPor favor, ingrese un valor entre 2 y 3: "))
    
    # Función objetivo
    print("\n📝 Definición de la función objetivo:")
    func_obj = []
    for i in range(num_vars):
        coef = float(input(f"\nCoeficiente para x{i+1}: "))
        func_obj.append(coef)
    
    # Puntos FEV
    num_fevs = int(input("\nIngrese el número de puntos FEV: "))
   # Inicializar la lista con el punto (0, 0) como una tupla
    puntos_fev = [(0, 0)]

    # Preguntar cuántos puntos adicionales se desean
    num_fevs = int(input("\nIngrese el número de puntos FEV: "))

    # Recibir los puntos ingresados por el usuario
    for i in range(num_fevs):
        print(f"\nPunto FEV {i + 1}:")
        punto = []
        for j in range(2):  # Asumimos que siempre tienes 2 variables (x e y)
            valor = float(input(f"x{j + 1}: "))
            punto.append(valor)
        puntos_fev.append(tuple(punto))  # Convertir la lista a una tupla antes de añadir

    return tipo, num_vars, func_obj, puntos_fev

def determinar_restricciones_2d(puntos_fev):
    """
    Determina las restricciones utilizando la envolvente convexa.
    """
    # Convertir los puntos FEV a un array numpy
    puntos = np.array(puntos_fev)
    restricciones = []

    # Calcular la envolvente convexa
    hull = ConvexHull(puntos)

    # Generar restricciones a partir de las facetas de la envolvente
    for simplex in hull.simplices:
        p1, p2 = puntos[simplex[0]], puntos[simplex[1]]
        print(f"\nProcesando segmento: {p1}, {p2}")

        # Calcular la ecuación de la recta (a*x1 + b*x2 = c)
        a = p2[1] - p1[1]  # Diferencia en y
        b = p1[0] - p2[0]  # Diferencia en x
        c = a * p1[0] + b * p1[1]  # Constante
        print("Calcular la ecuación de la recta (a*x1 + b*x2 = c)\n")
        print(f"a=Diferencia en y : {p2[1]} - {p1[1]}")
        print(f"b=Diferencia en x : {p1[0]} - {p2[0]}")
        print(f"({a}*{p1[0]}+{b}*{p1[1]} = c)\nc={c}")
        
        # Normalizar para convertir en desigualdad
        rhs = c
        desc = f"{a:.2f}x₁ + {b:.2f}x₂ ≤ {rhs:.2f}"
        restricciones.append({"coeficientes": [a, b], "op": "<=", "rhs": rhs, "descripcion": desc})
        print(f"\nRestricción agregada: {desc}")
    return restricciones

def determinar_restricciones_3d(puntos_fev):
    """Determina las restricciones para un problema 3D a partir de los puntos FEV"""
    restricciones = []
    n = len(puntos_fev)
    
    # Agregar restricciones x1 >= 0, x2 >= 0, x3 >= 0 si hay puntos en los planos
    if any(punto[0] == 0 for punto in puntos_fev):
        restricciones.append({"coeficientes": [1, 0, 0], "op": ">=", "rhs": 0, "descripcion": "x₁ ≥ 0"})
    if any(punto[1] == 0 for punto in puntos_fev):
        restricciones.append({"coeficientes": [0, 1, 0], "op": ">=", "rhs": 0, "descripcion": "x₂ ≥ 0"})
    if any(punto[2] == 0 for punto in puntos_fev):
        restricciones.append({"coeficientes": [0, 0, 1], "op": ">=", "rhs": 0, "descripcion": "x₃ ≥ 0"})
    
    # Encontrar restricciones probando combinaciones de 3 puntos
    for puntos in combinations(puntos_fev, 3):
        p1, p2, p3 = puntos
        
        # Intentar encontrar un plano que pase por estos 3 puntos
        try:
            # Resolver el sistema a*x + b*y + c*z = 1 para los tres puntos
            A = np.array([[p1[0], p1[1], p1[2]], 
                         [p2[0], p2[1], p2[2]], 
                         [p3[0], p3[1], p3[2]]])
            b = np.ones(3)
            
            # Si los puntos son colineales o casi colineales, esto fallará
            abc = linalg.solve(A, b)
            a, b, c = abc[0], abc[1], abc[2]
            
            # Verificar si todos los demás puntos están en el mismo semiplano
            lado_positivo = 0
            lado_negativo = 0
            epsilon = 1e-10
            
            for punto in puntos_fev:
                val = a * punto[0] + b * punto[1] + c * punto[2] - 1
                if abs(val) < epsilon:  # El punto está en el plano
                    continue
                elif val > 0:
                    lado_positivo += 1
                else:
                    lado_negativo += 1
            
            # Si todos los puntos están en un lado del plano o en el plano mismo
            if lado_positivo == 0 or lado_negativo == 0:
                # Normalizar para tener la forma ax + by + cz ≤ d
                if lado_positivo == 0:  # Todos los puntos están en el lado negativo del plano
                    d = 1
                else:
                    a, b, c = -a, -b, -c
                    d = -1
                
                # Crear la descripción de la restricción
                desc = f"{a:.2f}x₁ + {b:.2f}x₂ + {c:.2f}x₃ ≤ {d:.2f}"
                restricciones.append({"coeficientes": [a, b, c], "op": "<=", "rhs": d, "descripcion": desc})
        except:
            # Los puntos son colineales o hay otro problema numérico, ignorar
            pass
    
    return restricciones

def evaluar_funcion_objetivo(func_obj, puntos_fev):
    """Evalúa la función objetivo en cada punto FEV y encuentra el óptimo"""
    resultados = []
    
    for i, punto in enumerate(puntos_fev):
        # Calcular Z = c1*x1 + c2*x2 + ...
        z = sum(c * x for c, x in zip(func_obj, punto))
        resultados.append((i, punto, z))
    
    return resultados

def encontrar_optimo(tipo, resultados):
    """Encuentra el punto óptimo según el tipo de problema (max/min)"""
    if tipo.lower() == "max":
        optimo = max(resultados, key=lambda x: x[2])
    else:
        optimo = min(resultados, key=lambda x: x[2])
    
    return optimo

def graficar_problema_2d(puntos_fev, restricciones, func_obj, tipo, optimo):
    """Grafica el problema de programación lineal en 2D"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Graficar las restricciones
    x_vals = np.linspace(0, max([p[0] for p in puntos_fev]) * 1.2 + 1, 400)

    # Filtrar restricciones cuyo rhs sea diferente de 0
    # restricciones_validas = [r for r in restricciones if r["rhs"] != 0]

    for i, r in enumerate(restricciones):
        coef = r["coeficientes"]
        if len(coef) == 2 and coef[1] != 0:  # Solo si no es vertical
            a, b = coef
            rhs = r["rhs"]
            y_vals = (rhs - a * x_vals) / b
            ax.plot(x_vals, y_vals, linestyle='--', label=f"R{i+1}: {r['descripcion']}")
        elif len(coef) == 2 and coef[1] == 0:  # Línea vertical
            x_const = r["rhs"] / coef[0]
            ax.axvline(x_const, linestyle='--', label=f"R{i+1}: {r['descripcion']}")



    # Graficar los puntos FEV
    x_points = [p[0] for p in puntos_fev]
    y_points = [p[1] for p in puntos_fev]
    ax.scatter(x_points, y_points, color='blue', s=100, label='Puntos FEV')
    
    # Etiquetar los puntos
    for i, punto in enumerate(puntos_fev):
        ax.annotate(f"FEV{i+1} ({punto[0]}, {punto[1]})", 
                    (punto[0], punto[1]), 
                    xytext=(5, 5), 
                    textcoords='offset points')
    
    # Resaltar el punto óptimo
    ax.scatter([optimo[1][0]], [optimo[1][1]], color='red', s=150, edgecolor='black', label='Punto Óptimo')
    
    # Dibujar la región factible (aproximada)
    if len(puntos_fev) > 2:
        # Obtener el centro de los puntos
        centro_x = sum(x_points) / len(x_points)
        centro_y = sum(y_points) / len(y_points)
        
        # Ordenar los puntos según el ángulo desde el centro
        puntos_ordenados = sorted(puntos_fev, 
                                 key=lambda p: np.arctan2(p[1] - centro_y, p[0] - centro_x))
        
        # Cerrar el polígono
        puntos_ordenados.append(puntos_ordenados[0])
        
        # Dibujar el polígono
        x_poly = [p[0] for p in puntos_ordenados]
        y_poly = [p[1] for p in puntos_ordenados]
        ax.fill(x_poly, y_poly, alpha=0.2, color='blue', label='Región Factible')
    
    # Configurar el gráfico
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    
    # Calcular límites adecuados
    max_x = max(x_points) * 1.2
    max_y = max(y_points) * 1.2
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    
    # Añadir ejes
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    # Añadir título
    ax.set_title(f"{tipo.upper()}IMIZACIÓN Z = {func_obj[0]}$x_1$ + {func_obj[1]}$x_2$", fontsize=14)
    
    # Añadir leyenda
    ax.legend(loc='best')
    
    # Añadir cuadrícula
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('problema_pl_fev.png')
    plt.show()
    
    return fig

def graficar_problema_3d(puntos_fev, restricciones, func_obj, tipo, optimo):
    """Grafica el problema de programación lineal en 3D"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar los puntos FEV
    x_points = [p[0] for p in puntos_fev]
    y_points = [p[1] for p in puntos_fev]
    z_points = [p[2] for p in puntos_fev]
    ax.scatter(x_points, y_points, z_points, color='blue', s=100, label='Puntos FEV')
    
    # Etiquetar los puntos
    for i, punto in enumerate(puntos_fev):
        ax.text(punto[0], punto[1], punto[2], f"FEV{i+1}")
    
    # Resaltar el punto óptimo
    ax.scatter([optimo[1][0]], [optimo[1][1]], [optimo[1][2]], color='red', s=150, label='Punto Óptimo')

    # Configurar el gráfico
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_zlabel('$x_3$', fontsize=12)
    
    # Añadir título
    coef_str = ' + '.join([f"{c}$x_{i+1}$" for i, c in enumerate(func_obj)])
    ax.set_title(f"{tipo.upper()}IMIZACIÓN Z = {coef_str}", fontsize=14)
    
    # Añadir leyenda
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('problema_pl_fev_3d.png')
    plt.show()
    
    return fig

def main():
    print("\n" + "="*60)
    print(" 📊 ANÁLISIS DE PROGRAMACIÓN LINEAL CON PUNTOS FEV")
    print("="*60)
    
    # Solicitar la información al usuario o usar parámetros predeterminados
    use_default = input("\n¿Desea usar el ejemplo predeterminado? (s/n): ").lower() == 's'
    
    if use_default:
        tipo = "max"
        num_vars = 2
        func_obj = [1000, 2000]
        puntos_fev = [[0, 0], [8, 0], [6, 4], [5, 5], [0, 6.666666667]]
    else:
        # Solicitar información al usuario
        tipo, num_vars, func_obj, puntos_fev = crear_problema()
    
    # Mostrar resumen del problema
    print("\n" + "-"*60)
    print(" 📋 RESUMEN DEL PROBLEMA")
    print("-"*60)
    print(f"• Tipo de optimización: {tipo.upper()}IMIZAR")
    print(f"• Función objetivo: Z = {' + '.join([f'{c}x{i+1}' for i, c in enumerate(func_obj)])}")
    print("• Puntos FEV:")
    for i, punto in enumerate(puntos_fev):
        print(f"  - FEV{i+1}: ({', '.join([str(coord) for coord in punto])})")
    
    # Determinar restricciones del problema
    if num_vars == 2:
        restricciones = determinar_restricciones_2d(puntos_fev)
    else:
        restricciones = determinar_restricciones_3d(puntos_fev)
    
    # Mostrar restricciones
    print("\n" + "-"*60)
    print(" 📏 RESTRICCIONES IDENTIFICADAS")
    print("-"*60)
     # Filtrar restricciones cuyo rhs sea diferente de 0
     # restricciones_validas = [r for r in restricciones if r["rhs"] != 0]

    for i, r in enumerate(restricciones):
        print(f"R{i+1}: {r['descripcion']}")
    
    # Evaluar la función objetivo en cada punto FEV
    resultados = evaluar_funcion_objetivo(func_obj, puntos_fev)
    
    # Mostrar evaluación de la función objetivo
    print("\n" + "-"*60)
    print(" 🎯 EVALUACIÓN DE LA FUNCIÓN OBJETIVO")
    print("-"*60)
    for i, punto, z in resultados:
        print(f"• FEV{i+1} {punto}: Z = {z}")
    
    # Encontrar el punto óptimo
    optimo = encontrar_optimo(tipo, resultados)
    print("\n" + "-"*60)
    print(f" 🏆 SOLUCIÓN {tipo.upper()}")
    print("-"*60)
    print(f"• Punto óptimo: FEV{optimo[0]+1} {optimo[1]}")
    print(f"• Valor óptimo: Z = {optimo[2]}")
    
    # Graficar el problema
    print("\n" + "-"*60)
    print(" 📈 VISUALIZACIÓN GRÁFICA")
    print("-"*60)
    
    if num_vars == 2:
        graficar_problema_2d(puntos_fev, restricciones, func_obj, tipo, optimo)
    else:
        graficar_problema_3d(puntos_fev, restricciones, func_obj, tipo, optimo)
    
    print("\n¡Análisis completado! 👍")

if __name__ == "__main__":
    main()