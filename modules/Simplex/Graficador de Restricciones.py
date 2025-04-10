SCRIPT_DESCRIPTION = """
## Graficador de Restricciones

Este script permite visualizar restricciones de programación lineal en el plano.
Para cada restricción ingresada, se dibuja una gráfica individual mostrando
las soluciones no negativas que la satisfacen, y una gráfica combinada
mostrando la región factible.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def crear_problema():
    """Permite al usuario definir las restricciones de programación lineal"""
    print("\n===== CONFIGURACIÓN DE RESTRICCIONES =====")
    
    # Número de variables de decisión
    num_variables = int(input("\nIngrese el número de variables de decisión (x1, x2, ...): "))
    while num_variables <= 0:
        num_variables = int(input("\nEl número de variables debe ser positivo. Ingrese nuevamente: "))
    
    # Número de restricciones
    num_restricciones = int(input("\nIngrese el número de restricciones: "))
    while num_restricciones <= 0:
        num_restricciones = int(input("\nEl número de restricciones debe ser positivo. Ingrese nuevamente: "))
    
    # Restricciones
    coeficientes = []
    valores = []
    
    print("\nDefinición de restricciones:")
    for i in range(num_restricciones):
        print(f"\nRestricción {i+1}:")
        coefs = []
        for j in range(num_variables):
            coef = float(input(f"Coeficiente para x{j+1}: "))
            coefs.append(coef)
        coeficientes.append(coefs)
        
        valor = float(input("Lado derecho de la restricción: "))
        valores.append(valor)
    
    return num_variables, num_restricciones, coeficientes, valores

def visualizar_restriccion_individual(coef, valor, restriccion_num, x_max=10):
    """Visualiza una restricción individual"""
    if len(coef) != 2:
        print(f"No se puede graficar restricción {restriccion_num} (necesita exactamente 2 variables).")
        return
    
    x1_vals = np.linspace(0, x_max, 400)
    
    # Verificar si el coeficiente de x2 es cero (evitar división por cero)
    if coef[1] == 0:
        # Caso especial: línea vertical
        if coef[0] == 0:
            # Caso especial: 0x1 + 0x2 ≤ valor
            if valor >= 0:
                # Todo el espacio es factible o nada es factible
                plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                plt.fill_between(x1_vals, 0, x_max, alpha=0.3, color='blue')
            return
        # Caso x1 = valor/coef[0]
        x_limite = valor / coef[0]
        plt.axvline(x=x_limite, color='red', label=f'{coef[0]}x₁ ≤ {valor}')
        # Sombrear región factible
        if coef[0] > 0:
            plt.fill_betweenx([0, x_max], 0, x_limite, alpha=0.3, color='blue')
        else:
            plt.fill_betweenx([0, x_max], x_limite, x_max, alpha=0.3, color='blue')
    else:
        # Caso normal: función lineal
        y_vals = (valor - coef[0] * x1_vals) / coef[1]
        plt.plot(x1_vals, y_vals, color='red', label=f'{coef[0]}x₁ + {coef[1]}x₂ ≤ {valor}')
        
        # Sombrear región factible
        if coef[1] > 0:
            plt.fill_between(x1_vals, 0, y_vals, where=(y_vals >= 0), alpha=0.3, color='blue')
        else:
            plt.fill_between(x1_vals, y_vals, x_max, where=(y_vals <= x_max), alpha=0.3, color='blue')
    
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, x_max)
    plt.ylim(0, x_max)
    plt.title(f'Restricción {restriccion_num}: {coef[0]}x₁ + {coef[1]}x₂ ≤ {valor}')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linewidth=0.5)
    plt.legend()

def visualizar_region_factible(coeficientes, valores, x_max=10):
    """Visualiza la región factible definida por todas las restricciones"""
    if any(len(coef) != 2 for coef in coeficientes):
        print("No se puede graficar la región factible (todas las restricciones deben tener exactamente 2 variables).")
        return
    
    x1_vals = np.linspace(0, x_max, 400)
    
    # Graficamos cada restricción
    for i, (coef, valor) in enumerate(zip(coeficientes, valores)):
        if coef[1] == 0:
            # Restricción vertical
            if coef[0] != 0:
                x_limite = valor / coef[0]
                if 0 <= x_limite <= x_max:
                    plt.axvline(x=x_limite, color=f'C{i}', label=f'R{i+1}: {coef[0]}x₁ ≤ {valor}')
        else:
            # Restricción normal
            y_vals = (valor - coef[0] * x1_vals) / coef[1]
            plt.plot(x1_vals, y_vals, color=f'C{i}', label=f'R{i+1}: {coef[0]}x₁ + {coef[1]}x₂ ≤ {valor}')
    
    # Encontrar vértices de la región factible
    vertices = []
    
    # Añadir origen si satisface todas las restricciones
    if all((coef[0] * 0 + coef[1] * 0) <= valor for coef, valor in zip(coeficientes, valores)):
        vertices.append((0, 0))
    
    # Intersecciones con los ejes
    for i, (coef, valor) in enumerate(zip(coeficientes, valores)):
        # Intersección con eje x (x₂ = 0)
        if coef[0] != 0:
            x = valor / coef[0]
            if x >= 0:
                punto = (x, 0)
                if all((c[0] * punto[0] + c[1] * punto[1]) <= v for c, v in zip(coeficientes, valores)):
                    vertices.append(punto)
        
        # Intersección con eje y (x₁ = 0)
        if coef[1] != 0:
            y = valor / coef[1]
            if y >= 0:
                punto = (0, y)
                if all((c[0] * punto[0] + c[1] * punto[1]) <= v for c, v in zip(coeficientes, valores)):
                    vertices.append(punto)
    
    # Intersecciones entre restricciones
    for i in range(len(coeficientes)):
        for j in range(i + 1, len(coeficientes)):
            coef1, val1 = coeficientes[i], valores[i]
            coef2, val2 = coeficientes[j], valores[j]
            
            # Calcular determinante para verificar si las líneas no son paralelas
            det = coef1[0] * coef2[1] - coef1[1] * coef2[0]
            
            if abs(det) > 1e-10:  # No son paralelas
                # Resolver sistema de ecuaciones
                x = (coef2[1] * val1 - coef1[1] * val2) / det
                y = (coef1[0] * val2 - coef2[0] * val1) / det
                
                if x >= 0 and y >= 0:
                    punto = (x, y)
                    # Verificar si el punto satisface todas las restricciones
                    if all((c[0] * punto[0] + c[1] * punto[1]) <= v for c, v in zip(coeficientes, valores)):
                        vertices.append(punto)
    
    # Eliminar duplicados y ordenar para formar el polígono
    vertices = list(set([(round(v[0], 5), round(v[1], 5)) for v in vertices]))
    
    # Si tenemos vértices, sombrear la región factible
    if vertices:
        # Ordenar vértices para formar polígono
        def ordenar_vertices(verts):
            # Encontrar centroide
            cent = tuple(map(lambda x: sum(x) / len(x), zip(*verts)))
            # Ordenar por ángulo
            return sorted(verts, key=lambda v: np.arctan2(v[1] - cent[1], v[0] - cent[0]))
        
        try:
            vertices_ordenados = ordenar_vertices(vertices)
            poly = plt.Polygon(vertices_ordenados, alpha=0.3, color='blue')
            plt.gca().add_patch(poly)
            
            # Mostrar vértices
            for i, v in enumerate(vertices):
                plt.scatter(v[0], v[1], color='red', s=50)
                plt.text(v[0] + 0.1, v[1] + 0.1, f"V{i+1}({v[0]:.2f}, {v[1]:.2f})")
        except Exception as e:
            print(f"No se pudo dibujar la región factible como un polígono: {e}")
    
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, x_max)
    plt.ylim(0, x_max)
    plt.title('Región Factible Combinada')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linewidth=0.5)
    plt.legend()

def mostrar_restricciones(coeficientes, valores):
    """Muestra las restricciones en formato tabular"""
    num_vars = len(coeficientes[0])
    
    # Crear encabezados para las variables y RHS
    cols = [f'x{i+1}' for i in range(num_vars)]
    cols += ['RHS']
    
    # Crear datos para la tabla
    data = []
    for i, (coef, val) in enumerate(zip(coeficientes, valores)):
        row = coef + [val]
        data.append(row)
    
    # Crear etiquetas para las filas
    filas = [f'R{i+1}' for i in range(len(coeficientes))]
    
    # Crear DataFrame de pandas para mostrar la tabla
    df = pd.DataFrame(data, columns=cols, index=filas)
    
    print("\nRestricciones:")
    print(df.round(3))
    
    # Mostrar en formato ecuación
    print("\nEcuaciones de restricciones:")
    for i, (coef, val) in enumerate(zip(coeficientes, valores)):
        eq = " + ".join([f"{coef[j]}x{j+1}" for j in range(len(coef))])
        print(f"R{i+1}: {eq} ≤ {val}")

def main():
    """Función principal para ejecutar el graficador de restricciones"""
    print("\n📊 GRAFICADOR DE RESTRICCIONES DE PROGRAMACIÓN LINEAL 📊")
    print("======================================================")
    
    # Usuario puede definir el problema o usar un ejemplo predefinido
    opcion = input("\n¿Desea ingresar un nuevo problema (N) o usar un ejemplo predefinido (E)?: ").upper().strip()
    
    if opcion == 'N':
        num_variables, num_restricciones, coeficientes, valores = crear_problema()
    else:
        print("\nUsando ejemplo predefinido:")
        num_variables = 2
        num_restricciones = 3
        coeficientes = [
            [2, 1],
            [1, 3],
            [3, 2]
        ]
        valores = [10, 15, 18]
        
        # Mostrar el ejemplo
        for i in range(num_restricciones):
            eq = " + ".join([f"{coeficientes[i][j]}x{j+1}" for j in range(num_variables)])
            print(f"Restricción {i+1}: {eq} ≤ {valores[i]}")
    
    # Mostrar tabla de restricciones
    mostrar_restricciones(coeficientes, valores)
    
    # Verificar que estamos trabajando con un problema 2D
    if num_variables != 2:
        print("\n⚠️ Este graficador solo puede visualizar problemas con 2 variables de decisión.")
        return
    
    # Definir el rango máximo para la visualización
    # (basado en los valores de las restricciones y coeficientes)
    max_val = 0
    for i, (coef, val) in enumerate(zip(coeficientes, valores)):
        for j in range(num_variables):
            if coef[j] > 0:
                max_val = max(max_val, val / coef[j] * 1.5)
    
    # Establecer un mínimo razonable
    max_val = max(max_val, 10)
    
    # Crear figura para las gráficas
    fig = plt.figure(figsize=(15, 10))
    
    # Graficar cada restricción individualmente
    for i in range(num_restricciones):
        plt.subplot(2, (num_restricciones + 1) // 2, i + 1)
        visualizar_restriccion_individual(coeficientes[i], valores[i], i + 1, max_val)
    
    # Graficar la región factible combinada
    plt.subplot(2, (num_restricciones + 1) // 2, num_restricciones + 1)
    visualizar_region_factible(coeficientes, valores, max_val)
    
    plt.tight_layout()
    plt.savefig('restricciones.png')
    
    print("\n✅ Gráficos generados exitosamente!")
    print("\n¡Gracias por usar el graficador de restricciones! 👋")

if __name__ == "__main__":
    main()