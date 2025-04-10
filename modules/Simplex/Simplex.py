SCRIPT_DESCRIPTION = """
## Método Simplex

Este script resuelve problemas de **programación lineal** utilizando el _método simplex_.

### Funciones principales:
- Paso a paso del algoritmo simplex
- Visualización 2D y 3D del espacio de soluciones
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

def crear_problema():
    """Permite al usuario definir el problema de programación lineal"""
    print("\n===== CONFIGURACIÓN DEL PROBLEMA DE PROGRAMACIÓN LINEAL =====")
    
    # Elegir tipo de optimización
    tipo = input("\n¿Desea maximizar (max) o minimizar (min)?: ").lower().strip()
    while tipo not in ["max", "min"]:
        tipo = input("\nPor favor, ingrese 'max' para maximizar o 'min' para minimizar: ").lower().strip()
    
    # Número de variables de decisión
    num_vars = int(input("\nIngrese el número de variables de decisión (x1, x2, ...): "))
    
    # Función objetivo
    print("\nDefinición de la función objetivo:")
    func_obj = []
    for i in range(num_vars):
        coef = float(input(f"\nCoeficiente para x{i+1}: "))
        func_obj.append(coef)
    
    # Restricciones
    num_restricciones = int(input("\nIngrese el número de restricciones: "))
    restricciones = []
    lados_derechos = []
    
    print("\nDefinición de restricciones (todas deben ser de tipo ≤):")
    for i in range(num_restricciones):
        print(f"\nRestricción {i+1}:")
        coefs = []
        for j in range(num_vars):
            coef = float(input(f"\nCoeficiente para x{j+1}: "))
            coefs.append(coef)
        restricciones.append(coefs)
        rhs = float(input("\nLado derecho de la restricción: "))
        lados_derechos.append(rhs)
    
    return tipo, num_vars, num_restricciones, func_obj, restricciones, lados_derechos

def crear_tableau(tipo, num_vars, num_restricciones, func_obj, restricciones, lados_derechos):
    """Crea el tableau inicial basado en los datos proporcionados"""
    # Crear la matriz del tableau
    # Filas: restricciones + función objetivo
    # Columnas: variables originales + variables de holgura + lado derecho
    tableau = np.zeros((num_restricciones + 1, num_vars + num_restricciones + 1))
    
    # Llenar las restricciones con coeficientes y variables de holgura
    for i in range(num_restricciones):
        # Coeficientes de las variables originales
        tableau[i, :num_vars] = restricciones[i]
        # Variable de holgura (1 en su posición correspondiente)
        tableau[i, num_vars + i] = 1
        # Lado derecho de la restricción
        tableau[i, -1] = lados_derechos[i]
    
    # Fila de la función objetivo (Z)
    if tipo == "max":
        # Para maximización, los coeficientes se multiplican por -1
        tableau[-1, :num_vars] = [-x for x in func_obj]
    else:
        # Para minimización, primero convertimos a un problema de maximización
        tableau[-1, :num_vars] = func_obj
    
    return tableau

def mostrar_tabla(tableau, num_vars, num_restricciones, iteracion=None):
    """Muestra la tabla simplex actual en formato tabular"""
    # Crear encabezados para variables originales, de holgura y RHS
    cols = [f'x{i+1}' for i in range(num_vars)]
    cols += [f's{i+1}' for i in range(num_restricciones)]
    cols += ['RHS']
    
    # Crear etiquetas para las filas
    filas = [f'R{i+1}' for i in range(num_restricciones)]
    filas += ['Z']
    
    # Crear un DataFrame de pandas para mostrar la tabla
    df = pd.DataFrame(tableau, columns=cols, index=filas)
    
    # Mostrar título para la iteración actual
    if iteracion is not None:
        print(f"\n===== ITERACIÓN {iteracion} =====")
    
    print("\nTabla Simplex:")
    print(df.round(3))
    return df

def encontrar_variable_entrante(tableau, num_vars, tipo):
    """Encuentra la variable que debe entrar a la base"""
    z_row = tableau[-1, :-1]
    
    if tipo == "max":
        # Para maximización, buscamos el coeficiente más negativo
        if all(z_row >= 0):
            return None, True  # Solución óptima encontrada
        pivot_col = np.argmin(z_row)
    else:
        # Para minimización, buscamos el coeficiente más positivo
        if all(z_row <= 0):
            return None, True  # Solución óptima encontrada
        pivot_col = np.argmax(z_row)
    
    return pivot_col, False

def encontrar_variable_saliente(tableau, pivot_col):
    """Encuentra la variable que debe salir de la base"""
    ratios = []
    for i in range(len(tableau) - 1):  # Excluir la fila Z
        col_val = tableau[i, pivot_col]
        rhs_val = tableau[i, -1]
        
        if col_val > 0:
            ratios.append((rhs_val / col_val, i))
        else:
            ratios.append((float('inf'), i))
    
    # Ordenar por ratio y devolver la fila con el ratio mínimo
    ratios.sort()
    if ratios[0][0] == float('inf'):
        return None  # Problema no acotado
    
    return ratios[0][1]  # Devolver el índice de fila

def simplex_iteration(tableau, num_vars, num_restricciones, tipo):
    """Realiza una iteración del método simplex con explicaciones detalladas"""
    # Mostrar la tabla actual
    df = mostrar_tabla(tableau, num_vars, num_restricciones)
    
    # Encontrar la variable que entra
    pivot_col, optimo = encontrar_variable_entrante(tableau, num_vars, tipo)
    if optimo:
        print("\n✅ Solución óptima encontrada!")
        return tableau, True
    
    if pivot_col < num_vars:
        var_entrante = f"x{pivot_col+1}"
    else:
        var_entrante = f"s{pivot_col-num_vars+1}"
    
    print(f"\n1️⃣ Variable entrante: {var_entrante} (columna {pivot_col+1})")
    
    # Calcular ratios y encontrar variable saliente
    ratios = []
    for i in range(len(tableau) - 1):
        col_val = tableau[i, pivot_col]
        rhs_val = tableau[i, -1]
        
        if col_val > 0:
            ratio = rhs_val / col_val
            ratios.append(ratio)
            print(f"\n   Ratio para fila {i+1}: {rhs_val:.3f} / {col_val:.3f} = {ratio:.3f}")
        else:
            ratios.append(float('inf'))
            print(f"\n   Ratio para fila {i+1}: {rhs_val:.3f} / {col_val:.3f} = ∞ (no factible)")
    
    if all(r == float('inf') for r in ratios):
        print("\n⚠️ El problema no tiene solución acotada!")
        return tableau, True
    
    # Encontrar la fila pivote con el ratio mínimo
    pivot_row = np.argmin([r if r != float('inf') else float('inf') for r in ratios])
    pivot_val = tableau[pivot_row, pivot_col]
    
    if pivot_row < num_restricciones:
        var_saliente = f"s{pivot_row+1}"
    else:
        # Identificar qué variable básica está en la fila pivot
        var_saliente = "variable básica"
    
    print(f"\n2️⃣ Variable saliente: {var_saliente} (fila {pivot_row+1})")
    print(f"\n3️⃣ Elemento pivote: {pivot_val:.3f} [fila {pivot_row+1}, columna {pivot_col+1}]")
    
    # Normalizar la fila pivote
    print(f"\n4️⃣ Normalizando la fila pivote (dividiendo la fila {pivot_row+1} por {pivot_val:.3f}):")
    
    fila_original = tableau[pivot_row].copy()
    tableau[pivot_row] = tableau[pivot_row] / pivot_val
    
    print("\n   Fila original:", fila_original.round(3))
    print("   Fila normalizada:", tableau[pivot_row].round(3))
    
    # Hacer ceros en la columna pivote
    print(f"\n5️⃣ Haciendo ceros en la columna pivote:")
    
    for i in range(len(tableau)):
        if i != pivot_row:
            multiplier = tableau[i, pivot_col]
            if multiplier != 0:
                print(f"\n   Fila {i+1} = Fila {i+1} - ({multiplier:.3f} × Fila {pivot_row+1})")
                fila_original = tableau[i].copy()
                tableau[i] = tableau[i] - multiplier * tableau[pivot_row]
                print(f"     Original: {fila_original.round(3)}")
                print(f"     Resultado: {tableau[i].round(3)}")
    
    return tableau, False

def obtener_solucion(tableau, num_vars, num_restricciones):
    """Extrae y muestra la solución óptima del tableau final"""
    variables = [f'x{i+1}' for i in range(num_vars)]
    solution = {var: 0.0 for var in variables}
    
    # Para cada variable, verificar si es básica
    for col in range(num_vars):
        # Ver si la columna tiene exactamente un 1 y el resto ceros
        col_values = tableau[:-1, col]
        if np.sum(col_values == 1) == 1 and np.sum(col_values == 0) == len(col_values) - 1:
            # Encontrar la fila que tiene el 1
            row = np.where(col_values == 1)[0][0]
            solution[variables[col]] = tableau[row, -1]
    
    # Para un método más general que funcione incluso después de muchas iteraciones
    for i in range(len(tableau) - 1):  # Para cada restricción
        is_basic = False
        basic_var = None
        
        # Buscar si esta fila representa una variable básica
        for j in range(num_vars + num_restricciones):
            if j < num_vars and np.count_nonzero(tableau[:-1, j] == 1) == 1 and np.count_nonzero(tableau[:-1, j] == 0) == num_restricciones - 1:
                if tableau[i, j] == 1:
                    is_basic = True
                    basic_var = f'x{j+1}'
        
        if is_basic and basic_var:
            solution[basic_var] = tableau[i, -1]
    
    # Mostrar solución
    print("\n===== SOLUCIÓN ÓPTIMA =====")
    for var, val in solution.items():
        print(f"{var} = {val:.3f}")
    
    z_value = tableau[-1, -1]
    print(f"\nValor óptimo: Z = {z_value:.3f}")
    
    return solution, z_value

def visualizar_problema_2d(restricciones, lados_derechos, func_obj, tipo, solucion=None):
    """Visualiza gráficamente un problema de 2 variables"""
    plt.figure(figsize=(10, 8))
    
    # Definir los límites del gráfico (asumiendo que todas las variables son ≥ 0)
    max_x = 0
    max_y = 0
    
    # Encontrar límites razonables para el gráfico
    for i, restriccion in enumerate(restricciones):
        if restriccion[0] > 0:  # Si el coeficiente de x1 es positivo
            x_intercept = lados_derechos[i] / restriccion[0]
            max_x = max(max_x, x_intercept * 1.5)
        
        if restriccion[1] > 0:  # Si el coeficiente de x2 es positivo
            y_intercept = lados_derechos[i] / restriccion[1]
            max_y = max(max_y, y_intercept * 1.5)
    
    # Establecer límites mínimos si son demasiado pequeños
    max_x = max(max_x, 10)
    max_y = max(max_y, 10)
    
    x = np.linspace(0, max_x, 1000)
    
    # Graficar cada restricción
    for i, restriccion in enumerate(restricciones):
        if restriccion[1] == 0:  # Línea vertical x = constante
            plt.axvline(x=lados_derechos[i]/restriccion[0], 
                        label=f"\nRestricción {i+1}: {restriccion[0]}x₁ + {restriccion[1]}x₂ ≤ {lados_derechos[i]}")
        else:
            y = (lados_derechos[i] - restriccion[0] * x) / restriccion[1]
            plt.plot(x, y, label=f"\nRestricción {i+1}: {restriccion[0]}x₁ + {restriccion[1]}x₂ ≤ {lados_derechos[i]}")
    
    # Sombrear la región factible suavemente
    # Primero creamos una malla de puntos
    x_mesh, y_mesh = np.meshgrid(np.linspace(0, max_x, 500), np.linspace(0, max_y, 500))
    points = np.vstack((x_mesh.flatten(), y_mesh.flatten())).T
    
    # Evaluamos cada restricción para cada punto
    mask = np.ones(len(points), dtype=bool)
    for restriccion, rhs in zip(restricciones, lados_derechos):
        mask = mask & (np.dot(points, restriccion[:2]) <= rhs)
    
    # Convertimos la máscara de nuevo a la forma de la malla
    mask = mask.reshape(x_mesh.shape)
    
    # Sombreamos la región factible con un gradiente suave
    plt.contourf(
        x_mesh, y_mesh, mask.astype(float),
        levels=50, cmap='Blues', alpha=0.3
    )
    
    # Dibujar la función objetivo para diferentes valores de Z
    if tipo == "max":
        z_values = [0, func_obj[0] + func_obj[1], 2 * (func_obj[0] + func_obj[1])]
    else:
        z_values = [3 * (func_obj[0] + func_obj[1]), 2 * (func_obj[0] + func_obj[1]), func_obj[0] + func_obj[1]]
    
    for z in z_values:
        if func_obj[1] != 0:
            y_obj = (z - func_obj[0] * x) / func_obj[1]
            plt.plot(x, y_obj, 'g--', alpha=0.5)
            # Anotar el valor de Z en el centro de la línea
            mid_x = max_x / 2
            mid_y = (z - func_obj[0] * mid_x) / func_obj[1]
            if 0 <= mid_y <= max_y:
                plt.text(mid_x, mid_y, f"Z = {z}", color='green')
    
    # Marcar los vértices de la región factible
    vertices = []
    
    # Añadir el origen
    vertices.append((0, 0))
    
    # Intersecciones con los ejes
    for i, restriccion in enumerate(restricciones):
        if restriccion[0] > 0:
            vertices.append((lados_derechos[i] / restriccion[0], 0))
        if restriccion[1] > 0:
            vertices.append((0, lados_derechos[i] / restriccion[1]))
    
    # Intersecciones entre restricciones
    for i, r1 in enumerate(restricciones):
        for j, r2 in enumerate(restricciones):
            if i < j:  # Para evitar duplicados
                det = r1[0] * r2[1] - r1[1] * r2[0]
                if abs(det) > 1e-10:  # Si no son paralelas
                    x_int = (r2[1] * lados_derechos[i] - r1[1] * lados_derechos[j]) / det
                    y_int = (r1[0] * lados_derechos[j] - r2[0] * lados_derechos[i]) / det
                    if x_int >= 0 and y_int >= 0:  # Solo si están en el primer cuadrante
                        # Verificar si está en la región factible
                        is_feasible = True
                        for k, restriccion in enumerate(restricciones):
                            if np.dot([x_int, y_int], restriccion[:2]) > lados_derechos[k] + 1e-10:
                                is_feasible = False
                                break
                        if is_feasible:
                            vertices.append((x_int, y_int))
    
    # Eliminar duplicados y ordenar los vértices
    vertices = list(set([(round(v[0], 5), round(v[1], 5)) for v in vertices]))
    vertices = [(v[0], v[1]) for v in vertices if 0 <= v[0] <= max_x and 0 <= v[1] <= max_y]
    
    # Marcar los vértices en el gráfico
    for i, v in enumerate(vertices):
        plt.scatter(v[0], v[1], color='red', s=50)
        plt.text(v[0] + 0.1, v[1] + 0.1, f"V{i+1}({v[0]:.2f}, {v[1]:.2f})")
    
    # Si se proporciona la solución óptima, marcarla
    if solucion:
        opt_x = solucion.get('x1', 0)
        opt_y = solucion.get('x2', 0)
        plt.scatter(opt_x, opt_y, color='green', s=100, marker='*')
        plt.text(opt_x, opt_y + 0.5, f"Solución Óptima\n({opt_x:.2f}, {opt_y:.2f})", 
                 ha='center', va='bottom', color='green', fontweight='bold')
    
    # Configurar el gráfico
    plt.grid(True)
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Visualización Gráfica del Problema de Programación Lineal')
    plt.legend(loc='upper right')
    
    # Función objetivo en formato texto
    if tipo == "max":
        obj_text = f"Maximizar Z = {func_obj[0]}x₁ + {func_obj[1]}x₂"
    else:
        obj_text = f"Minimizar Z = {func_obj[0]}x₁ + {func_obj[1]}x₂"
    plt.text(max_x/2, max_y*0.95, obj_text, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('region_factible.png')
    plt.show()

def visualizar_problema_3d(restricciones, lados_derechos, func_obj, tipo, solucion=None):
    """Visualiza un problema de 3 variables mediante proyecciones 2D"""
    print("\n⚠️ La visualización 3D es aproximada y se muestra mediante proyecciones 2D.")
    
    # Crear figuras para las tres proyecciones principales
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Definir límites para los gráficos
    max_vals = [0, 0, 0]
    
    # Encontrar límites razonables
    for i, restriccion in enumerate(restricciones):
        for j in range(3):
            if restriccion[j] > 0:
                val = lados_derechos[i] / restriccion[j]
                max_vals[j] = max(max_vals[j], val * 1.5)
    
    # Establecer mínimos si son demasiado pequeños
    max_vals = [max(val, 10) for val in max_vals]
    
    # Crear las proyecciones: xy, xz, yz
    projections = [(0, 1), (0, 2), (1, 2)]
    titles = ['Proyección x₁-x₂', 'Proyección x₁-x₃', 'Proyección x₂-x₃']
    
    for idx, (ax, proj, title) in enumerate(zip(axs, projections, titles)):
        i, j = proj
        x_label = f'x{i+1}'
        y_label = f'x{j+1}'
        
        # Crear matriz para la región factible
        x = np.linspace(0, max_vals[i], 100)
        y = np.linspace(0, max_vals[j], 100)
        x_mesh, y_mesh = np.meshgrid(x, y)
        
        # Graficar cada restricción en la proyección
        for r_idx, restriccion in enumerate(restricciones):
            # Si la restricción involucra solo estas dos variables
            if all(restriccion[k] == 0 for k in range(3) if k != i and k != j):
                if restriccion[j] == 0:  # Línea vertical
                    ax.axvline(x=lados_derechos[r_idx]/restriccion[i], 
                              label=f"R{r_idx+1}")
                else:
                    y_vals = (lados_derechos[r_idx] - restriccion[i] * x) / restriccion[j]
                    ax.plot(x, y_vals, label=f"R{r_idx+1}")
            else:
                # Para restricciones que involucran la tercera variable,
                # asumimos que la tercera variable es 0 para esta proyección
                if restriccion[j] == 0:
                    continue
                y_vals = (lados_derechos[r_idx] - restriccion[i] * x) / restriccion[j]
                ax.plot(x, y_vals, '--', label=f"R{r_idx+1} (proyección)")
        
        # Marcar la solución óptima si existe
        if solucion:
            opt_x = solucion.get(f'x{i+1}', 0)
            opt_y = solucion.get(f'x{j+1}', 0)
            ax.scatter(opt_x, opt_y, color='green', s=100, marker='*')
            ax.text(opt_x, opt_y + max_vals[j]*0.05, f"Óptimo", 
                   ha='center', va='bottom', color='green', fontweight='bold')
        
        # Configuración del gráfico
        ax.grid(True)
        ax.set_xlim(0, max_vals[i])
        ax.set_ylim(0, max_vals[j])
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('proyecciones_3d.png')
    plt.show()
    
    # Intento de visualización en 3D (aunque generalmente es difícil de interpretar)
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Para cada par de restricciones, calculamos su intersección
        for (i, r1), (j, r2) in combinations(enumerate(restricciones), 2):
            # Esto es una simplificación y solo funciona para ciertos tipos de restricciones
            # Un enfoque completo requeriría más matemáticas de geometría 3D
            
            # Crear puntos en los ejes
            for axis in range(3):
                if r1[axis] > 0:
                    point = [0, 0, 0]
                    point[axis] = lados_derechos[i] / r1[axis]
                    ax.scatter(point[0], point[1], point[2], color='blue')
                
                if r2[axis] > 0:
                    point = [0, 0, 0]
                    point[axis] = lados_derechos[j] / r2[axis]
                    ax.scatter(point[0], point[1], point[2], color='blue')
        
        # Marcar la solución óptima
        if solucion:
            opt_x = solucion.get('x1', 0)
            opt_y = solucion.get('x2', 0)
            opt_z = solucion.get('x3', 0)
            ax.scatter([opt_x], [opt_y], [opt_z], color='green', s=100, marker='*')
            ax.text(opt_x, opt_y, opt_z, f"Óptimo", color='green')
        
        # Configurar el gráfico 3D
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_zlabel('x₃')
        ax.set_title('Visualización 3D aproximada')
        
        # Limitar los ejes
        ax.set_xlim(0, max_vals[0])
        ax.set_ylim(0, max_vals[1])
        ax.set_zlim(0, max_vals[2])
        
        plt.tight_layout()
        plt.savefig('visualizacion_3d.png')
        plt.show()
    except Exception as e:
        print(f"No se pudo generar la visualización 3D: {e}")

def metodo_simplex():
    """Función principal para ejecutar el método simplex"""
    # Obtener datos del problema
    print("\n🔢 MÉTODO SIMPLEX PARA PROGRAMACIÓN LINEAL 🔢")
    print("=============================================")
    
    # Usuario puede definir el problema o usar un ejemplo predefinido
    opcion = input("\n¿Desea ingresar un nuevo problema (N) o usar un ejemplo predefinido (E)?: ").upper().strip()
    
    if opcion == 'N':
        tipo, num_vars, num_restricciones, func_obj, restricciones, lados_derechos = crear_problema()
    else:
        print("\nUsando ejemplo predefinido:")
        tipo = "max"
        num_vars = 3
        num_restricciones = 3
        func_obj = [5, 9, 7]
        restricciones = [
            [1, 3, 2],
            [3, 4, 2],
            [2, 1, 2]
        ]
        lados_derechos = [10, 12, 8]
        
        print(f"Tipo: {'Maximización' if tipo == 'max' else 'Minimización'}")
        print(f"Función objetivo: Z = {' + '.join([f'{c}x{i+1}' for i, c in enumerate(func_obj)])}")
        for i in range(num_restricciones):
            print(f"Restricción {i+1}: {' + '.join([f'{c}x{j+1}' for j, c in enumerate(restricciones[i])])} ≤ {lados_derechos[i]}")
    
    # Crear tableau inicial
    tableau = crear_tableau(tipo, num_vars, num_restricciones, func_obj, restricciones, lados_derechos)
    
    # Mostrar tableau inicial
    print("\n===== TABLA INICIAL =====")
    mostrar_tabla(tableau, num_vars, num_restricciones)
    
    # Iterar hasta encontrar solución óptima
    optimal = False
    iteration = 1
    
    while not optimal:
        print(f"\n===== INICIANDO ITERACIÓN {iteration} =====")
        tableau, optimal = simplex_iteration(tableau, num_vars, num_restricciones, tipo)
        
        if optimal:
            break
        
        iteration += 1
        if iteration > 10:  # Límite de seguridad
            print("\n⚠️ Se alcanzó el límite máximo de iteraciones!")
            break
    
    # Obtener y mostrar la solución final
    solucion, valor_optimo = obtener_solucion(tableau, num_vars, num_restricciones)
    
    # Visual
    # Visualizar el problema gráficamente si es posible
    visualizar = input("\n¿Desea visualizar gráficamente el problema? (S/N): ").upper().strip() == 'S'
    
    if visualizar:
        if num_vars == 2:
            print("\n🔄 Generando visualización 2D del problema...")
            visualizar_problema_2d(restricciones, lados_derechos, func_obj, tipo, solucion)
        elif num_vars == 3:
            print("\n🔄 Generando visualización 3D del problema...")
            visualizar_problema_3d(restricciones, lados_derechos, func_obj, tipo, solucion)
        else:
            print("\n⚠️ La visualización gráfica solo está disponible para problemas con 2 o 3 variables.")
            
            # Para problemas con más variables, ofrecemos visualizar pares de variables
            if input("¿Desea visualizar proyecciones seleccionadas? (S/N): ").upper().strip() == 'S':
                print("\nSeleccione dos variables para visualizar:")
                var1 = int(input(f"Primera variable (1-{num_vars}): ")) - 1
                var2 = int(input(f"Segunda variable (1-{num_vars}): ")) - 1
                
                if 0 <= var1 < num_vars and 0 <= var2 < num_vars and var1 != var2:
                    # Crear proyección de las restricciones para estas dos variables
                    proyeccion_restricciones = []
                    proyeccion_lados_derechos = []
                    
                    for i, restriccion in enumerate(restricciones):
                        # Proyectar asumiendo otras variables en 0
                        nueva_restriccion = [0, 0]
                        nueva_restriccion[0] = restriccion[var1]
                        nueva_restriccion[1] = restriccion[var2]
                        proyeccion_restricciones.append(nueva_restriccion)
                        proyeccion_lados_derechos.append(lados_derechos[i])
                    
                    # Proyectar función objetivo
                    proyeccion_func_obj = [func_obj[var1], func_obj[var2]]
                    
                    # Crear una solución proyectada
                    solucion_proyectada = {
                        'x1': solucion.get(f'x{var1+1}', 0),
                        'x2': solucion.get(f'x{var2+1}', 0)
                    }
                    
                    print(f"\n🔄 Generando proyección para variables x{var1+1} y x{var2+1}...")
                    visualizar_problema_2d(proyeccion_restricciones, proyeccion_lados_derechos, 
                                          proyeccion_func_obj, tipo, solucion_proyectada)
                else:
                    print("Selección de variables inválida.")
    
    print("\n¡Gracias por usar el método simplex! 👋")

if __name__ == "__main__":
    metodo_simplex()