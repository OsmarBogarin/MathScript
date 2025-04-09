import matplotlib.pyplot as plt
import numpy as np

def despejar_x2(Z, x1, coef_x1, coef_x2):
    """Despeja x2 de la ecuación Z = coef_x1*x1 + coef_x2*x2"""
    return (Z - coef_x1*x1) / coef_x2

def calcular_pendiente_ordenada(coef_x1, coef_x2):
    """Calcula la pendiente y ordenada al origen de la recta Z = coef_x1*x1 + coef_x2*x2"""
    pendiente = -coef_x1 / coef_x2
    return pendiente

def explicar_calculo_pendiente_ordenada(coef_x1, coef_x2, Z):
    """Explica paso a paso cómo se calcula la pendiente y ordenada al origen"""
    print("\n" + "-"*50)
    print(" CÁLCULO PASO A PASO DE LA FORMA PENDIENTE-ORDENADA")
    print("-"*50)
    
    print("\nPaso 1: Partimos de la función objetivo Z = {}x₁ + {}x₂".format(coef_x1, coef_x2))
    
    print("\nPaso 2: Despejamos x₂ para obtener la forma pendiente-ordenada (x₂ = mx₁ + b)")
    print("   Z = {}x₁ + {}x₂".format(coef_x1, coef_x2))
    print("   {}x₂ = Z - {}x₁".format(coef_x2, coef_x1))
    print("   x₂ = (Z - {}x₁) / {}".format(coef_x1, coef_x2))
    print("   x₂ = -{}x₁/{} + Z/{}".format(coef_x1, coef_x2, coef_x2))
    
    pendiente = -coef_x1 / coef_x2
    print("   x₂ = {}x₁ + Z/{}".format(round(pendiente, 4), coef_x2))
    
    print("\nPaso 3: Identificamos la pendiente y la ordenada al origen")
    print("   • Pendiente (m) = -{}/{} = {}".format(coef_x1, coef_x2, round(pendiente, 4)))
    print("   • Ordenada al origen (b) = Z/{}".format(coef_x2))
    
    print("\nPaso 4: Sustituimos los diferentes valores de Z para obtener cada recta")
    ordenada = Z / coef_x2
    print("   • Para Z = {}: x₂ = {}x₁ + {}/{}".format(Z, round(pendiente, 4), Z, coef_x2))
    print("   • Para Z = {}: x₂ = {}x₁ + {}".format(Z, round(pendiente, 4), round(ordenada, 4)))
    
    print("\nPaso 5: Análisis de las intersecciones con los ejes")
    print("   a) Intersección con el eje x₂ (cuando x₁ = 0):")
    print("      • Sustituimos x₁ = 0 en la ecuación: x₂ = {}(0) + {}".format(round(pendiente, 4), round(ordenada, 4)))
    print("      • Intersección en el punto (0, {})".format(round(ordenada, 4)))
    
    print("   b) Intersección con el eje x₁ (cuando x₂ = 0):")
    print("      • Igualamos a cero: 0 = {}x₁ + {}".format(round(pendiente, 4), round(ordenada, 4)))
    print("      • {}x₁ = -{}".format(round(pendiente, 4), round(ordenada, 4)))
    x1_interseccion = -ordenada / pendiente
    print("      • x₁ = {}/({}) = {}".format(-round(ordenada, 4), round(pendiente, 4), round(x1_interseccion, 4)))
    print("      • Intersección en el punto ({}, 0)".format(round(x1_interseccion, 4)))
    
    return pendiente, ordenada

def graficar_funcion_objetivo(coef_x1, coef_x2, valores_Z, titulo=None):
    """Grafica las rectas correspondientes a los diferentes valores de Z"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Límites para el gráfico
    x1_min, x1_max = 0, max(12, max(valores_Z) / coef_x1 * 1.2) if coef_x1 > 0 else 12
    x2_min, x2_max = 0, max(12, max(valores_Z) / coef_x2 * 1.2) if coef_x2 > 0 else 12
    
    # Crea un rango de valores para x1
    x1_vals = np.linspace(x1_min, x1_max, 100)
    
    # Colores para las diferentes líneas
    colores = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    # Pendiente común para todas las rectas
    pendiente = calcular_pendiente_ordenada(coef_x1, coef_x2)
    
    # Para cada valor de Z, grafica la recta correspondiente
    for i, Z in enumerate(valores_Z):
        color = colores[i % len(colores)]
        
        # Ordenada al origen para este valor de Z
        ordenada = Z / coef_x2
        
        # Calcula los valores de x2 para cada x1
        x2_vals = [despejar_x2(Z, x1, coef_x1, coef_x2) for x1 in x1_vals]
        
        # Grafica la recta
        ax.plot(x1_vals, x2_vals, color=color, label=f'Z = {Z}: $x_2 = {pendiente:.3f}x_1 + {ordenada:.3f}$')
        
        # Marca la intersección con el eje x2 (cuando x1 = 0)
        ax.scatter([0], [ordenada], color=color, s=50)
        ax.text(0.1, ordenada, f'(0, {ordenada:.2f})', fontsize=9, va='bottom')
        
        # Marca la intersección con el eje x1 (cuando x2 = 0)
        if coef_x1 != 0:
            x1_interseccion = Z / coef_x1
            if x1_min <= x1_interseccion <= x1_max:
                ax.scatter([x1_interseccion], [0], color=color, s=50)
                ax.text(x1_interseccion, 0.1, f'({x1_interseccion:.2f}, 0)', fontsize=9, ha='center', va='bottom')
    
    # Añade los ejes
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    # Configura el gráfico
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Título dinámico basado en los coeficientes
    if titulo is None:
        titulo = f'Rectas de nivel para la función objetivo Z = {coef_x1}$x_1$ + {coef_x2}$x_2$'
    ax.set_title(titulo, fontsize=14)
    
    # Añade la leyenda
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig, ax

def explicar_comparacion_rectas(coef_x1, coef_x2, valores_Z):
    """Explica detalladamente la comparación entre las diferentes rectas"""
    print("\n" + "-"*50)
    print(" COMPARACIÓN DETALLADA ENTRE LAS RECTAS")
    print("-"*50)
    
    print("\nPaso 1: Análisis de la pendiente común")
    pendiente = -coef_x1 / coef_x2
    print("   • Todas las rectas tienen la misma pendiente: m = -{}/{} = {}".format(coef_x1, coef_x2, round(pendiente, 4)))
    print("   • Esto significa que todas las rectas son paralelas entre sí")
    
    print("\nPaso 2: Cálculo de las ordenadas al origen para cada valor de Z")
    print("   • Fórmula para la ordenada al origen: b = Z/{}".format(coef_x2))
    for Z in valores_Z:
        ordenada = Z / coef_x2
        print("   • Para Z = {}: b = {}/{} = {}".format(Z, Z, coef_x2, round(ordenada, 4)))
    
    print("\nPaso 3: Cálculo de las intersecciones con el eje x₁")
    print("   • Fórmula para la intersección con el eje x₁: x₁ = Z/{}".format(coef_x1))
    for Z in valores_Z:
        x1_interseccion = Z / coef_x1
        print("   • Para Z = {}: x₁ = {}/{} = {}".format(Z, Z, coef_x1, round(x1_interseccion, 4)))
    
    print("\nPaso 4: Comparación de las distancias al origen")
    print("   • La distancia perpendicular de una recta al origen se calcula con la fórmula:")
    print("     d = |c|/√(a² + b²), donde ax + by + c = 0 es la ecuación general de la recta")
    
    # Convertir a forma general ax + by + c = 0
    # Desde x₂ = mx₁ + b, tenemos: -mx₁ + x₂ - b = 0
    a = -pendiente
    b = 1
    denominador = np.sqrt(a**2 + b**2)
    
    print("   • Para nuestra ecuación x₂ = {}x₁ + b, la forma general es: {}x₁ + x₂ - b = 0".format(round(pendiente, 4), round(a, 4)))
    print("   • Donde a = {}, b = {}, y c = -b = -Z/{}".format(round(a, 4), b, coef_x2))
    print("   • El denominador común es √({}² + {}²) = {}".format(round(a, 4), b, round(denominador, 4)))
    
    for Z in valores_Z:
        ordenada = Z / coef_x2
        c = -ordenada
        distancia = abs(c) / denominador
        print("   • Para Z = {}: c = -{} = {}, distancia = |{}|/{} = {}".format(
            Z, round(ordenada, 4), round(c, 4), round(c, 4), round(denominador, 4), round(distancia, 4)))
    
    print("\nPaso 5: Conclusiones")
    print("   • Las rectas son paralelas (misma pendiente: {})".format(round(pendiente, 4)))
    print("   • A medida que Z aumenta, las rectas se alejan más del origen")
    print("   • En la dirección de la función objetivo ({}, {}), los valores de Z aumentan".format(coef_x1, coef_x2))
    if coef_x1 > 0 and coef_x2 > 0:
        print("   • Como ambos coeficientes son positivos, al aumentar x₁ y x₂, Z también aumenta")
    
    return pendiente

def analizar_funcion_objetivo(coef_x1, coef_x2, valores_Z, tipo="max"):
    """Analiza las características de las rectas de nivel de la función objetivo"""
    print("\n" + "="*60)
    print(" ANÁLISIS DE LA FUNCIÓN OBJETIVO {0}IMIZAR Z = {1}x₁ + {2}x₂".format(tipo.upper(), coef_x1, coef_x2))
    print("="*60)
    
    # Análisis paso a paso
    pendiente, ordenada = explicar_calculo_pendiente_ordenada(coef_x1, coef_x2, valores_Z[0])
    
    # Comparación entre rectas
    explicar_comparacion_rectas(coef_x1, coef_x2, valores_Z)
    
    print("\nRESUMEN FINAL:")
    print(f"\na) Ecuaciones de las rectas en forma pendiente-ordenada:")
    print(f"   La forma general es: x₂ = {pendiente:.3f}x₁ + (Z/{coef_x2})")
    
    for Z in valores_Z:
        ordenada = Z / coef_x2
        print(f"   Para Z = {Z}: x₂ = {pendiente:.3f}x₁ + {ordenada:.3f}")
    
    print(f"\nb) Análisis de pendiente e intersecciones:")
    print(f"   • La pendiente de todas las rectas es: {pendiente:.3f}")
    print(f"   • Esta pendiente es constante para todas las rectas de nivel (son paralelas)")
    
    print(f"\nc) Observaciones sobre la función objetivo:")
    if tipo.lower() == "max":
        print(f"   • Al maximizar, buscamos el punto más alejado en la dirección ({coef_x1}, {coef_x2})")
    else:
        print(f"   • Al minimizar, buscamos el punto más alejado en la dirección ({-coef_x1}, {-coef_x2})")
    
    return pendiente

def main():
    print("\n" + "="*50)
    print(" VISUALIZADOR DE FUNCIONES OBJETIVO EN PROGRAMACIÓN LINEAL")
    print("="*50)
    
    # Solicitar la información al usuario o usar parámetros predeterminados
    use_default = input("\n¿Desea usar el ejemplo del ejercicio (Z = 2x₁ + 3x₂, Z = 6, 12, 18)? (s/n): ").lower() == 's'
    
    if use_default:
        tipo = "max"
        coef_x1 = 2
        coef_x2 = 3
        valores_Z = [6, 12, 18]
    else:
        # Solicitar información al usuario
        tipo = input("Ingrese el tipo de función objetivo (max/min): ").strip().lower()
        coef_x1 = float(input("Ingrese el coeficiente de x₁: "))
        coef_x2 = float(input("Ingrese el coeficiente de x₂: "))
        
        valores_Z_input = input("Ingrese los valores de Z separados por comas: ")
        valores_Z = [float(z.strip()) for z in valores_Z_input.split(',')]
    
    # Analizar y graficar la función objetivo
    analizar_funcion_objetivo(coef_x1, coef_x2, valores_Z, tipo)
    
    # Graficar la función objetivo
    fig, ax = graficar_funcion_objetivo(coef_x1, coef_x2, valores_Z)
    plt.show()

if __name__ == "__main__":
    main()