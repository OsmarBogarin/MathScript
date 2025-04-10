import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DESCRIPTION = """
## Gráficas de Funciones Objetivo en Programación Lineal

Este script analiza y visualiza funciones objetivo para problemas de **programación lineal**.

### Funciones principales:
- Cálculo de pendiente y ordenada al origen para rectas de nivel
- Visualización gráfica de rectas de nivel para la función objetivo
- Análisis detallado de la dirección de crecimiento/decrecimiento
"""

def crear_problema():
    """Permite al usuario definir la función objetivo y valores Z a analizar"""
    print("\n===== CONFIGURACIÓN DE LA FUNCIÓN OBJETIVO =====")
    
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
    
    # Valores de Z para graficar
    num_valores_z = int(input("\nIngrese el número de valores Z a graficar: "))
    valores_z = []
    for i in range(num_valores_z):
        valor = float(input(f"\nValor Z{i+1}: "))
        valores_z.append(valor)
    
    return tipo, num_vars, func_obj, valores_z

def despejar_x2(Z, x1, coef_x1, coef_x2):
    """Despeja x2 de la ecuación Z = coef_x1*x1 + coef_x2*x2"""
    return (Z - coef_x1*x1) / coef_x2

def calcular_pendiente_ordenada(coef_x1, coef_x2, Z):
    """Calcula la pendiente y ordenada al origen de la recta Z = coef_x1*x1 + coef_x2*x2"""
    pendiente = -coef_x1 / coef_x2
    ordenada = Z / coef_x2
    return pendiente, ordenada

def explicar_calculo_pendiente_ordenada(coef_x1, coef_x2, Z):
    """Explica paso a paso cómo se calcula la pendiente y ordenada al origen"""
    print("\n" + "="*60)
    print(" 📊 ANÁLISIS DE LA RECTA DE NIVEL Z = {}".format(Z))
    print("="*60)
    
    print("\n1️⃣ TRANSFORMACIÓN A FORMA ESTÁNDAR:")
    print("\n• Partimos de la función objetivo Z = {}x₁ + {}x₂".format(coef_x1, coef_x2))
    
    print("\n2️⃣ DESPEJE DE LA VARIABLE x₂:")
    print("   Z = {}x₁ + {}x₂".format(coef_x1, coef_x2))
    print("   {}x₂ = Z - {}x₁".format(coef_x2, coef_x1))
    print("   x₂ = (Z - {}x₁) / {}".format(coef_x1, coef_x2))
    print("   x₂ = -{}x₁/{} + Z/{}".format(coef_x1, coef_x2, coef_x2))
    
    pendiente = -coef_x1 / coef_x2
    ordenada = Z / coef_x2
    print("   x₂ = {}x₁ + {}".format(round(pendiente, 4), round(ordenada, 4)))
    
    print("\n3️⃣ IDENTIFICACIÓN DE PARÁMETROS:")
    print("   • Pendiente (m) = -{}/{} = {}".format(coef_x1, coef_x2, round(pendiente, 4)))
    print("   • Ordenada al origen (b) = Z/{} = {}".format(coef_x2, round(ordenada, 4)))
    
    print("\n4️⃣ ANÁLISIS DE INTERSECCIONES CON LOS EJES:")
    print("   a) Intersección con el eje x₂ (cuando x₁ = 0):")
    print("      • Punto (0, {})".format(round(ordenada, 4)))
    
    x1_interseccion = Z / coef_x1 if coef_x1 != 0 else float('inf')
    if x1_interseccion != float('inf'):
        print("   b) Intersección con el eje x₁ (cuando x₂ = 0):")
        print("      • Punto ({}, 0)".format(round(x1_interseccion, 4)))
    else:
        print("   b) No hay intersección con el eje x₁ (recta paralela al eje).")
    
    print("\n5️⃣ ECUACIÓN DE LA RECTA EN DIFERENTES FORMAS:")
    print("   • Forma pendiente-ordenada:  x₂ = {}x₁ + {}".format(round(pendiente, 4), round(ordenada, 4)))
    print("   • Forma general:  {}x₁ + {}x₂ = {}".format(coef_x1, coef_x2, Z))
    
    return pendiente, ordenada

def calcular_distancia_origen(coef_x1, coef_x2, Z):
    """Calcula la distancia perpendicular desde el origen a la recta Z = coef_x1*x1 + coef_x2*x2"""
    return abs(Z) / np.sqrt(coef_x1**2 + coef_x2**2)

def comparar_rectas_nivel(coef_x1, coef_x2, valores_Z):
    """Compara las diferentes rectas de nivel de la función objetivo"""
    print("\n" + "="*60)
    print(" 🔍 COMPARACIÓN ENTRE RECTAS DE NIVEL")
    print("="*60)
    
    # Crear una tabla para mostrar los parámetros
    datos = {
        'Valor Z': valores_Z,
        'Pendiente': [-coef_x1/coef_x2] * len(valores_Z),
        'Ordenada': [Z/coef_x2 for Z in valores_Z],
        'Distancia al origen': [calcular_distancia_origen(coef_x1, coef_x2, Z) for Z in valores_Z]
    }
    
    df = pd.DataFrame(datos)
    print("\nTabla de parámetros para las rectas de nivel:")
    print(df.round(4))
    
    pendiente = -coef_x1 / coef_x2
    
    print("\n1️⃣ ANÁLISIS DE PENDIENTE:")
    print(f"   • Todas las rectas tienen la misma pendiente: m = {round(pendiente, 4)}")
    print("   • Esto significa que todas las rectas son paralelas entre sí")
    
    print("\n2️⃣ ANÁLISIS DE ORDENADAS AL ORIGEN:")
    for Z in valores_Z:
        ordenada = Z / coef_x2
        print(f"   • Para Z = {Z}: ordenada = {round(ordenada, 4)}")
    
    print("\n3️⃣ ANÁLISIS DE DISTANCIAS AL ORIGEN:")
    for Z in valores_Z:
        distancia = calcular_distancia_origen(coef_x1, coef_x2, Z)
        print(f"   • Para Z = {Z}: distancia = {round(distancia, 4)}")
    
    print("\n4️⃣ INTERPRETACIÓN GEOMÉTRICA:")
    if all(Z >= 0 for Z in valores_Z):
        Z_min = min(valores_Z)
        Z_max = max(valores_Z)
        print(f"   • Las rectas se alejan del origen a medida que Z aumenta de {Z_min} a {Z_max}")
    
    print("\n5️⃣ VECTOR DE GRADIENTE:")
    print(f"   • El gradiente de la función Z = {coef_x1}x₁ + {coef_x2}x₂ es ({coef_x1}, {coef_x2})")
    print("   • Este vector indica la dirección de máximo crecimiento de la función")

def graficar_funcion_objetivo(tipo, coef_x1, coef_x2, valores_Z):
    """Grafica las rectas correspondientes a los diferentes valores de Z"""
    print("\n" + "="*60)
    print(" 📈 VISUALIZACIÓN GRÁFICA DE LA FUNCIÓN OBJETIVO")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Límites para el gráfico (asumiendo variables no negativas)
    max_z = max(valores_Z)
    max_x1 = max(15, max_z / coef_x1 * 1.5) if coef_x1 > 0 else 15
    max_x2 = max(15, max_z / coef_x2 * 1.5) if coef_x2 > 0 else 15
    
    # Crea un rango de valores para x1
    x1_vals = np.linspace(0, max_x1, 100)
    
    # Colores para las diferentes líneas
    colores = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    # Vector gradiente para mostrar la dirección de crecimiento
    vector_escala = min(max_x1, max_x2) * 0.2
    vector_x = coef_x1 * vector_escala / np.sqrt(coef_x1**2 + coef_x2**2)
    vector_y = coef_x2 * vector_escala / np.sqrt(coef_x1**2 + coef_x2**2)
    
    # Para cada valor de Z, grafica la recta correspondiente
    for i, Z in enumerate(valores_Z):
        color = colores[i % len(colores)]
        pendiente, ordenada = calcular_pendiente_ordenada(coef_x1, coef_x2, Z)
        
        # Verifica si x2 puede ser despejada
        if coef_x2 != 0:
            # Calcula los valores de x2 para cada x1
            x2_vals = [despejar_x2(Z, x1, coef_x1, coef_x2) for x1 in x1_vals]
            
            # Grafica la recta
            ax.plot(x1_vals, x2_vals, color=color, linewidth=2, 
                   label=f'Z = {Z}: x₂ = {pendiente:.3f}x₁ + {ordenada:.3f}')
            
            # Marca la intersección con el eje x2 (cuando x1 = 0)
            ax.scatter([0], [ordenada], color=color, s=50)
            ax.text(0.1, ordenada, f'(0, {ordenada:.2f})', fontsize=9, va='bottom')
            
            # Marca la intersección con el eje x1 (cuando x2 = 0)
            if coef_x1 != 0:
                x1_interseccion = Z / coef_x1
                if 0 <= x1_interseccion <= max_x1:
                    ax.scatter([x1_interseccion], [0], color=color, s=50)
                    ax.text(x1_interseccion, 0.1, f'({x1_interseccion:.2f}, 0)', fontsize=9, ha='center', va='bottom')
        else:
            # Si x2 no puede ser despejada (coef_x2 = 0), dibuja una línea vertical
            x1_val = Z / coef_x1
            ax.axvline(x=x1_val, color=color, linewidth=2, 
                      label=f'Z = {Z}: x₁ = {x1_val:.3f}')
    
    # Dibujar el vector gradiente
    if tipo.lower() == "max":
        ax.arrow(0, 0, vector_x, vector_y, head_width=vector_escala*0.1, 
                head_length=vector_escala*0.2, fc='purple', ec='purple', linewidth=2)
        ax.text(vector_x*0.7, vector_y*0.7, f'∇Z = ({coef_x1}, {coef_x2})', 
               color='purple', fontsize=10, ha='center')
    else:
        ax.arrow(0, 0, -vector_x, -vector_y, head_width=vector_escala*0.1, 
                head_length=vector_escala*0.2, fc='purple', ec='purple', linewidth=2)
        ax.text(-vector_x*0.7, -vector_y*0.7, f'-∇Z = ({-coef_x1}, {-coef_x2})', 
               color='purple', fontsize=10, ha='center')
    
    # Añade los ejes
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    # Configura el gráfico
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_xlim(0, max_x1)
    ax.set_ylim(0, max_x2)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Título dinámico basado en los coeficientes
    if tipo.lower() == "max":
        titulo = f'Rectas de nivel para {tipo.upper()}IMIZAR Z = {coef_x1}$x_1$ + {coef_x2}$x_2$'
    else:
        titulo = f'Rectas de nivel para {tipo.upper()}IMIZAR Z = {coef_x1}$x_1$ + {coef_x2}$x_2$'
    ax.set_title(titulo, fontsize=14)
    
    # Añade la leyenda
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('funcion_objetivo.png')
    plt.show()
    
    return fig, ax

def graficar_funcion_objetivo_3d(tipo, coefs, valores_Z):
    """Grafica las rectas de nivel para funciones objetivo de 3 variables mediante proyecciones 2D"""
    print("\n" + "="*60)
    print(" 📊 VISUALIZACIÓN DE FUNCIÓN OBJETIVO DE 3 VARIABLES")
    print("="*60)
    
    print("\n⚠️ La visualización 3D se muestra mediante proyecciones 2D de los planos.")
    
    # Crear figuras para las tres proyecciones principales
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Límites para los gráficos
    max_vals = [15, 15, 15]
    
    # Crear las proyecciones: xy, xz, yz
    projections = [(0, 1), (0, 2), (1, 2)]
    titles = ['Proyección x₁-x₂', 'Proyección x₁-x₃', 'Proyección x₂-x₃']
    
    for idx, (ax, proj, title) in enumerate(zip(axs, projections, titles)):
        i, j = proj
        x_label = f'x{i+1}'
        y_label = f'x{j+1}'
        
        # Crear ejes
        x = np.linspace(0, max_vals[i], 100)
        
        # Para cada valor de Z
        for k, Z in enumerate(valores_Z):
            color = ['blue', 'green', 'red', 'purple', 'orange'][k % 5]
            
            # Extraer los coeficientes relevantes para esta proyección
            coef_i = coefs[i]
            coef_j = coefs[j]
            
            # Si podemos despejar la variable j
            if coef_j != 0:
                # Calcular los valores asumiendo que la tercera variable es 0
                k_index = 3 - i - j  # Índice de la variable ausente en esta proyección
                coef_k = coefs[k_index]
                
                # Ecuación del plano proyectada a esta vista (asumiendo x_k = 0)
                y_vals = [(Z - coef_i * x_val) / coef_j for x_val in x]
                
                # Graficar la línea
                ax.plot(x, y_vals, color=color, linewidth=2, 
                       label=f'Z = {Z} (x{k_index+1}=0)')
            
        # Configurar el gráfico
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlim(0, max_vals[i])
        ax.set_ylim(0, max_vals[j])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('funcion_objetivo_3d.png')
    plt.show()
    
    return fig

def analizar_funcion_objetivo(tipo, coefs, valores_Z):
    """Analiza la función objetivo para diferentes valores de Z"""
    print("\n" + "="*60)
    print(f" 🎯 ANÁLISIS DE FUNCIÓN OBJETIVO: {tipo.upper()}IMIZAR Z = ", end="")
    print(" + ".join([f"{c}x{i+1}" for i, c in enumerate(coefs)]))
    print("="*60)
    
    print("\n💡 INTERPRETACIÓN DE LA FUNCIÓN OBJETIVO:")
    
    if tipo.lower() == "max":
        print(f"• Buscamos MAXIMIZAR Z = {' + '.join([f'{c}x{i+1}' for i, c in enumerate(coefs)])}")
        print(f"• El vector gradiente ∇Z = ({', '.join([str(c) for c in coefs])}) indica la dirección de máximo crecimiento")
    else:
        print(f"• Buscamos MINIMIZAR Z = {' + '.join([f'{c}x{i+1}' for i, c in enumerate(coefs)])}")
        print(f"• El vector -∇Z = ({', '.join([str(-c) for c in coefs])}) indica la dirección de máximo decrecimiento")
    
    if len(coefs) == 2:
        # Análisis para función de 2 variables
        coef_x1, coef_x2 = coefs
        
        # Calcular pendiente y ordenada para un valor ejemplo de Z
        pendiente, ordenada = calcular_pendiente_ordenada(coef_x1, coef_x2, valores_Z[0])
        
        print("\n📏 GEOMETRÍA DE LAS RECTAS DE NIVEL:")
        print(f"• Todas las rectas tienen pendiente m = {pendiente:.4f}")
        print(f"• Las rectas son paralelas y su distancia al origen aumenta con |Z|")
        
        # Tabla de características para cada valor de Z
        print("\n📊 TABLA DE CARACTERÍSTICAS:")
        data = {
            'Valor Z': valores_Z,
            'Pendiente': [pendiente] * len(valores_Z),
            'Ordenada': [Z/coef_x2 for Z in valores_Z],
            'Distancia al origen': [calcular_distancia_origen(coef_x1, coef_x2, Z) for Z in valores_Z]
        }
        df = pd.DataFrame(data)
        print(df.round(4))
        
        # Explicaciones detalladas para cada valor de Z
        for Z in valores_Z:
            explicar_calculo_pendiente_ordenada(coef_x1, coef_x2, Z)
        
        # Comparar las rectas
        comparar_rectas_nivel(coef_x1, coef_x2, valores_Z)
        
        # Visualizar la función objetivo
        graficar_funcion_objetivo(tipo, coef_x1, coef_x2, valores_Z)
    
    elif len(coefs) == 3:
        # Análisis para función de 3 variables
        print("\n📏 GEOMETRÍA DE LOS PLANOS DE NIVEL:")
        print(f"• Z = {' + '.join([f'{c}x{i+1}' for i, c in enumerate(coefs)])}")
        print("• Cada valor de Z define un plano en el espacio 3D")
        print(f"• El vector normal a todos los planos es ({', '.join([str(c) for c in coefs])})")
        
        # Tabla de características para cada valor de Z
        print("\n📊 TABLA DE CARACTERÍSTICAS:")
        data = {
            'Valor Z': valores_Z,
            'Vector normal': [f"({', '.join([str(c) for c in coefs])})" for _ in valores_Z],
            'Distancia al origen': [abs(Z)/np.sqrt(sum([c**2 for c in coefs])) for Z in valores_Z]
        }
        df = pd.DataFrame(data)
        print(df)
        
        # Visualización 3D
        graficar_funcion_objetivo_3d(tipo, coefs, valores_Z)

def main():
    print("\n" + "="*60)
    print(" 📈 VISUALIZADOR DE FUNCIONES OBJETIVO EN PROGRAMACIÓN LINEAL")
    print("="*60)
    print(SCRIPT_DESCRIPTION)
    
    # Solicitar la información al usuario o usar parámetros predeterminados
    use_default = input("\n¿Desea usar el ejemplo predeterminado (Z = 2x₁ + 3x₂, Z = 6, 12, 18)? (s/n): ").lower() == 's'
    
    if use_default:
        tipo = "max"
        num_vars = 2
        coefs = [2, 3]
        valores_Z = [6, 12, 18]
    else:
        # Solicitar información al usuario
        tipo, num_vars, coefs, valores_Z = crear_problema()
    
    # Mostrar resumen del problema
    print("\n" + "-"*60)
    print(" 📋 RESUMEN DEL PROBLEMA")
    print("-"*60)
    print(f"• Tipo de optimización: {tipo.upper()}IMIZAR")
    print(f"• Función objetivo: Z = {' + '.join([f'{c}x{i+1}' for i, c in enumerate(coefs)])}")
    print(f"• Valores de Z a analizar: {', '.join([str(z) for z in valores_Z])}")
    
    # Analizar y visualizar la función objetivo
    analizar_funcion_objetivo(tipo, coefs, valores_Z)
    
    print("\n¡Análisis completado! 👍")

if __name__ == "__main__":
    main()