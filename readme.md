# MathScript

Una aplicación web para ejecutar scripts matemáticos desde dispositivos móviles o de escritorio.

Este proyecto tiene como objetivo simplificar la ejecución de cálculos complejos y promover un entorno limpio para gestionar entradas y salidas en proyectos educativos y técnicos.

## Características

- Interfaz web responsive adaptada para dispositivos móviles
- Sistema modular para ejecutar diferentes scripts de Python
- Visualización de gráficos generados por matplotlib
- Estructura de carpetas para organizar scripts por categorías

## Requisitos

- Python 3.7 o superior
- Flask
- Matplotlib
- NumPy
- Pandas (para algunos scripts específicos)

## Instalación

1. Clone este repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd mathscript-webapp
   ```

2. Instale las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Cree la estructura de carpetas para los módulos:
   ```bash
   mkdir -p modules/simplex
   ```

4. Coloque sus scripts Python en los directorios correspondientes:
   ```bash
   cp ruta/a/simplex_v3.py modules/simplex/
   ```

## Estructura de directorios

```
mathscript-webapp/
├── app.py                  # Aplicación Flask principal
├── requirements.txt        # Dependencias del proyecto
├── static/                 # Archivos estáticos
│   └── css/
│       └── style.css       # Estilos CSS personalizados
├── templates/              # Plantillas HTML
│   ├── index.html          # Página principal
│   ├── modules.html        # Lista de scripts en un módulo
│   └── execute.html        # Página para ejecutar un script
└── modules/                # Directorio de módulos
    ├── simplex/            # Ejemplo de módulo
    │   └── simplex_v3.py   # Script de simplex
    └── otro_modulo/        # Otro módulo de ejemplo
        └── script_ejemplo.py
```

## Ejecución

1. Inicie la aplicación:
   ```bash
   python app.py
   ```

2. Abra su navegador en `http://localhost:5000`

3. Seleccione un módulo, luego un script, y finalmente ejecútelo.

## Cómo añadir nuevos scripts

Para añadir nuevos scripts a la aplicación:

1. Cree un nuevo directorio en la carpeta `modules` si desea una nueva categoría, o use una existente.
2. Copie sus scripts Python al directorio creado.
3. Los scripts deben incluir una función `main()` que se ejecutará cuando se seleccione el script.
4. Si usa matplotlib para generar gráficos, estos se mostrarán automáticamente en la interfaz web.

## Posibles problemas y soluciones

### Problema: Los gráficos no se muestran
- Asegúrese de que está usando `plt.savefig()` en sus scripts para guardar las figuras.
- Verifique que matplotlib esté correctamente instalado.

### Problema: Errores al ejecutar scripts
- Verifique las dependencias específicas que requiere su script.
- Los scripts que usan interfaz gráfica directa (como `plt.show()`) pueden necesitar modificaciones.

## Limitaciones actuales

- No admite entrada interactiva compleja (como múltiples líneas)
- No soporta widgets interactivos de matplotlib
- Los scripts deben terminar en un tiempo razonable
