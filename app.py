# Math Script - Gestor de Scripts Matemáticos y Simulaciones
# Copyright (c) 2025 Osmar
#
# Este software está licenciado bajo la Licencia MIT.
# Consulte el archivo LICENSE para más detalles.
import builtins
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import importlib.util
import sys
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Configurar matplotlib para modo no interactivo
import matplotlib.pyplot as plt
import traceback
import markdown

app = Flask(__name__)

# Directorio base donde se almacenan los módulos
BASE_MODULES_DIR = "modules"

def get_module_directories():
    """Obtiene la lista de directorios de módulos"""
    if not os.path.exists(BASE_MODULES_DIR):
        os.makedirs(BASE_MODULES_DIR)
    
    return [d for d in os.listdir(BASE_MODULES_DIR) 
            if os.path.isdir(os.path.join(BASE_MODULES_DIR, d))]

def get_scripts_in_directory(directory):
    """Obtiene la lista de scripts Python en un directorio específico"""
    full_path = os.path.join(BASE_MODULES_DIR, directory)
    if not os.path.exists(full_path):
        return []
    
    return [f for f in os.listdir(full_path) 
            if f.endswith('.py') and os.path.isfile(os.path.join(full_path, f))]

def load_module(directory, script_name):
    """Carga dinámicamente un módulo Python desde un archivo"""
    module_path = os.path.join(BASE_MODULES_DIR, directory, script_name)
    module_name = script_name[:-3]  # Quitar la extensión .py
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    
    # Agregar el directorio al path para que el módulo pueda importar sus dependencias
    module_dir = os.path.join(BASE_MODULES_DIR, directory)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    
    spec.loader.exec_module(module)
    return module

# Clase para capturar la entrada/salida
class IOCapture:
    def __init__(self):
        self.inputs = []
        self.output = []
        self.current_input_index = 0
    
    def input_function(self, prompt):
        self.output.append(prompt)  # Siempre mostrar el prompt
        if self.current_input_index < len(self.inputs):
            value = self.inputs[self.current_input_index]
            self.current_input_index += 1
            self.output.append(f"{value}")
            return value 
        else:
            # Simular un error de entrada cuando no hay más entradas
            raise Exception("No hay más entradas disponibles")

    def print_function(self, *args, **kwargs):
        # Convertir argumentos a cadena y unirlos
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        text = sep.join(str(arg) for arg in args) + end
        self.output.append(text)
    
    def get_output(self):
        return ''.join(self.output)

@app.route('/')
def index():
    module_dirs = get_module_directories()
    return render_template('index.html', module_dirs=module_dirs)

@app.route('/modules/<directory>')
def list_modules(directory):
    scripts = get_scripts_in_directory(directory)
    return render_template('modules.html', directory=directory, scripts=scripts)

@app.route('/execute/<directory>/<script>')
def execute_script_view(directory, script):
    try:
        module = load_module(directory, script)
        raw_description = getattr(module, 'SCRIPT_DESCRIPTION', 'Sin descripción disponible.')
        description = markdown.markdown(raw_description)  # Convertir a HTML
    except Exception:
        description = '<p>No se pudo cargar la descripción del script.</p>'
    
    return render_template('execute.html', directory=directory, script=script, description=description)

@app.route('/api/execute', methods=['POST'])
def execute_script_api():
    data = request.json
    directory = data.get('directory')
    script = data.get('script')
    inputs = data.get('inputs', [])
    
    if not directory or not script:
        return jsonify({'error': 'Directorio o script no especificado'}), 400
    
    # Capturar la salida y redirigir entrada/salida
    io_capture = IOCapture()
    io_capture.inputs = inputs
    
    # Guardar las funciones originales
    original_input = builtins.input
    original_print = builtins.print
    
    # Redirigir entrada/salida
    builtins.input = io_capture.input_function
    builtins.print = io_capture.print_function

    # Guardar figuras generadas
    original_savefig = plt.savefig
    figures = []
    
    def capture_figure(*args, **kwargs):
        buffer = io.BytesIO()
        
        # Eliminar 'format' si ya está presente en kwargs para evitar conflicto
        kwargs.pop('format', None)

        original_savefig(buffer, format='png', **kwargs)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        figures.append(f"data:image/png;base64,{image_base64}")
        plt.close()
    
    plt.savefig = capture_figure
    
    try:
        # Cargar y ejecutar el módulo
        module = load_module(directory, script)
        
        # Si el módulo tiene una función main o metodo_simplex, ejecutarla
        if hasattr(module, 'main'):
            module.main()
        elif hasattr(module, 'metodo_simplex'):  # Para el caso específico de simplex
            module.metodo_simplex()
        else:
            # Ejecutar el módulo directamente
            pass
        
        output = io_capture.get_output()
        needs_more_input = False
        
    except Exception as e:
        output = io_capture.get_output()
        error_trace = traceback.format_exc()
        if "No hay más entradas disponibles" in str(e):
            needs_more_input = True
        else:
            output += f"\n\nError: {str(e)}\n{error_trace}"
            needs_more_input = False
    
    finally:
        # Restaurar funciones originales
        builtins.input = original_input
        builtins.print = original_print
        plt.savefig = original_savefig
       

    return jsonify({
        'output': output,
        'figures': figures,
        'needsMoreInput': needs_more_input
    })

if __name__ == '__main__':
    # Asegurarse de que existe el directorio base
    if not os.path.exists(BASE_MODULES_DIR):
        os.makedirs(BASE_MODULES_DIR)
    
    app.run(debug=True, host='0.0.0.0')
