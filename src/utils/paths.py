from pathlib import Path
import os
from typing import Optional


# Funciones de utilidades para paths y URIs

def ensure_path(path_str: str | Path) -> Path:
    """
    Convierte un string o Path en objeto Path y crea los directorios necesarios.

    - Si la ruta es un directorio, lo crea directamente.
    - Si la ruta es un archivo, crea su carpeta contenedora.

    Args:
        path_str (str | Path): Ruta (archivo o carpeta) a convertir/crear.

    Returns:
        Path: Objeto Path garantizado (no crea el archivo, solo el directorio padre).
    """
    path = Path(path_str)

    # 2) Si es archivo, crear carpeta contenedora
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # 3) Si es directorio, crear directamente
        path.mkdir(parents=True, exist_ok=True)

    return path

def normalize_mlflow_uri(uri: Optional[str]) -> Optional[str]:
    """Normaliza la URI de MLflow.

    - Si es None devuelve None.
    - Si es del tipo `file:./mlruns` convierte a `file:///abs/path/to/mlruns`.
    - Si ya es una URL HTTP o `file://` la devuelve tal cual.

    Args:
        uri: cadena con la URI de tracking configurada.

    Returns:
        URI normalizada o None.
    """
    if not uri:
        return uri

    uri = str(uri)
    # manejar case: file:./mlruns (sin doble slash)
    if uri.startswith("file:") and not uri.startswith("file://"):
        path_part = uri[len("file:"):]
        abs_path = os.path.abspath(path_part)
        return f"file://{abs_path}"

    return uri


def build_model_registry_uri(model_name: str, version: Optional[str | int]) -> str:
    """Construye una URI para cargar modelos desde el MLflow Model Registry.

    Ejemplo: build_model_registry_uri('RFRegressor', '3') -> 'models:/RFRegressor/3'

    Args:
        model_name: nombre del modelo en el registry.
        version: número de versión (str o int). Si es None, lanza ValueError.

    Returns:
        URI tipo 'models:/<name>/<version>'.
    """
    if version is None:
        raise ValueError("version is required to build a model registry URI")

    return f"models:/{model_name}/{version}"

def build_model_local_path(model_name: str, version: Optional[str | int], model_file: str,) -> str:
    """Construye una ruta local para cargar modelos desde el filesystem.

    Ejemplo: build_model_local_path('RFRegressor', '3', 'model.pkl') -> 'models/RFRegressor/3/model.pkl'

    Args:
        model_name: nombre del modelo.
        version: número de versión (str o int). Si es None, lanza ValueError.
        model_file: nombre del archivo del modelo (ej. 'model.pkl').
        
    Returns:
        Ruta local tipo 'models/<name>/<version>/<model_file>'.
    """
    if version is None:
        raise ValueError("version is required to build a local model path")

    return f"models/{model_name}/{version}/{model_file}"


def get_next_version_path(model_dir_path: str | Path) -> Path:
    """
    Encuentra la ruta del subdirectorio de la versión más alta (última) más uno.
    Crea y devuelve la ruta para la nueva versión (ej. '4' si la última fue '3').

    Args:
        model_dir_path: Ruta base del modelo (ej. /ruta/a/modelos/mi_modelo).

    Returns:
        Path: Objeto Path del subdirectorio de la NUEVA versión, listo para usar.
    """
    base_path = Path(model_dir_path)
    
    # 1. Asegurar que el directorio base exista
    ensure_path(base_path)

    # 2. Buscar subdirectorios y encontrar la versión numérica máxima
    all_versions = [d for d in base_path.iterdir() if d.is_dir()]
    numeric_versions = []
    for d in all_versions:
        try:
            version_num = int(d.name)
            numeric_versions.append(version_num)
        except ValueError:
            # Ignorar subdirectorios que no son versiones numéricas
            continue

    # 3. Determinar el número de la siguiente versión
    if not numeric_versions:
        # Si no hay versiones, la próxima es la 1
        next_version_num = 1
        print(f"No se encontró ninguna versión. Creando versión inicial '{next_version_num}' en: {base_path}")
    else:
        # Si hay versiones, la próxima es la versión máxima + 1
        latest_version_num = max(numeric_versions)
        next_version_num = latest_version_num + 1
        print(f"Última versión encontrada: {latest_version_num}. Creando la siguiente: '{next_version_num}'.")

    # 4. Construir y crear el Path de la nueva versión
    new_version_path = base_path / str(next_version_num)
    
    # Usamos ensure_path para crear el nuevo directorio de la versión
    return ensure_path(new_version_path)

def get_latest_version_path(model_dir_path: str | Path) -> Path:
    """
    Encuentra la ruta del subdirectorio con la versión más alta (última)
    dentro de un directorio base. Si no existe ninguna versión numérica,
    crea y devuelve la ruta para la versión '1'.

    Args:
        model_dir_path: Ruta base del modelo (ej. /ruta/a/modelos/mi_modelo).

    Returns:
        Path: Objeto Path del subdirectorio de la versión más reciente (o '1' si se crea).
    """
    base_path = Path(model_dir_path)

    # 1. Asegurar que el directorio base exista
    ensure_path(base_path)

    # 2. Buscar subdirectorios y filtrar numéricos
    all_versions = [d for d in base_path.iterdir() if d.is_dir()]
    numeric_versions = []
    for d in all_versions:
        try:
            version_num = int(d.name)
            numeric_versions.append((version_num, d))
        except ValueError:
            # Ignorar subdirectorios que no son versiones numéricas
            continue

    # 3. Lógica para manejar la ausencia de versiones
    if not numeric_versions:
        # Crear la versión 1 si no se encontró ninguna versión numérica
        print(f"No se encontró ninguna versión. Creando versión inicial '1' en: {base_path}")
        new_version_path = base_path / "1"
        
        # Usamos ensure_path para crear el nuevo directorio de la versión '1'
        return ensure_path(new_version_path)

    # 4. Encontrar la versión numérica máxima
    latest_version_num, latest_path = max(numeric_versions, key=lambda x: x[0])

    return latest_path