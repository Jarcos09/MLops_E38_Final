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