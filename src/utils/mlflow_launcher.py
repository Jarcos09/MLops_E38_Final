import os
import subprocess
import time
import requests
from src.config.config import conf
from typing import Optional


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def is_mlflow_available(tracking_uri: Optional[str]) -> bool:
    """Comprueba si un servidor MLflow está accesible en la tracking_uri.

    Para URIs http realiza una petición simple a la API; para file:// devuelve True
    porque no requiere servidor.
    """
    if not tracking_uri:
        return False

    t = str(tracking_uri)
    if t.startswith("file:") or t.startswith("file://"):
        return True

    if t.startswith("http"):
        try:
            # petición POST vacía a la API de registered-models (leve)
            resp = requests.post(f"{t}/api/2.0/mlflow/registered-models/search", json={}, timeout=2)
            return resp.status_code < 500
        except Exception:
            return False

    return False


def ensure_mlflow_server(
    tracking_uri: Optional[str],
    backend_store_uri: str = "sqlite:///mlflow.db",
    artifact_root: str = "./mlruns",
    host: str = "127.0.0.1",
    port: int = 5000,
    pid_file: str = conf.data.mlflow_pid_file,
) -> Optional[subprocess.Popen]:
    """Asegura que MLflow server esté corriendo cuando la tracking_uri apunta a HTTP local.

    Si el servidor ya responde no hace nada. Si no responde y la tracking_uri apunta
    a una dirección HTTP local (localhost/127.0.0.1) intenta arrancar `mlflow server` en background.

    Devuelve el objeto Popen si arrancó el servidor, o None si no fue necesario/posible.
    """
    if not tracking_uri or not str(tracking_uri).startswith("http"):
        return None

    # comprobar disponibilidad
    if is_mlflow_available(tracking_uri):
        return None

    # si hay pid file y proceso vivo, no arrancar
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            if _is_process_alive(pid):
                return None
        except Exception:
            pass

    # sólo arrancar si la tracking_uri apunta a localhost/127.0.0.1
    t = str(tracking_uri)
    if "localhost" not in t and "127.0.0.1" not in t:
        return None

    cmd = [
        "mlflow",
        "server",
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        os.path.abspath(artifact_root),
        "--host",
        host,
        "--port",
        str(port),
    ]

    # arrancar en background
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # escribir pid
        try:
            with open(pid_file, "w") as f:
                f.write(str(proc.pid))
        except Exception:
            pass

        # esperar brevemente a que el servidor suba
        timeout = 10
        start = time.time()
        while time.time() - start < timeout:
            if is_mlflow_available(tracking_uri):
                return proc
            time.sleep(0.5)

        # si no respondió, devolver proc de todas formas
        return proc
    except FileNotFoundError:
        # mlflow binario no encontrado en PATH
        return None


def stop_mlflow_server(pid_file: str = conf.data.mlflow_pid_file) -> None:
    """Detiene el servidor MLflow arrancado mediante ensure_mlflow_server si existe el pid file."""
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 15)
            except Exception:
                pass
        except Exception:
            pass
        try:
            os.remove(pid_file)
        except Exception:
            pass
