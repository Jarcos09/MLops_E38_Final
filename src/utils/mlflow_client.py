import mlflow
from mlflow.tracking import MlflowClient
import requests
from typing import List, Optional, Dict
from src.utils import paths


class MLFlowClient:
    """Encapsula interacciones con MLflow Tracking/Registry.

    Provee métodos para:
    - inicializar el cliente con una tracking URI (http o file:...)
    - comprobar disponibilidad remota
    - listar modelos registrados
    - obtener la última versión de un modelo
    - renderizar una tabla de texto con los modelos y versiones
    """

    def __init__(self, tracking_uri: Optional[str] = None, http_timeout: float = 3.0):
        self.raw_uri = tracking_uri
        self.tracking_uri = paths.normalize_mlflow_uri(tracking_uri)
        self.http_timeout = http_timeout

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def check_remote_available(self) -> None:
        """Si la URI es HTTP, verifica que el servidor MLflow responda a una llamada simple.

        Lanza requests.exceptions.RequestException en caso de fallo.
        """
        if not self.tracking_uri:
            return

        if self.tracking_uri.startswith("http"):
            url = f"{self.tracking_uri}/api/2.0/mlflow/registered-models/search"
            # GET vacío es la manera recomendada por la API para listar
            resp = requests.get(url, json={}, timeout=self.http_timeout)
            resp.raise_for_status()

    def list_registered_models(self) -> List[Dict]:
        """Devuelve la lista de modelos registrados (raw objects retornados por Mlflow client).

        Fallback a search_registered_models si list_registered_models no existe.
        """
        try:
            return self.client.list_registered_models()
        except Exception:
            # algunos clientes/versions exponen search_registered_models
            try:
                return self.client.search_registered_models()
            except Exception:
                # si tampoco funciona, propagar
                raise

    def search_model_versions(self, model_name: str):
        return self.client.search_model_versions(f"name='{model_name}'")
    
    def get_model_version(self, model_name: str, version: str):
        """Obtiene información de una versión específica del modelo desde el registry."""
        try:
            mv = self.client.get_model_version(name=model_name, version=version)
            return {
                "version": getattr(mv, "version", None),
                "stage": getattr(mv, "current_stage", None) or getattr(mv, "stage", None),
                "source": getattr(mv, "source", None),
                "run_id": getattr(mv, "run_id", None),
            }
        except Exception as e:
            raise RuntimeError(
                f"No se pudo obtener la versión {version} del modelo '{model_name}': {e}"
            )


    def get_latest_version(self, model_name: str) -> Optional[Dict]:
        """Retorna la última versión (numérica mayor) del modelo como dict.

        El diccionario contiene: version, stage, source, run_id.
        Si no hay versiones devuelve None.
        """
        versions = []
        for mv in self.search_model_versions(model_name):
            v = getattr(mv, "version", None)
            if v is not None:
                try:
                    versions.append(int(v))
                except Exception:
                    # versión no numérica, intentar comparar como string
                    versions.append(v)

        if not versions:
            return None

        # Determinar mayor versión (numérica si es posible)
        try:
            max_v = max(int(v) for v in versions)
            max_v_str = str(max_v)
        except Exception:
            max_v_str = str(max(versions))

        # buscar el ModelVersion que coincida con max_v_str
        for mv in self.search_model_versions(model_name):
            if getattr(mv, "version", None) == max_v_str:
                return {
                    "version": mv.version,
                    "stage": getattr(mv, "current_stage", None) or getattr(mv, "stage", None),
                    "source": getattr(mv, "source", None),
                    "run_id": getattr(mv, "run_id", None),
                }

        return None

    def render_models_table(self) -> str:
        """Construye una tabla de texto con los modelos y sus versiones."""
        registered = self.list_registered_models()
        if not registered:
            return "No hay modelos registrados en MLflow.\n"

        headers = ["Model Name", "Last Version", "Available Versions"]
        rows = []

        for rm in registered:
            name = getattr(rm, "name", "—")
            versions = []
            last_version = None

            search_res = self.search_model_versions(name)
            for mv in search_res:
                v = getattr(mv, "version", None)
                if v:
                    versions.append(f"{v}")
                    last_version = v if not last_version or int(v) > int(last_version) else last_version

            versions_str = ", ".join(versions) if versions else "—"
            last_version_str = f"{last_version}" if last_version else "—"

            rows.append([name, last_version_str, versions_str])

        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(3)]

        def fmt_row(row):
            return "| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(3)) + " |"

        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

        lines = [sep, fmt_row(headers), sep]
        for row in rows:
            lines.append(fmt_row(row))
        lines.append(sep)

        table_text = f"MLflow Tracking URI: {self.tracking_uri}\n\n" + "\n".join(lines) + "\n"
        return table_text
