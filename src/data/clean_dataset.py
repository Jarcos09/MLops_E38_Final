# clean_dataset.py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from loguru import logger
from typing import Union, Optional

## Definimos la CLASE DatasetCleaner
## input_path: ruta del archivo CSV de entrada.
## output_path: ruta donde se guardará el CSV limpio.
## skew_threshold: define un umbral para decidir si se usa la media o la mediana al imputar datos faltantes.
## self.df: contendrá el DataFrame cargado

class DatasetCleaner:
    # Inicialización de la clase y definición de parámetros principales
    def __init__(self, input_path: Union[Path, pd.DataFrame, None], output_path: Optional[Path], skew_threshold: float):
        """input_path puede ser:
        - Path: ruta a un CSV que se leerá cuando se llame a load_dataset()
        - pd.DataFrame: un DataFrame ya cargado (se usará directamente)
        - None: permite crear el objeto y asignar `df` manualmente antes de ejecutar los pasos

        output_path puede ser None si se quiere trabajar en memoria y no guardar el CSV.
        """
        self.input_path = input_path            # Ruta del dataset original o DataFrame
        self.output_path = output_path          # Ruta de salida del dataset limpio (o None)
        self.skew_threshold = skew_threshold    # Umbral de asimetría para decidir método de imputación
        self.df = None                          # DataFrame que contendrá los datos cargados
        self._from_dataframe = isinstance(input_path, pd.DataFrame)

    # Carga del dataset original
    def load_dataset(self):
        logger.info(f"Cargando dataset original desde: {self.input_path}")  # Log informativo de carga
        # Si el input fue provisto como DataFrame, simplemente usarlo
        if isinstance(self.input_path, pd.DataFrame):
            self.df = self.input_path.copy()
            logger.debug("Dataset cargado desde DataFrame en memoria")
            return

        if self.input_path is None:
            raise ValueError("input_path es None: debe proporcionar una ruta o un DataFrame antes de llamar a load_dataset().")

        self.df = pd.read_csv(self.input_path)                              # Lectura CSV desde disco

    # Reemplazo de cadenas vacías por valores nulos
    def replace_empty_strings(self):
        logger.info("Reemplazando strings vacíos por NaN")                  # Log de inicio de reemplazo
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)         # Sustituye espacios vacíos por NaN

    # Conversión de todas las columnas a tipo numérico
    def convert_to_numeric(self):
        logger.info("Convirtiendo columnas a tipo numérico")                # Log de conversión
        self.df = self.df.apply(pd.to_numeric, errors='coerce')             # Convierte columnas; valores no numéricos se vuelven NaN

    # Imputación de valores faltantes basada en la asimetría de cada columna
    def impute_missing_values(self):
        logger.info("Imputando valores faltantes según asimetría")          # Log de imputación
        for col in self.df.columns:                                         # Itera sobre todas las columnas
            col_skew = skew(self.df[col].dropna())                          # Calcula asimetría excluyendo NaN
            if -self.skew_threshold <= col_skew <= self.skew_threshold:     # Si la asimetría es baja
                self.df[col] = self.df[col].fillna(self.df[col].mean())     # Imputa con la media
            else:
                self.df[col] = self.df[col].fillna(self.df[col].median())   # Imputa con la mediana si la asimetría es alta

    # Guardado del dataset limpio
    def save_cleaned_dataset(self):
        if self.output_path is None:
            logger.warning("output_path es None: no se guardará el dataset en disco; se devolverá el DataFrame en memoria.")
            return self.df

        logger.success(f"Guardando dataset limpio en: {self.output_path}")  # Log de guardado exitoso
        self.df.to_csv(self.output_path, index=False)                       # Exporta el DataFrame limpio a CSV
        return self.df

    # Ejecución completa del proceso de limpieza
    def run(self):
        self.load_dataset()             # Carga los datos (desde Path o DataFrame)
        self.replace_empty_strings()    # Limpia strings vacíos
        self.convert_to_numeric()       # Convierte a numérico
        self.impute_missing_values()    # Imputa valores faltantes
        return self.save_cleaned_dataset()     # Guarda el dataset limpio (o devuelve DataFrame si output_path es None)