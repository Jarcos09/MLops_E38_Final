# src/data/preprocess_data.py
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import joblib
from loguru import logger
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, input_path: Path, output_paths: dict, config: dict):
        """
        Inicializa el preprocesador con rutas y configuración.

        Parámetros
        input_path : Path
            Ruta del CSV limpio a procesar.
        output_paths : dict
            Rutas de salida para `X_TRAIN`, `X_TEST`, `Y_TRAIN`, `Y_TEST`.
        config : dict
            Configuración del preprocesamiento (p. ej. `target_columns`, `feature_columns`,
            `encoding`, `test_size`, `random_state`, `target_transform`).
        """
        self.input_path = input_path        # Ruta del dataset limpio
        self.output_paths = output_paths    # Diccionario de rutas de salida
        self.config = config                # Configuración de preprocesamiento
        self.df = None                      # DataFrame cargado en memoria
        self.X = None                       # Variables predictoras
        self.y = None                       # Variables objetivo
        self.pipeline = None                # Pipeline de codificación
        self.feature_names = None           # Nombres de características después de codificación

    def load_data(self):
        """
        Carga el dataset limpio desde la ruta especificada en `input_path`
        y lo almacena en memoria como un DataFrame.
        """
        logger.info(f"Cargando dataset limpio desde: {self.input_path}")    # Log de carga
        self.df = pd.read_csv(self.input_path)                              # Lectura CSV

    def separate_variables(self):
        """
        Separa el DataFrame cargado en variables predictoras (`X`)
        y variables objetivo (`y`), de acuerdo con las columnas definidas en `config`.
        """
        # Obtener listas de columnas esperadas desde la configuración
        feature_cols = list(self.config.get("feature_columns", []))
        target_cols = list(self.config.get("target_columns", []))

        # Comprobar que el DataFrame tenga todas las columnas requeridas
        missing_features = [c for c in feature_cols if c not in self.df.columns]
        missing_targets = [c for c in target_cols if c not in self.df.columns]
        missing = missing_features + missing_targets

        # Comportamiento configurables: si la config incluye allow_missing_columns=True,
        # permitimos continuar rellenando columnas faltantes con NaN; en caso contrario
        # mantenemos el comportamiento estricto y levantamos un error.
        allow_missing = bool(self.config.get("allow_missing_columns", True))

        if len(missing) > 0 and not allow_missing:
            # No proceder si faltan columnas; lanzar excepción para que el llamador lo gestione
            raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing}")

        # Si permitimos columnas faltantes, crear esas columnas con NaN para mantener el orden
        if len(missing) > 0 and allow_missing:
            logger.warning(f"Faltan columnas {missing} pero 'allow_missing_columns' está activado: se rellenarán con NaN.")
            # Añadir columnas faltantes en el DataFrame original antes de seleccionar
            for col in missing_features:
                if col not in self.df.columns:
                    self.df[col] = np.nan
            for col in missing_targets:
                if col not in self.df.columns:
                    self.df[col] = np.nan

        # Seleccionar únicamente las columnas solicitadas (ignorando otras columnas adicionales)
        # Si faltaban y allow_missing=True, ahora existirán (rellenas con NaN)
        self.X = self.df[feature_cols].copy()
        self.y = self.df[target_cols].copy()

    def encode_features(self):
        """
        Configura la codificación categórica de las variables mediante `OneHotEncoder`
        dentro de un `Pipeline`, según los parámetros definidos en `config["encoding"]`.
        """
        cat_features = self.X.columns.tolist()                          # Todas las columnas como categóricas
        self.X[cat_features] = self.X[cat_features].astype("category")  # Conversión a tipo category

        encoder = OneHotEncoder(
            drop=self.config["encoding"]["drop"],
            sparse_output=self.config["encoding"]["sparse_output"],
            handle_unknown=self.config["encoding"]["handle_unknown"]
        )

        self.pipeline = Pipeline(steps=[
            ("preprocessor", ColumnTransformer(
                transformers=[("cat", encoder, cat_features)],
                remainder='drop'
            ))
        ])

    def split_and_transform(self):
        """
        Divide los datos en conjuntos de entrenamiento y prueba, aplica la transformación
        de las variables objetivo (`PowerTransformer`) y ejecuta la codificación definida
        en el pipeline.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"]
        )

        yao = PowerTransformer(method=self.config["target_transform"])                          # Transformación de columnas objetivo
        y_train = pd.DataFrame(yao.fit_transform(y_train), columns=self.y.columns, index=y_train.index)
        y_test = pd.DataFrame(yao.transform(y_test), columns=self.y.columns, index=y_test.index)

        X_train_proc = self.pipeline.fit_transform(X_train)                                     # Ajusta y transforma entrenamiento
        X_test_proc = self.pipeline.transform(X_test)                                           # Transforma prueba
        self.feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()  # Nombres de columnas

        return X_train_proc, X_test_proc, y_train, y_test, X_train.index, X_test.index
    
    def generate_gmm(self, X_train_proc, dest: Path | str | None = None):
        """
        Genera un modelo GMM y lo guarda junto con feature_names.
        """
        n_components = self.config["gmm_n_components"]
        random_state = self.config["gmm_random_state"]
        gmm_file = dest or self.config["gmm_file"]

        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(X_train_proc)

        payload = {
            "model": gmm,
            "feature_names": self.feature_names
        }

        if gmm_file:
            joblib.dump(payload, gmm_file)
            logger.info(f"GMM guardado en: {gmm_file}")
        else:
            logger.warning("No se proporcionó ruta para guardar el modelo GMM.")

    def save_outputs(self, X_train_proc, X_test_proc, y_train, y_test, train_idx, test_idx):
        """
        Guarda los conjuntos procesados (X e y) en formato CSV en las rutas
        definidas en `output_paths`. Incluye los nombres de características generadas.
    
        Parámetros
        X_train_proc : np.ndarray
            Matriz de variables predictoras transformadas del conjunto de entrenamiento.
    
        X_test_proc : np.ndarray
            Matriz de variables predictoras transformadas del conjunto de prueba.
    
        y_train : pd.DataFrame
            Conjunto de variables objetivo transformadas correspondientes al entrenamiento.
    
        y_test : pd.DataFrame
            Conjunto de variables objetivo transformadas correspondientes a la prueba.
    
        train_idx : pd.Index
            Índices originales del conjunto de entrenamiento, usados para mantener la referencia al exportar.
    
        test_idx : pd.Index
            Índices originales del conjunto de prueba, usados para mantener la referencia al exportar.
        """
        pd.DataFrame(X_train_proc, columns=self.feature_names, index=train_idx).to_csv(self.output_paths["X_TRAIN"], index=False)
        pd.DataFrame(X_test_proc, columns=self.feature_names, index=test_idx).to_csv(self.output_paths["X_TEST"], index=False)
        y_train.to_csv(self.output_paths["Y_TRAIN"], index=False)
        y_test.to_csv(self.output_paths["Y_TEST"], index=False)
        logger.success("Preprocesamiento completado y archivos guardados.")  # Log de éxito

    def save_preprocessor(self, dest: Path | str | None = None):
        """Guarda el pipeline de preprocesamiento (pipeline) en disco.

        Si `dest` es None, se intentará leer la ruta desde
        self.config.get('preprocessor_file'). Se usa `paths.ensure_path`
        para crear la carpeta si es necesario.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline no está inicializado. Ejecuta encode_features() y ajusta el pipeline antes de guardar.")

        target = dest or self.config["preprocessor_file"]
        if target is None:
            raise ValueError("No se proporcionó ruta para guardar el preprocessor (dest es None y config['preprocessor_file'] no existe).")

        joblib.dump(self.pipeline, target)
        logger.info(f"Preprocessor serializado guardado en: {target}")
        return target

    def run(self):
        """
        Ejecuta de forma secuencial todo el pipeline de preprocesamiento:
        carga, separación, codificación, transformación y guardado de datos.
        """
        self.load_data()                                                                                # Carga de dataset
        self.separate_variables()                                                                       # Separación X e y
        self.encode_features()                                                                          # Configuración de codificación
        X_train_proc, X_test_proc, y_train, y_test, train_idx, test_idx = self.split_and_transform()    # División y transformación
        self.generate_gmm(X_train_proc)                                                                 # Generación y guardado de GMM
        self.save_outputs(X_train_proc, X_test_proc, y_train, y_test, train_idx, test_idx)
        
        # Persistir el pipeline de preprocesamiento (opcional pero recomendado)
        try:
            self.save_preprocessor()
        except Exception as e:
            # No interrumpir el flujo si no se puede guardar el pipeline
            logger.warning(f"Advertencia: no se pudo guardar el preprocessor serializado: {e}")         # Guardado de archivos