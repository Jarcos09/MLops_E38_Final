# src/data/features.py
import typer
from src.data.preprocess_data import DataPreprocessor
from src.config.config import conf
from src.utils import paths

app = typer.Typer()

# Comando principal de la aplicación para preprocesamiento de datos
@app.command()
def preprocess():
    # Asegura que exista la carpeta de salida para datos procesados
    paths.ensure_path(conf.paths.processed)
    paths.ensure_path(conf.paths.models)

    # Inicialización del objeto DataPreprocessor con rutas y configuraciones
    preprocessor = DataPreprocessor(
        input_path=conf.data.interim_data_file,                     # Ruta del dataset intermedio
        output_paths={                                              # Diccionario de rutas para guardar conjuntos procesados
            "X_TRAIN": conf.data.processed_data.x_train_file,
            "X_TEST": conf.data.processed_data.x_test_file,
            "Y_TRAIN": conf.data.processed_data.y_train_file,
            "Y_TEST": conf.data.processed_data.y_test_file
        },
        config={                                                                # Configuración de preprocesamiento
            "target_columns": conf.preprocessing.target_columns,                # Columnas objetivo
            "feature_columns": conf.preprocessing.feature_columns,              # Columnas de características
            "allow_missing_columns": conf.preprocessing.allow_missing_columns,  # Permitir columnas faltantes
            "encoding": {                                                       # Parámetros de codificación
                "drop": conf.preprocessing.encoding.drop,
                "sparse_output": conf.preprocessing.encoding.sparse_output,
                "handle_unknown": conf.preprocessing.encoding.handle_unknown
            },
            "test_size": conf.preprocessing.test_size,                          # Tamaño del conjunto de prueba
            "random_state": conf.preprocessing.random_state,                    # Semilla para reproducibilidad
            "target_transform": conf.preprocessing.target_transform,            # Transformaciones para columnas objetivo
            "preprocessor_file": conf.preprocessing.preprocessor_file,          # Archivo para guardar el preprocesador serializado
            "gmm_n_components": conf.preprocessing.gmm_n_components,            # Número de componentes para GMM
            "gmm_random_state": conf.preprocessing.gmm_random_state,            # Semilla para GMM
            "gmm_file": conf.preprocessing.gmm_file                             # Archivo para guardar el modelo GMM
        }
    )

    # Ejecución completa del preprocesamiento
    preprocessor.run()  # Limpieza, transformación, codificación y división del dataset

# Ejecución del CLI cuando se ejecuta el script directamente
if __name__ == "__main__":
    app()  # Lanza la aplicación Typer