# src/dataset.py
import typer
from src.data.download_dataset import DatasetDownloader
from src.config.config import conf
from src.utils import paths

app = typer.Typer()

# Comando principal de la aplicación para descarga de datasets
@app.command()
def download():
    # Asegura que exista la ruta de salida intermedia
    paths.ensure_path(conf.paths.raw)

    # Inicialización del objeto DatasetDownloader con ID y ruta de salida
    downloader = DatasetDownloader(
        dataset_id=conf.download.dataset_id,    # ID del dataset a descargar
        output_path=conf.data.raw_data_file     # Ruta donde se guardará el dataset descargado
    )
    
    # Ejecución del proceso de descarga
    downloader.run()    # Descarga el dataset y lo guarda en la ruta especificada

# Ejecución del CLI cuando se ejecuta el script directamente
if __name__ == "__main__":
    app()   # Lanza la aplicación Typer
