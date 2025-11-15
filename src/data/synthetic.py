# src/data/generate_synthetic_data.py
import typer
from src.data.synthetic_generator import SyntheticDataGenerator
from src.config.config import conf
from src.utils import paths

app = typer.Typer()

# Comando principal de la aplicación para preprocesamiento de datos
@app.command()
def preprocess():
    # Asegura que exista la carpeta de salida para datos procesados
    paths.ensure_path(conf.paths.synthetic)
    paths.ensure_path(conf.paths.models)
    
    generator = SyntheticDataGenerator(
        config={                                                        # Configuración de preprocesamiento
            "gmm_file": conf.preprocessing.gmm_file,                    # Columnas objetivo
            "n_samples": conf.synthetic.n_samples,                      # Columnas de características
            "apply_drift": conf.synthetic.apply_drift,                  # Permitir columnas faltantes
            "gamma_reweight": conf.synthetic.gamma_reweight, 
            "lowcard_uniform_mix": conf.synthetic.lowcard_uniform_mix,
            "mean_shift_std": conf.synthetic.mean_shift_std,
            "var_scale": conf.synthetic.var_scale
        },
        output_path=conf.data.synthetic_data_file
    )

    generator.run()

# Ejecución del CLI cuando se ejecuta el script directamente
if __name__ == "__main__":
    app()  # Lanza la aplicación Typer
