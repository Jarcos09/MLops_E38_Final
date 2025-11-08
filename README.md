# E38_Fase_3

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Fase 3 Avance de Proyecto, Gestion del Proyecto de Machine Learning

## Project Organization

```
├── LICENSE                         <- Open-source license if one is chosen
├── Makefile                        <- Makefile with convenience commands like `make data` or `make train`
├── README.md                       <- The top-level README for developers using this project.
├── params.yaml                     <- Centralized configuration file for pipeline parameters.
├── data                
│   ├── external                    <- Data from third party sources.
│   ├── interim                     <- Intermediate data that has been transformed.
│   ├── processed                   <- The final, canonical data sets for modeling.
│   └── raw                         <- The original, immutable data dump.
│               
├── docs                            <- A default mkdocs project; see www.mkdocs.org for details
│               
├── models                          <- Trained and serialized models, model predictions, or model summaries
│               
├── notebooks                       <- Jupyter notebooks. Naming convention is a number (for ordering),
│                                      the creator's initials, and a short `-` delimited description, e.g.
│                                      `1.0-jqp-initial-data-exploration`.
│               
├── pyproject.toml                  <- Project configuration file with package metadata for 
│                                      MLFlow/DVC and configuration for tools like black
│               
├── references                      <- Data dictionaries, manuals, and all other explanatory materials.
│               
├── reports                         <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                     <- Generated graphics and figures to be used in reporting
│               
├── requirements.txt                <- The requirements file for reproducing the analysis environment, e.g.
│                                      generated with `pip freeze > requirements.txt`
│               
├── setup.cfg                       <- Configuration file for flake8
├── dvc.yaml                        <- DVC pipeline definition
├── dvc.lock                        <- Locked versions of DVC tracked files
│               
└── src                             <- Source code for the project
    ├── __init__.py                 <- Makes `src` a Python module
    ├── utils
    │   ├── __init__.py
    │   ├── cmd.py                  <- Helper functions to execute shell commands
    │   └── paths.py                <- Paths manager to create and ensure directories
    ├── config
    │   ├── __init__.py
    │   ├── dvc_setup.py            <- Functions to set dvc repos
    │   └── config.py               <- Store useful variables and configuration
    ├── data
    │   ├── __init__.py
    │   ├── clean_dataset.py        <- Script to clean raw data
    │   ├── cleaning.py             <- Main cleaning scripts
    │   ├── dataset.py              <- Scripts to download or generate data
    │   ├── download_dataset.py     <- Scripts to fetch datasets from external sources
    │   ├── features.py             <- Code to create features for modeling
    │   └── preprocess_data.py      <- Preprocessing pipelines for ML
    └── modeling
        ├── __init__.py
        ├── plots_modeling.py       <- Plot logic to generate figures
        ├── plots.py                <- Code to create visualizations
        ├── predict_model.py        <- Model prediction logic and MLFlow integration
        ├── predict.py              <- Code to run model inference with trained models
        ├── train_model.py          <- Model training logic and MLFlow integration
        └── train.py                <- Entry point to train models
```

--------

# Fase 2 | Avance de Proyecto
# Equipo 38

En esta actividad se continuará con el desarrollo del proyecto, dando seguimiento a los avances realizados en la Fase 1. Se mantendrá la propuesta de valor, el análisis elaborado con el ML Canvas, así como los datos, modelos y experimentos previamente desarrollados. El objetivo ahora es estructurar el proyecto de Machine Learning de forma profesional, aplicando buenas prácticas como la refactorización del código, el control de versiones, el seguimiento de experimentos, el registro de métricas y modelos, y el aseguramiento de la reproducibilidad.

--------

## 🎯 Objetivos

- Continuar con el desarrollo de proyectos de Machine Learning, a partir de los requerimientos, una propuesta de valor y un conjunto de datos preprocesados.
- Estructurar proyectos de Machine Learning de manera organizada (utilizando el template de Cookiecutter)
- Aplicar buenas prácticas de codificación en cada etapa del pipeline y realizar Refactorización del código.
- Registrar métricas y aplicar control de versiones  a los experimentos utilizando herramientas de loging y tracking  (MLFlow/DVC)
- Visualizar y comparar resultados (métricas) y gestionar el registro de los modelos (Data Registry MLFlow/DVC)

--------

## 👥 Roles del Equipo
| Integrante | Matrícula | Rol |
|---|---|---|
| Jaime Alejandro Mendívil Altamirano| `A01253316` | SRE / DevOps |
| Christian Erick Mercado Flores | `A00841954` | Software Engineer  |
| Saul Mora Perea | `A01796295` | Data Engineer  |
| Juan Carlos Pérez Nava | `A01795941` | Data Scientist  |
| Mario Javier Soriano Aguilera | `A01384282` | ML Engineer  |

--------

## 📦 Instalar paqueterías
```bash
pip install -r requirements.txt --quiet
```
## 💼 Clonar repositorio
```bash
git clone https://github.com/Jarcos09/MLops_E38_F2.git
cd MLops_E38_F2/
```

--------
🔧 Recomendación previa a la ejecución

Antes de ejecutar cualquier comando con make, asegúrate de:
- Estar ubicado en la carpeta raíz del proyecto.
- Tener activado el ambiente virtual correspondiente.
Esto garantiza que las rutas, dependencias y configuraciones se interpreten correctamente durante la ejecución automatizada.


## 📚 Makefile

Descargar Dataset:
```bash
make data
```

Realizar limpieza del Dataset:
```bash
make clean_data
```

Realizar FE:
```bash
make FE
```

Ejecuta (data → clean_data → FE):
```bash
make prepare
```

Ejecutar localmente servidor de MLFlow:
```bash
make mlflow-server
```

Inicia el servidor MLFLow:
```bash
make mlflow-start
```

Detiene el servidor MLFLow:
```bash
make mlflow-stop
```

Verifica si el servidor MLFLow está activo:
```bash
make mlflow-status
```

Realizar entrenamiento:
```bash
make train
```

Ejecuta (data → clean_data → FE → train):
```bash
make all
```

Realizar preducción:
```bash
make predict
```

Configuración completa de DVC GDRIVE remoto:
```bash
make dvc_gdrive_setup
```

Configuración completa de DVC AWS remoto:
```bash
make dvc_aws_setup
```

Ejecutar el pipeline completo de DVC (data → clean → FE → train → predict):
```bash
make dvc_repro
```

Subir los outputs del pipeline al remoto:
```bash
make dvc_push
```

Descargar los datos versionados del remoto:
```bash
make dvc_pull
```

Verificar qué etapas del pipeline están desactualizadas:
```bash
make dvc_status
```

--------

## 🧠 MLflow

**MLflow** es una herramienta para gestionar el ciclo de vida de modelos de Machine Learning: rastrea experimentos, guarda métricas y versiona modelos.

---

### Iniciar servidor local

Se puede utilizar el comando:
```bash
make dvc_setup
```

También se puede ejecutar el servidor en modo local con SQLite y carpeta `mlruns`:
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
````

### Interfaz
http://localhost:5000

### Integración en el Proyecto
* `train_model.py`: Registra métricas, parámetros y modelos (Random Forest, XGBoost).

* `predict_model.py`: Usa modelos registrados para generar predicciones.

* `config/config.py`: Define la URI de tracking (mlflow_tracking_uri).

--------

## 💾 DVC

### Inicialización de Repositorio DVC

Se puede utilizar el comando:
```bash
make dvc_gdrive_setup
```
o

```bash
make dvc_aws_setup
```

También, se puede inicializar manualmente de la siguiente manera:
```bash
dvc init
```
### GDRIVE
#### Agregar Repositorio DVC (GDrive)
```bash
dvc remote add -d data "$GDRIVE_REMOTE_URL"
```

#### Configuración de DVC (GDrive)
```bash
dvc remote modify data gdrive_client_id "$GDRIVE_CLIENT_ID"
dvc remote modify data gdrive_client_secret "$GDRIVE_CLIENT_SECRET"
```

### AWS
#### Agregar Repositorio DVC (AWS)
```bash
dvc remote add -d data "$AWS_REMOTE_URL"
```

#### Configuración de DVC (AWS)
```bash
dvc remote modify team_remote region "$AWS_REGION"
dvc remote modify team_remote profile "$AWS_PROFILE"
```

### Verificar Repositorios DVC Configurados
```bash
dvc remote list
```

### Repositorio DVC (GDrive)
[Carpeta Principal del Proyecto en Google Drive](https://drive.google.com/drive/u/2/folders/1VnjNYOpP2uSaaUtFdRzW45iwZJUbt-5v)

### Repositorio DVC (AWS)
Lista todos los objetos dentro de todos los subdirectorios:
```bash
aws s3 ls s3://itesm-mna/202502-equipo38 --recursive --profile equipo38 | head
``` 

--------

## 📊 Plots

### Generar plots

Ejemplo de histograma:
```bash
python -m src.modeling.plots --plot-type histogram --column X3 --filename x3_hist.png
```

Ejemplo de scatter plot:
```bash
python -m src.modeling.plots --plot-type scatter --x X1 --y Y1 --filename x1_y1_scatter.png
```

Ejemplo de correlation matrix:
```bash
python -m src.modeling.plots --plot-type correlation --filename corr_matrix.png
```

--------

## 🚀 Model serving (FastAPI)

Servicio HTTP para exponer el modelo entrenado.

---

### Endpoints

Se cuenta con `2` endpoints:
- Examinación de operatividad: `GET /health`
- Predicción: `POST /predict` (JSON)

#### Endpoint de predicción
Esquema de entrada (JSON):

{
  "model_type": "xgb",
  "instances": [
    {"X1": 1.0, "X2": 2.0, "X3": 0.5},
    { ... }
  ]
}

Ejemplo de respuesta:

{
    "predictions": [
        {"prediction": 0.123},
        {"prediction": 0.456}
    ]
}

### Ejecución del servicio localmente:

```bash
pip install -r requirements.txt
# Desde la raíz del proyecto
# Si tu app está en `src.api.app` o `src.serving.app` ajusta el módulo en consecuencia.
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

###  Ejemplo request `curl`:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" \
  -d '{"model_type":"xgb","instances":[{"X1":0.98,"X2":514.5,"X3":294.0,"X4":110.25,"X5":7.0,"X6":2.0,"X7":0.0,"X8":0.0}]}'
```

### Ruta y versión del artefacto del modelo

El proyecto registra modelos en MLflow y también guarda un artefacto local. Se puede referenciar el artefacto usando dos formas:

- Ruta local (archivo): `models/rf_regressor.pkl` (configurado en `params.yaml`, propiedad `training.rf_model_file`).
- Registro MLflow (Model Registry): `models:/RFRegressor/<version>` (por ejemplo `models:/RFRegressor/1`).

---

## 📦 Contenerizar la API (Docker)

Se provee un `Dockerfile` en la raíz del proyecto para construir una imagen reproducible que incluya el servicio FastAPI y los artefactos del proyecto (incluyendo `models/` si lo deseas copiar dentro de la imagen).

1) Construir la imagen (ejemplo tag semántico):

```bash
docker build -t ml-service:latest .
# o versión explícita
docker build -t cremercado/ml-service:1.0.0 .
```

2) Ejecutar localmente (mapea puerto 8000):

```bash
docker run --rm -p 8000:8000 cremercado/ml-service:1.0.0
```

3) Publicar en Docker Hub (pasos):

```bash
# 1) Taguear la imagen local con tu repo en Docker Hub
docker tag ml-service:latest cremercado/ml-service:1.0.0

# 2) Iniciar sesión (te pedirá usuario/contraseña)
docker login

# 3) Push
docker push cremercado/ml-service:1.0.0
```

Tagging / versioning policy recomendada:
- `cremercado/ml-service:1.0.0`  — versión fijada por release
- `cremercado/ml-service:latest` — apuntar a la última imagen publicada
- `cremercado/ml-service:staging` — para despliegues de pre-producción