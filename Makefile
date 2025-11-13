#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = MLops_E38_F2
PYTHON_VERSION = 3.13.7
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt --quiet

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m pytest tests
## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	gsutil -m rsync -r gs://DVC/data/ data/
	

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	gsutil -m rsync -r data/ gs://DVC/data/

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

#################################################################################
# Gestión del servidor MLflow (inicio, detención, estado)                      #
#################################################################################

## Levanta MLflow Server
.PHONY: mlflow-server
mlflow-server:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 0.0.0.0 \
		--port 5000

# Inicia el servidor MLflow en segundo plano
.PHONY: mlflow-start
mlflow-start:
	@if pgrep -f "mlflow" > /dev/null; then \
		echo "MLflow ya está en ejecución (PID(s): $$(pgrep -f 'mlflow' | tr '\n' ' '))."; \
	else \
		echo "Iniciando el servidor de MLflow..."; \
		nohup mlflow server \
			--backend-store-uri sqlite:///mlflow.db \
			--default-artifact-root ./mlruns \
			--host 0.0.0.0 \
			--port 5000 \
			> mlflow.log 2>&1 & \
		echo $$! > mlflow.pid; \
		sleep 3; \
		if pgrep -f "mlflow" > /dev/null; then \
			PID=$$(cat mlflow.pid 2>/dev/null || echo "desconocido"); \
			echo "MLflow iniciado correctamente (PID principal: $$PID)."; \
		else \
			echo "Error: MLflow no se pudo iniciar. Revisa el log en mlflow.log."; \
			rm -f mlflow.pid; \
		fi; \
	fi

# Detiene el servidor MLflow
.PHONY: mlflow-stop
mlflow-stop:
	@echo "Intentando detener MLflow..."
	@if [ -f mlflow.pid ]; then \
		PID=$$(cat mlflow.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "Deteniendo MLflow con PID guardado ($$PID)..."; \
			kill $$PID && rm -f mlflow.pid; \
			sleep 2; \
			if ps -p $$PID > /dev/null 2>&1; then \
				echo "Error: el proceso $$PID sigue activo."; \
			else \
				echo "MLflow detenido correctamente (PID $$PID)."; \
			fi; \
		else \
			echo "El PID guardado ($$PID) no está activo. Eliminando archivo mlflow.pid."; \
			rm -f mlflow.pid; \
		fi; \
	else \
		echo "No se encontró archivo mlflow.pid."; \
	fi; \
	\
	echo "Limpiando posibles procesos residuales de MLflow..."; \
	pkill -f "mlflow" 2>/dev/null && echo "Procesos adicionales de MLflow detenidos." || echo "No había procesos residuales de MLflow."; \
	echo "MLflow completamente detenido."

# Determina si MLflow está en ejecución
.PHONY: mlflow-status
mlflow-status:
	@if pgrep -f "mlflow" > /dev/null; then \
		echo "MLflow está en ejecución (PID(s): $$(pgrep -f 'mlflow' | tr '\n' ' '))."; \
	else \
		echo "MLflow no está en ejecución."; \
	fi
	
#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m src.data.dataset

## Make clean
.PHONY: clean_data
clean_data:
	$(PYTHON_INTERPRETER) -m src.data.cleaning

## Make FE
.PHONY: FE
FE:
	$(PYTHON_INTERPRETER) -m src.data.features

## Make train
.PHONY: train
train:
	make mlflow-start
	@$(PYTHON_INTERPRETER) -m src.modeling.train
	make mlflow-stop

## Make prepare: ejecuta data → clean_data → FE
.PHONY: prepare
prepare: data clean_data FE

## Make que ejecuta data → clean_data → FE → train
.PHONY: all
all: prepare train

## Make predict
.PHONY: predict
predict:
	make mlflow-start
	@$(PYTHON_INTERPRETER) -m src.modeling.predict
	make mlflow-stop

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

#################################################################################
# DVC COMMANDS                                                                  #
#################################################################################

## Configura DVC con remoto y credenciales
## Servicio de Google Drive
## Invoca a la aplicación dvc_setup con el parámetro gdrive para que configure al DVC
.PHONY: dvc_gdrive_setup
dvc_gdrive_setup:
	python -m src.config.dvc_setup gdrive

## Configura AWS con remoto y credenciales
## Servicio de S3 en AWS
## Invoca a la aplicación dvc_setup con el parámetro aws para que configure al AWS
.PHONY: dvc_aws_setup
dvc_aws_setup:
	python -m src.config.dvc_setup aws

## Reproduce todo el pipeline según dvc.yaml
## dvc_repro ejecuta la instrucción dvc repro, la cual es una instrucción directa de dvc, la cual esta
## asociada al contenido del archivo dvc.yaml. Considerar que la ejecución del pipeline dvc.yaml,
## hace las consideraciones de solo ejecutar lo que sea necesario, esto es una funcionalidad de la
## aplicación dvc.

.PHONY: dvc_repro
dvc_repro:
	make mlflow-start
	dvc repro
	make mlflow-stop

## Sube los datos versionados al remoto (GDrive/S3)
## El comando dvc push utiliza los archivos de configuración dvc.yaml y dvc.lock, además del 
## archivo de configuración global de DVC (.dvc/config), para determinar qué y a dónde subir los
## datos. 
## dvc.yaml le dice a DVC cómo construir el pipeline.
## dvc.lock le dice a DVC qué partes del pipeline ya están actualizadas y coinciden con una 
## versión anterior. 

.PHONY: dvc_push
dvc_push:
	dvc push

## Descarga los datos versionados del remoto
.PHONY: dvc_pull
dvc_pull:
	dvc pull

## Verifica qué etapas están desactualizadas
.PHONY: dvc_status
dvc_status:
	dvc status

#################################################################################
# DOCKER COMMANDS                                                               #
#################################################################################

## Construye la imagen Docker localmente
.PHONY: docker-build
docker-build:
	docker build -t ml-service:1.0.0 .

## Ejecuta el contenedor localmente en segundo plano (puerto 8000)
.PHONY: docker-run
docker-run:
	docker run -d --rm -p 8000:8000 --name ml-service ml-service:1.0.0

## Detiene el contenedor si está corriendo
.PHONY: docker-stop
docker-stop:
	@if [ "$$(docker ps -q -f name=ml-service)" ]; then \
		echo "Deteniendo contenedor ml-service..."; \
		docker stop ml-service; \
	else \
		echo "No hay contenedor ml-service en ejecución."; \
	fi

## Baja (pull) la imagen publicada en Docker Hub
.PHONY: docker-pull
docker-pull:
	docker pull cremercado/ml-service:1.0.0
