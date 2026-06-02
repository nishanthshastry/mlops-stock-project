PYTHON=python
PIP=pip

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

fetch-data:
	$(PYTHON) -m mlops_stock_project.data.make_dataset

build-features:
	$(PYTHON) -m mlops_stock_project.features.build_features

train:
	$(PYTHON) -m mlops_stock_project.models.train_model

drift:
	$(PYTHON) -m mlops_stock_project.monitoring.drift

retrain:
	$(PYTHON) -m mlops_stock_project.monitoring.retrain

run-api:
	uvicorn mlops_stock_project.api.app:app \
	--reload \
	--host 0.0.0.0 \
	--port 8000 \
	--app-dir src

mlflow-ui:
	mlflow ui

test:
	pytest -v

lint:
	ruff check .

format:
	black .

format-check:
	black --check .

dvc-pull:
	dvc pull

dvc-push:
	dvc push

dvc-repro:
	dvc repro

docker-build:
	docker build \
	-t mlops-stock-api \
	-f dockerfiles/Dockerfile .

docker-run:
	docker run -p 8000:8000 mlops-stock-api

ci:
	ruff check .
	black --check .
	pytest -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf mlruns