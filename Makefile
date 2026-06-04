PYTHON=python
PIP=pip

export PYTHONPATH=src

.PHONY: \
install \
fetch-data \
build-features \
train \
shap \
backtest \
regime \
drift \
retrain \
simulate-baseline \
simulate-retraining \
simulate-drift \
run-api \
mlflow-ui \
test \
lint \
format \
format-check \
dvc-pull \
dvc-push \
dvc-repro \
docker-build \
docker-run \
full-rebuild \
pipeline \
validate \
all \
full-monitoring \
ci \
clean


# INSTALLATION
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt


# DATA PIPELINE
fetch-data:
	$(PYTHON) -m mlops_stock_project.data.make_dataset

build-features:
	$(PYTHON) -m mlops_stock_project.features.build_features


# MODEL TRAINING
train:
	$(PYTHON) -m mlops_stock_project.models.train_model


# EVALUATION / ANALYSIS
shap:
	$(PYTHON) -m mlops_stock_project.explainability.shap_analysis

backtest:
	$(PYTHON) -m mlops_stock_project.backtesting.backtest_strategy

regime:
	$(PYTHON) -m mlops_stock_project.evaluation.regime_analysis


# MONITORING
drift:
	$(PYTHON) -m mlops_stock_project.monitoring.drift

retrain:
	$(PYTHON) -m mlops_stock_project.monitoring.retrain

full-monitoring:
	$(MAKE) drift
	$(MAKE) retrain


# SIMULATIONS
simulate-baseline:
	$(PYTHON) -m mlops_stock_project.models.simulate_baseline

simulate-retraining:
	$(PYTHON) -m mlops_stock_project.models.simulate_retraining

simulate-drift:
	$(PYTHON) -m mlops_stock_project.models.simulate_drift_retraining


# API
run-api:
	uvicorn mlops_stock_project.api.app:app \
		--reload \
		--host 0.0.0.0 \
		--port 8000 \
		--app-dir src


# MLFLOW
mlflow-ui:
	mlflow ui


# TESTING
test:
	pytest -v

validate:
	$(MAKE) train
	$(MAKE) drift
	$(MAKE) retrain
	$(MAKE) test


# LINTING / FORMATTING
lint:
	ruff check .

format:
	black .

format-check:
	black --check .


# DVC
dvc-pull:
	dvc pull

dvc-push:
	dvc push

dvc-repro:
	dvc repro


# DOCKER
docker-build:
	docker build \
		-t mlops-stock-api \
		-f dockerfiles/Dockerfile .

docker-run:
	docker run -p 8000:8000 mlops-stock-api


# FULL PIPELINES
all:
	$(MAKE) build-features
	$(MAKE) train
	$(MAKE) shap
	$(MAKE) backtest
	$(MAKE) regime

pipeline:
	$(MAKE) fetch-data
	$(MAKE) build-features
	$(MAKE) train
	$(MAKE) shap
	$(MAKE) backtest
	$(MAKE) regime
	$(MAKE) simulate-baseline
	$(MAKE) simulate-retraining
	$(MAKE) simulate-drift

full-rebuild:
	$(MAKE) clean
	$(MAKE) fetch-data
	$(MAKE) build-features
	$(MAKE) train
	$(MAKE) shap
	$(MAKE) backtest
	$(MAKE) regime


# CI PIPELINE
ci:
	ruff check .
	black --check .
	pytest -v


# CLEANUP
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -name ".DS_Store" -delete

	rm -rf reports/explainability/*
	rm -rf reports/backtesting/*
	rm -rf reports/figures/*
	rm -rf reports/monitoring/*

	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .coverage

	rm -rf mlruns