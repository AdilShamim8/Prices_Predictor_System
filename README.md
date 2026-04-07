# Prices Predictor System

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![ZenML](https://img.shields.io/badge/ZenML-0.64.0-blueviolet?logo=zenml)](https://zenml.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.15.1-orange?logo=mlflow)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **production-grade, end-to-end MLOps pipeline** for house price prediction built on the [Ames Housing dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset). This project is a showcase of clean software engineering principles applied to machine learning: Strategy + Factory design patterns, ZenML pipeline orchestration, MLflow experiment tracking, and a Streamlit prediction UI — all operating together without a single hardcoded path or undeclared dependency.

---

## Architecture Overview

```
data/archive.zip
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  ZenML Training Pipeline  (pipelines/training_pipeline.py)       │
│                                                                  │
│  data_ingestion ──► handle_missing_values ──► feature_engineering│
│       │                                             │            │
│       ▼                                             ▼            │
│  outlier_detection ──► data_splitter ──► model_building          │
│                                               │                  │
│                                               ▼                  │
│                                       model_evaluator            │
│                                    (MSE + R² logged to MLflow)   │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼  run_deployment.py
┌──────────────────────────────────────────────────────────────────┐
│  ZenML Deployment Pipeline  (pipelines/deployment_pipeline.py)   │
│                                                                  │
│  training ──► mlflow_model_deployer ──► prediction server        │
│                                               │                  │
│                                               ▼                  │
│                                    inference_pipeline            │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼  streamlit run app.py
┌─────────────────────────────┐
│  Streamlit UI (app.py)      │
│  · Property form            │
│  · POST → MLflow server     │
│  · Display predicted price  │
└─────────────────────────────┘
```

### Design Patterns Used

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Strategy** | `src/` — all processing modules | Swap algorithms (e.g. mean vs median imputation) at runtime without changing call sites |
| **Factory** | `src/ingest_data.py` | Return correct `DataIngestor` subclass based on file extension |
| **Context** | `src/` — `*Handler` / `*Engineer` classes | Decouple client code from concrete strategy implementation |

---

## Project Structure

```
prices-predictor-system/
├── src/                         # Core business logic (Strategy + Factory)
│   ├── ingest_data.py           # Factory: DataIngestorFactory
│   ├── handle_missing_values.py # Strategy: Drop / Fill (mean, median, mode, constant)
│   ├── feature_engineering.py   # Strategy: Log, StandardScale, MinMaxScale, OneHotEncode
│   ├── outlier_detection.py     # Strategy: ZScore, IQR
│   ├── model_building.py        # sklearn pipeline builder
│   └── model_evaluator.py       # Strategy: RegressionModelEvaluationStrategy
│
├── steps/                       # ZenML pipeline steps (thin wrappers over src/)
│   ├── data_ingestion_step.py
│   ├── handle_missing_values_step.py
│   ├── feature_engineering_step.py
│   ├── outlier_detection_step.py
│   ├── data_splitter_step.py
│   ├── model_building_step.py
│   ├── model_evaluator_step.py
│   ├── dynamic_importer.py
│   ├── model_loader.py
│   ├── prediction_service_loader.py
│   └── predictor.py
│
├── pipelines/
│   ├── training_pipeline.py     # ZenML @pipeline: full train run
│   └── deployment_pipeline.py   # ZenML @pipeline: continuous deploy + inference
│
├── analysis/
│   └── EDA.ipynb                # Exploratory data analysis notebook
│
├── explanations/                # Design pattern reference implementations
│   ├── factory_design_pattern.py
│   ├── strategy_design_pattern.py
│   └── template_design_pattern.py
│
├── data/
│   └── archive.zip              # Raw Ames Housing dataset (ZIP)
│
├── app.py                       # Streamlit prediction UI
├── run_pipeline.py              # Entry point: train
├── run_deployment.py            # Entry point: deploy / stop
├── sample_predict.py            # CLI: send one prediction to the server
├── config.yaml                  # ZenML stack configuration
├── pyproject.toml               # Package metadata + editable install config
├── requirements.txt             # Pinned runtime dependencies
├── Makefile                     # Developer convenience targets
└── .gitignore
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/prices-predictor-system.git
cd prices-predictor-system

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install all deps + make the packages importable
pip install -r requirements.txt
pip install -e .
```

Or using the Makefile:

```bash
make install
```

### 2. Initialise ZenML

```bash
zenml init
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml stack register mlflow_stack \
    -a default \
    -o default \
    -d mlflow_deployer \
    -e mlflow_tracker
zenml stack set mlflow_stack
```

### 3. Train the Model

```bash
make train
# or:
python run_pipeline.py
```

This runs the full training pipeline: ingestion → cleaning → feature engineering → outlier removal → train/test split → model fit → evaluation. Metrics (MSE, R²) are logged to MLflow automatically.

### 4. Inspect Experiments in MLflow

```bash
mlflow ui --backend-store-uri $(zenml experiment-tracker describe --output-format=json | python -m json.tool | python -c "import sys,json; print(json.load(sys.stdin)['tracking_uri'])")
```

Then open [http://localhost:5000](http://localhost:5000).

### 5. Deploy the Model

```bash
make deploy
# or:
python run_deployment.py
```

This trains + deploys the model as a local MLflow prediction server (REST API at `http://127.0.0.1:8000/invocations`).

### 6. Launch the Streamlit App

```bash
make app
# or:
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

### 7. Send a Sample Prediction (CLI)

```bash
python sample_predict.py
# Custom URL:
python sample_predict.py --url http://127.0.0.1:8000/invocations
```

### 8. Stop the Prediction Server

```bash
make stop-deploy
# or:
python run_deployment.py --stop-service
```

---

## Pipeline Steps Reference

| Step | Input | Output | Notes |
|------|-------|--------|-------|
| `data_ingestion_step` | `file_path: str` | `pd.DataFrame` | Extracts CSV from ZIP via `DataIngestorFactory` |
| `handle_missing_values_step` | `DataFrame` | `DataFrame` | Default: mean imputation for numerics |
| `feature_engineering_step` | `DataFrame` | `DataFrame` | Log-transforms `Gr Liv Area` + `SalePrice` |
| `outlier_detection_step` | `DataFrame` | `DataFrame` | Z-score filter on `SalePrice` |
| `data_splitter_step` | `DataFrame` | `X_train, X_test, y_train, y_test` | 80/20 split |
| `model_building_step` | `X_train, y_train` | `sklearn Pipeline` | Preprocessing + Linear Regression |
| `model_evaluator_step` | model + test data | `metrics dict, MSE` | Logs to MLflow |

---

## Configuration

### `config.yaml`

```yaml
# Passed to zenml stack configure or used by the pipeline
docker_settings:
  required_integrations:
    - mlflow
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PREDICTION_URL` | `http://127.0.0.1:8000/invocations` | Override the MLflow server URL in `app.py` |

---

## Development

```bash
# Run all quality checks
make install

# Run a single step in isolation (example)
python -c "from pipelines.training_pipeline import ml_pipeline; ml_pipeline()"

# Verify no developer paths remain
grep -r "ayushsingh" . --include="*.py"   # should return nothing

# Clean up workspace artifacts
make clean
```

---

## License

MIT © 2026 — See [LICENSE](LICENSE) for details.
