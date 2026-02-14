  # `Production-Grade House Price Prediction System`

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ZenML](https://img.shields.io/badge/ZenML-0.64.0-brightgreen.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.15.1-orange.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

**An end-to-end MLOps pipeline for house price prediction featuring automated training, deployment, and inference with best practices in software design patterns and machine learning operations.**

[Features](#-key-features) •
[Architecture](#-architecture) •
[Installation](#-installation) •
[Usage](#-usage) •
[Project Structure](#-project-structure) •
[Documentation](#-documentation)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Pipeline Components](#-pipeline-components)
- [Design Patterns](#-design-patterns)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## Overview

This project implements a **production-ready machine learning system** for predicting house prices using the Ames Housing dataset. Unlike typical ML projects that end at model training, this system demonstrates enterprise-grade MLOps practices including:

- **Automated ML Pipelines**: Orchestrated workflows using ZenML
- **Experiment Tracking**: Comprehensive tracking with MLflow
- **Model Versioning**: Automated model registry and versioning
- **Continuous Deployment**: Automated model deployment pipelines
- **Design Patterns**: Implementation of Factory, Strategy, and Template patterns for maintainability
- **Production API**: RESTful API for real-time predictions

The system is designed with **scalability**, **maintainability**, and **reproducibility** as core principles, making it suitable for real-world production environments.

---

##  Key Features

### MLOps Excellence

- **End-to-End Pipeline Orchestration**: Fully automated ML workflows from data ingestion to deployment
- **Experiment Tracking**: Track all experiments, metrics, and artifacts with MLflow
- **Model Registry**: Automated model versioning and promotion to production
- **Reproducibility**: Every pipeline run is tracked and reproducible

### Software Engineering Best Practices

- **Design Patterns**: Factory, Strategy, and Template patterns for clean, maintainable code
- **Modular Architecture**: Loosely coupled components for easy testing and extension
- **Type Hints**: Full type annotations for better code quality
- **Logging**: Comprehensive logging throughout the pipeline

### Advanced ML Techniques

- **Comprehensive EDA**: Univariate, bivariate, and multivariate analysis
- **Feature Engineering**: Log transformations for skewed distributions
- **Outlier Detection**: Robust outlier handling using IQR method
- **Missing Value Imputation**: Intelligent handling of missing data
- **Model Evaluation**: Multiple metrics including MSE and R²

### Deployment & Serving

- **Automated Deployment**: Continuous deployment pipeline with ZenML + MLflow
- **REST API**: Production-ready API endpoint for predictions
- **Batch Inference**: Support for batch prediction workflows
- **Model Monitoring**: Track model performance in production

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  Data Ingestion → Missing Values → Feature Engineering →        │
│  Outlier Detection → Data Splitting → Model Training →          │
│  Model Evaluation → Model Registry                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│  Load Trained Model → Deploy to MLflow Server →                 │
│  Start Prediction Service                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  Load Data → Preprocess → Predict → Return Results              │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

1. **Data Ingestion**: Flexible data loading using Factory pattern (supports ZIP, CSV, Excel)
2. **Data Preprocessing**: Handle missing values, feature engineering, outlier detection
3. **Model Training**: Train Linear Regression model with StandardScaler
4. **Model Evaluation**: Calculate MSE, R², and other metrics
5. **Model Registration**: Register model to MLflow Model Registry
6. **Deployment**: Deploy model as REST API service
7. **Inference**: Serve predictions via HTTP endpoint

---

##  Technology Stack

### Core ML & MLOps
- **[ZenML](https://zenml.io/)** `0.64.0` - ML pipeline orchestration framework
- **[MLflow](https://mlflow.org/)** `2.15.1` - Experiment tracking and model deployment
- **[scikit-learn](https://scikit-learn.org/)** `1.3.2` - Machine learning algorithms

### Data Science & Analysis
- **[Pandas](https://pandas.pydata.org/)** `2.0.3` - Data manipulation
- **[NumPy](https://numpy.org/)** `1.24.4` - Numerical computing
- **[Matplotlib](https://matplotlib.org/)** `3.7.5` - Data visualization
- **[Seaborn](https://seaborn.pydata.org/)** `0.13.2` - Statistical visualizations
- **[StatsModels](https://www.statsmodels.org/)** `0.14.1` - Statistical modeling

### Utilities
- **[Click](https://click.palletsprojects.com/)** `8.1.3` - CLI interface

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/AdilShamim8/Prices_Predictor_System.git
cd Prices_Predictor_System
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Initialize ZenML

```bash
# Initialize ZenML repository
zenml init

# Start ZenML dashboard (optional)
zenml up
```

---

## Quick Start

### Train the Model

Run the complete training pipeline:

```bash
python run_pipeline.py
```

This will:
1. Load the housing data
2. Preprocess and clean the data
3. Train the Linear Regression model
4. Evaluate model performance
5. Register the model to MLflow

### Deploy the Model

Deploy the trained model to a production-ready API:

```bash
python run_deployment.py
```

This command:
1. Runs the continuous deployment pipeline
2. Deploys the model to an MLflow server
3. Starts a REST API endpoint for predictions

### Make Predictions

#### Option 1: Use the Sample Script

```bash
python sample_predict.py
```

#### Option 2: Send HTTP Request

```python
import requests
import json

url = "http://127.0.0.1:8000/invocations"

data = {
    "dataframe_records": [{
        "Order": 1,
        "PID": 526301100,
        "MS SubClass": 20,
        "Lot Frontage": 80.0,
        "Lot Area": 9600,
        "Overall Qual": 5,
        "Overall Cond": 7,
        # ... other features
    }]
}

response = requests.post(url, headers={"Content-Type": "application/json"}, 
                        data=json.dumps(data))
prediction = response.json()
print(f"Predicted Price: ${prediction[0]:,.2f}")
```

### View Experiment Results

Start the MLflow UI to view all experiments and metrics:

```bash
mlflow ui --backend-store-uri <tracking_uri>
```

Then navigate to `http://localhost:5000` in your browser.

---

## Project Structure

```
Prices_Predictor_System/
├── analysis/                          # Exploratory Data Analysis modules
│   └── analyze_src/
│       ├── basic_data_inspection.py   # Data types and statistics inspection
│       ├── univariate_analysis.py     # Single variable analysis
│       ├── bivariate_analysis.py      # Two variable relationships
│       ├── multivariate_analysis.py   # Correlation and pair plots
│       └── missing_values_analysis.py # Missing data visualization
│
├── data/                              # Data storage
│   └── archive.zip                    # Raw housing dataset
│
├── extracted_data/                    # Extracted CSV files
│
├── pipelines/                         # ML pipeline definitions
│   ├── training_pipeline.py           # End-to-end training workflow
│   └── deployment_pipeline.py         # Deployment and inference pipelines
│
├── src/                               # Core ML components (Strategy pattern)
│   ├── ingest_data.py                 # Data ingestion with Factory pattern
│   ├── handle_missing_values.py       # Missing value imputation strategies
│   ├── feature_engineering.py         # Feature transformation strategies
│   ├── outlier_detection.py           # Outlier detection strategies
│   ├── data_splitter.py               # Train-test splitting strategies
│   ├── model_building.py              # Model training strategies
│   └── model_evaluator.py             # Model evaluation strategies
│
├── steps/                             # ZenML pipeline steps
│   ├── data_ingestion_step.py         # Data loading step
│   ├── handle_missing_values_step.py  # Missing value handling step
│   ├── feature_engineering_step.py    # Feature engineering step
│   ├── outlier_detection_step.py      # Outlier removal step
│   ├── data_splitter_step.py          # Data splitting step
│   ├── model_building_step.py         # Model training step
│   ├── model_evaluator_step.py        # Model evaluation step
│   ├── model_loader.py                # Production model loading
│   ├── dynamic_importer.py            # Dynamic data import for inference
│   ├── prediction_service_loader.py   # Load deployed prediction service
│   └── predictor.py                   # Prediction execution
│
├── explanations/                      # Design pattern examples
│   ├── factory_design_pattern.py      # Factory pattern explanation
│   ├── strategy_design_pattern.py     # Strategy pattern explanation
│   └── template_design_pattern.py     # Template pattern explanation
│
├── config.yaml                        # Configuration file
├── requirements.txt                   # Python dependencies
├── run_pipeline.py                    # Main training script
├── run_deployment.py                  # Deployment script
├── sample_predict.py                  # Sample prediction script
├── LICENSE                            # Apache 2.0 License
└── README.md                          # This file
```

---

## Pipeline Components

### 1️ Data Ingestion

**Purpose**: Flexible data loading supporting multiple formats

**Implementation**: Factory Design Pattern

```python
# Automatically selects appropriate ingestor based on file extension
data_ingestor = DataIngestorFactory.get_data_ingestor(".zip")
df = data_ingestor.ingest("data/archive.zip")
```

**Supported Formats**:
- ZIP archives
- CSV files  
- Excel spreadsheets

---

### 2️⃣ Missing Value Handling

**Purpose**: Intelligent imputation of missing data

**Implementation**: Strategy Design Pattern

**Strategies Available**:
- Mean imputation (numerical features)
- Median imputation (numerical features)
- Mode imputation (categorical features)
- Drop missing rows/columns

```python
# Example: Median imputation
handler = MissingValuesHandler(MedianImputationStrategy())
df_cleaned = handler.handle_missing_values(df)
```

---

### 3️ Feature Engineering

**Purpose**: Transform features to improve model performance

**Implementation**: Strategy Design Pattern

**Transformations**:
- **Log Transformation**: Reduces skewness in distributions (e.g., `Gr Liv Area`, `SalePrice`)
- **Standard Scaling**: Normalizes features to zero mean and unit variance
- **Min-Max Scaling**: Scales features to a specific range [0,1]

```python
# Log transformation for skewed features
engineer = FeatureEngineer(LogTransformation(features=["Gr Liv Area", "SalePrice"]))
df_transformed = engineer.apply_transformations(df)
```

---

### 4️ Outlier Detection

**Purpose**: Identify and remove extreme values that could skew predictions

**Method**: Interquartile Range (IQR) method

```python
# Remove outliers from SalePrice
detector = OutlierDetector(IQROutlierDetectionStrategy())
df_clean = detector.detect_and_remove_outliers(df, column_name="SalePrice")
```

**Formula**: 
- Lower Bound = Q1 - 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR

---

### 5️ Model Training

**Purpose**: Train regression model on processed data

**Implementation**: Strategy Design Pattern + Scikit-learn Pipeline

**Model Architecture**:
```
Input Features → StandardScaler → Linear Regression → Price Prediction
```

**Features Used**:
- Property characteristics (square footage, number of rooms, etc.)
- Quality and condition ratings
- Year built and renovation year
- Garage and basement details

---

### 6️ Model Evaluation

**Purpose**: Assess model performance using multiple metrics

**Metrics Calculated**:
- **Mean Squared Error (MSE)**: Average squared difference between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of MSE (same unit as target)
- **R² Score**: Proportion of variance explained by the model

```python
evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
metrics = evaluator.evaluate(model, X_test, y_test)
```

---

### 7️ Model Deployment

**Purpose**: Deploy trained model as REST API service

**Deployment Flow**:
1. Load best model from MLflow Model Registry
2. Deploy to MLflow server with 3 workers
3. Start HTTP endpoint at `http://127.0.0.1:8000/invocations`

**Service Management**:
```bash
# Start service
python run_deployment.py

# Stop service
python run_deployment.py --stop-service
```

---

##  Design Patterns

This project implements **three fundamental design patterns** to ensure clean, maintainable, and extensible code.

### 1. Factory Design Pattern

**Location**: `src/ingest_data.py`

**Purpose**: Create appropriate data ingestors based on file type without exposing instantiation logic.

**Benefits**:
- Easy to add support for new file formats
- Single point of instantiation logic
- Client code doesn't need to know about concrete classes

**Example**:
```python
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        if file_extension == ".zip":
            return ZipDataIngestor()
        elif file_extension == ".csv":
            return CSVDataIngestor()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
```

---

### 2. Strategy Design Pattern

**Locations**: 
- `src/feature_engineering.py`
- `src/handle_missing_values.py`
- `src/outlier_detection.py`
- `src/data_splitter.py`
- `src/model_building.py`

**Purpose**: Define a family of algorithms, encapsulate each one, and make them interchangeable.

**Benefits**:
- Switch between algorithms at runtime
- Add new strategies without modifying existing code
- Better code organization and testability

**Example**:
```python
# Define strategy interface
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# Implement concrete strategies
class LogTransformation(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        # Apply log transformation
        return df_transformed

# Use strategy
engineer = FeatureEngineer(LogTransformation(features=["SalePrice"]))
df_transformed = engineer.apply_transformations(df)
```

---

### 3. Template Design Pattern

**Locations**:
- `analysis/analyze_src/missing_values_analysis.py`
- `analysis/analyze_src/multivariate_analysis.py`

**Purpose**: Define the skeleton of an algorithm, deferring some steps to subclasses.

**Benefits**:
- Code reuse through inheritance
- Control over algorithm structure
- Extension points for customization

**Example**:
```python
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """Template method defining the analysis workflow"""
        self.identify_missing_values(df)
        self.visualize_missing_values(df)
    
    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        pass
    
    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        pass
```

---

## Model Performance

### Dataset Information

- **Dataset**: Ames Housing Dataset
- **Total Samples**: ~2,900 houses
- **Features**: 38 numerical features
- **Target Variable**: SalePrice (in USD)

### Model Metrics

After preprocessing and outlier removal:

| Metric | Value |
|--------|-------|
| **R² Score** | 0.85+ |
| **RMSE** | ~$25,000 |
| **MSE** | ~625,000,000 |

> **Note**: Actual metrics may vary based on train-test split and random seed.

### Feature Importance

Top predictive features:
1. **Overall Quality**: Overall material and finish quality rating
2. **Gr Liv Area**: Above grade living area (square feet)
3. **Total Bsmt SF**: Total square feet of basement area
4. **Year Built**: Original construction year
5. **Garage Area**: Size of garage in square feet

---

## API Documentation

### Prediction Endpoint

**URL**: `http://127.0.0.1:8000/invocations`

**Method**: POST

**Content-Type**: application/json

### Request Format

```json
{
  "dataframe_records": [
    {
      "Order": 1,
      "PID": 526301100,
      "MS SubClass": 20,
      "Lot Frontage": 80.0,
      "Lot Area": 9600,
      "Overall Qual": 5,
      "Overall Cond": 7,
      "Year Built": 1961,
      "Year Remod/Add": 1961,
      "Mas Vnr Area": 0.0,
      "BsmtFin SF 1": 700.0,
      "BsmtFin SF 2": 0.0,
      "Bsmt Unf SF": 150.0,
      "Total Bsmt SF": 850.0,
      "1st Flr SF": 856,
      "2nd Flr SF": 854,
      "Low Qual Fin SF": 0,
      "Gr Liv Area": 1710.0,
      "Bsmt Full Bath": 1,
      "Bsmt Half Bath": 0,
      "Full Bath": 1,
      "Half Bath": 0,
      "Bedroom AbvGr": 3,
      "Kitchen AbvGr": 1,
      "TotRms AbvGrd": 7,
      "Fireplaces": 2,
      "Garage Yr Blt": 1961,
      "Garage Cars": 2,
      "Garage Area": 500.0,
      "Wood Deck SF": 210.0,
      "Open Porch SF": 0,
      "Enclosed Porch": 0,
      "3Ssn Porch": 0,
      "Screen Porch": 0,
      "Pool Area": 0,
      "Misc Val": 0,
      "Mo Sold": 5,
      "Yr Sold": 2010
    }
  ]
}
```

### Response Format

```json
[182750.25]
```

### Feature Descriptions

| Feature | Description |
|---------|-------------|
| **MS SubClass** | Type of dwelling |
| **Lot Frontage** | Linear feet of street connected to property |
| **Lot Area** | Lot size in square feet |
| **Overall Qual** | Overall material and finish quality (1-10) |
| **Overall Cond** | Overall condition rating (1-10) |
| **Year Built** | Original construction date |
| **Year Remod/Add** | Remodel date |
| **Gr Liv Area** | Above grade living area (sq ft) |
| **Full Bath** | Number of full bathrooms |
| **Bedroom AbvGr** | Number of bedrooms above basement |
| **Garage Cars** | Size of garage in car capacity |
| **Garage Area** | Size of garage in square feet |

---

## Exploratory Data Analysis

The project includes comprehensive EDA modules in the `analysis/` directory:

### Basic Data Inspection

```python
from analysis.analyze_src.basic_data_inspection import (
    DataInspector, 
    DataTypesInspectionStrategy,
    SummaryStatisticsInspectionStrategy
)

inspector = DataInspector(DataTypesInspectionStrategy())
inspector.execute_inspection(df)
```

### Univariate Analysis

Analyze individual feature distributions:

```python
from analysis.analyze_src.univariate_analysis import (
    UnivariateAnalyzer,
    NumericalUnivariateAnalysis
)

analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
analyzer.execute_analysis(df, feature="SalePrice")
```

### Bivariate Analysis

Explore relationships between two features:

```python
from analysis.analyze_src.bivariate_analysis import (
    BivariateAnalyzer,
    NumericalVsNumericalAnalysis
)

analyzer = BivariateAnalyzer(NumericalVsNumericalAnalysis())
analyzer.execute_analysis(df, feature1="Gr Liv Area", feature2="SalePrice")
```

### Multivariate Analysis

Generate correlation heatmaps and pair plots:

```python
from analysis.analyze_src.multivariate_analysis import SimpleMultivariateAnalysis

analyzer = SimpleMultivariateAnalysis()
analyzer.analyze(df[['SalePrice', 'Gr Liv Area', 'Overall Qual']])
```

---

## Testing

### Run Unit Tests

```bash
pytest tests/
```

### Test Model Prediction

```bash
python sample_predict.py
```

---

## CI/CD (Future Enhancement)

Planned CI/CD pipeline stages:

1. **Lint & Format**: Code quality checks with pylint, black, isort
2. **Unit Tests**: Run pytest suite
3. **Integration Tests**: Test full pipeline execution
4. **Model Validation**: Ensure model meets performance thresholds
5. **Deployment**: Automated deployment to staging/production

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit your changes**:
   ```bash
   git commit -m "Add YourFeature"
   ```
4. **Push to the branch**:
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatter
black .

# Run linter
pylint src/ steps/ pipelines/

# Run tests
pytest tests/
```

---

## Future Enhancements

- [ ] **Advanced Models**: Implement XGBoost, Random Forest, Neural Networks
- [ ] **Hyperparameter Tuning**: Add automated hyperparameter optimization
- [ ] **Feature Selection**: Implement automated feature selection
- [ ] **Model Explainability**: Add SHAP values for model interpretation
- [ ] **Data Drift Detection**: Monitor data distribution changes
- [ ] **Model Monitoring**: Track model performance metrics in production
- [ ] **A/B Testing**: Compare multiple model versions
- [ ] **Docker Containerization**: Package application in Docker
- [ ] **Kubernetes Deployment**: Deploy to Kubernetes cluster
- [ ] **FastAPI Integration**: Replace MLflow serving with FastAPI
- [ ] **Streamlit Dashboard**: Build interactive web interface

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Adil Shamim**

- GitHub: [@AdilShamim8](https://github.com/AdilShamim8)
- Project Link: [https://github.com/AdilShamim8/Prices_Predictor_System](https://github.com/AdilShamim8/Prices_Predictor_System)

---

## Acknowledgments

- **Ames Housing Dataset**: Dean De Cock for providing the comprehensive housing dataset
- **ZenML**: For the excellent MLOps framework
- **MLflow**: For robust experiment tracking and model management
- **Open Source Community**: For the amazing tools and libraries

---

## References & Resources

### Documentation
- [ZenML Documentation](https://docs.zenml.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

### Design Patterns
- [Factory Pattern](https://refactoring.guru/design-patterns/factory-method)
- [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)
- [Template Method Pattern](https://refactoring.guru/design-patterns/template-method)

### MLOps Best Practices
- [Google MLOps: Continuous delivery and automation pipelines in ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Microsoft MLOps Maturity Model](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model)

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

</div>
