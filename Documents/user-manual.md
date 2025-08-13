# Weather Prediction System User Manual

## Table of Contents
1. Introduction
2. System Architecture
3. Installation and Setup
4. Using the Application
5. API Reference
6. ML Pipeline
7. Monitoring
8. Troubleshooting
9. Advanced Configuration

## Introduction

This weather prediction system provides accurate weather forecasts for multiple cities using machine learning models. The system includes data collection, model training, prediction, monitoring, and a user-friendly interface to view forecasts and historical data.

## System Architecture

The system consists of the following components:

- **Backend API**: FastAPI service providing prediction data and metrics
- **Frontend**: React web application for user interaction
- **ML Pipeline**: Data collection, processing, prediction, and model retraining
- **Airflow**: Workflow orchestration for scheduled tasks
- **MLFlow**: ML experiment tracking and model registry
- **Monitoring**: Prometheus and Grafana for system monitoring

## Installation and Setup

### Prerequisites

- Docker and Docker Compose
- Git
- OpenWeatherMap API key

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project-3
   ```

2. Create a `.env` file with the following variables:
   ```
   OPENWEATHER_API_KEY=your_api_key_here
   REACT_APP_API_URL=http://localhost:8000
   GRAFANA_ADMIN_PASSWORD=admin
   AIRFLOW_UID=50000
   _AIRFLOW_WWW_USER_USERNAME=airflow
   _AIRFLOW_WWW_USER_PASSWORD=airflow
   ```

3. Start the entire system:
   ```bash
   docker-compose up -d
   ```

   Or start only the application components:
   ```bash
   docker-compose -f docker-compose.app.yml up -d
   ```

4. Initialize the storage structure:
   ```bash
   docker exec weather-prediction-backend python -m src.data.storage_setup
   ```

## Using the Application

### Accessing the Web Interface

Open your browser and navigate to:
- Frontend: http://localhost:80
- Airflow: http://localhost:8080 (username: airflow, password: airflow)
- Grafana: http://localhost:3000 (username: admin, password: admin or as specified in .env)
- MLFlow: http://localhost:5000

### Viewing Predictions

1. Navigate to the main frontend page
2. Select a city from the dropdown menu
3. View the current prediction and historical data

## API Reference

The backend API provides the following endpoints:

- `GET /`: Health check endpoint
- `GET /cities`: Returns a list of all available cities
- `GET /predictions/{city_name}`: Returns the latest prediction for a specific city
- `GET /history/{city_name}?days=7`: Returns historical predictions and actual values for a city

## ML Pipeline

The ML pipeline is managed with DVC and consists of the following stages:

1. **Data Collection**: Fetches weather data from OpenWeatherMap API
   ```bash
   dvc repro data_collection
   ```

2. **Prediction**: Generates weather predictions using trained models
   ```bash
   dvc repro predict
   ```

3. **Drift Detection**: Monitors for data drift between predictions and actual values
   ```bash
   dvc repro drift_detection
   ```

4. **Model Retraining**: Retrains models when drift is detected
   ```bash
   dvc repro model_retraining
   ```

5. **Model Evaluation**: Evaluates model performance
   ```bash
   dvc repro model_evaluation
   ```

6. **Weekly Monitoring**: Runs the entire monitoring pipeline
   ```bash
   dvc repro weekly_monitoring
   ```

### Running the Full Pipeline

To run the entire pipeline:
```bash
dvc repro
```

## Monitoring

### Prometheus

Prometheus collects metrics from the backend API and node exporter. Access the Prometheus UI at http://localhost:9090.

### Grafana

Grafana provides dashboards for visualizing system and model metrics. Access Grafana at http://localhost:3000.

## Running MLflow

To start the MLflow tracking server, use the following Python command:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

This command makes the MLflow UI accessible at `http://localhost:5000`. Ensure that the `mlruns` directory is in the same location as your workspace or specify the `--backend-store-uri` and `--default-artifact-root` options if needed.

## Troubleshooting

### Common Issues

1. **Services not starting properly**:
   - Check Docker logs: `docker-compose logs [service_name]`
   - Ensure all required ports are available

2. **No prediction data available**:
   - Verify the data collection pipeline has run: `dvc repro data_collection`
   - Check API key is set correctly in .env file

3. **Airflow tasks failing**:
   - Check Airflow logs in the Airflow UI
   - Verify dependencies are installed correctly

## Advanced Configuration

### Customizing Cities

Edit the `config/cities.yaml` file to add or remove cities for predictions.

### Model Configuration

Adjust model parameters in `config/model.yaml`.

### Pipeline Configuration

Modify pipeline settings in `config/pipeline.yaml`.

### Logging Configuration

Configure logging behavior in `config/logging.yaml`.

### Scaling the System

For higher load deployments:
1. Add more Airflow workers
2. Scale the backend service horizontally
3. Configure a load balancer for API endpoints