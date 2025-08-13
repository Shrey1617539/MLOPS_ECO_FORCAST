# backend/src/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import FastAPI
from src.utils.logging_utils import get_logger
from datetime import datetime

logger = get_logger(__name__)

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total number of HTTP requests', 
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 
    'HTTP request latency in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'prediction_requests_total', 
    'Total number of prediction requests',
    ['city']
)

PREDICTION_ERROR = Counter(
    'prediction_errors_total', 
    'Total number of prediction errors',
    ['city', 'error_type']
)

MODEL_VERSION = Info(
    'model_info', 
    'Information about the trained models'
)

DATA_PROCESSING_TIME = Histogram(
    'data_processing_duration_seconds',
    'Data processing duration in seconds',
    ['operation']
)

DRIFT_GAUGE = Gauge(
    'model_drift_score',
    'Model drift score (higher means more drift)',
    ['city', 'metric']
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process the request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e 
        finally:
            # Record request metrics
            endpoint = request.url.path
            method = request.method
            
            REQUEST_COUNT.labels(
                method=method, 
                endpoint=endpoint, 
                status_code=status_code
            ).inc()
            
            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint
            ).observe(time.time() - start_time)
        
        return response

def initialize_metrics(app: FastAPI):
    """Initialize Prometheus metrics for the FastAPI app."""
    app.add_middleware(PrometheusMiddleware)
    
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type="text/plain")
    
    @app.get("/health", include_in_schema=False)
    async def health():
        return {"status": "healthy"}
    
    # Set initial model info
    try:
        import os
        import json
        from pathlib import Path
        
        # Get model versions from DVC
        models_dir = Path("/app/models/trained")
        if models_dir.exists():
            model_files = list(models_dir.glob("*_model.h5"))
            model_info = {}
            
            for model_file in model_files:
                city_name = model_file.stem.replace("_model", "")
                
                # Try to get the timestamp from the file modification time
                timestamp = datetime.fromtimestamp(
                    os.path.getmtime(model_file)
                ).isoformat()
                
                model_info[city_name] = {
                    "file": str(model_file),
                    "updated_at": timestamp
                }
            
            MODEL_VERSION.info(model_info)
    except Exception as e:
        logger.error(f"Error initializing model info metrics: {str(e)}")
