from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import json
import os


from src.utils.metrics import initialize_metrics
from src.utils.config import get_config
from src.models.predict import load_latest_predictions
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Weather Prediction API",
    description="API for weather predictions and monitoring",
    version="1.0.0"
)

initialize_metrics(app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Weather Prediction API is running"}

# Get all available cities
@app.get("/cities", response_model=List[Dict[str, Any]])
async def get_cities():
    """Get all cities configured in the system"""
    try:
        cities = get_config("cities", "cities")
        return cities
    except Exception as e:
        logger.error(f"Error retrieving cities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Get latest prediction for a specific city
@app.get("/predictions/{city_name}")
async def get_prediction(city_name: str):
    """Get the latest prediction for a specific city"""
    try:
        # Normalize city name
        normalized_city = city_name.lower().replace(' ', '_')
        
        # Load latest prediction
        prediction = load_latest_predictions(normalized_city)
        
        if not prediction:
            raise HTTPException(status_code=404, detail=f"No prediction found for {city_name}")
        
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prediction for {city_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Get historical predictions (and actuals when available) for a specific city
@app.get("/history/{city_name}")
async def get_history(city_name: str, days: int = 7):
    """
    Get historical predictions and actual values for a specific city
    
    Args:
        city_name: Name of the city
        days: Number of days of history to retrieve (default: 7)
    """
    try:
        # Normalize city name
        normalized_city = city_name.lower().replace(' ', '_')
        
        # Load predictions
        predictions_path = Path(f"/app/data/predictions/{normalized_city}.csv")
        if not predictions_path.exists():
            raise HTTPException(status_code=404, detail=f"No prediction data found for {city_name}")
        
        pred_df = pd.read_csv(predictions_path)
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        
        # Filter for last N days
        cutoff_date = datetime.now() - timedelta(days=days)
        pred_df = pred_df[pred_df['date'] >= cutoff_date].copy()
        
        # Load actual data
        actuals_path = Path(f"/app/data/raw/{normalized_city}.csv")
        if actuals_path.exists():
            actual_df = pd.read_csv(actuals_path)
            actual_df['date'] = pd.to_datetime(actual_df['date'])
            
            # Filter for last N days
            actual_df = actual_df[actual_df['date'] >= cutoff_date].copy()
            
            # Merge predictions with actuals
            model_config = get_config('model')
            target = model_config.get('target', 'temperature')
            
            merged_df = pd.merge(
                pred_df[['date', f'predicted_{target}']],
                actual_df[['date', target]],
                on='date',
                how='outer'
            )
            
            # Convert to records format for API response
            merged_df['date'] = merged_df['date'].dt.strftime('%Y-%m-%d')
            history = merged_df.to_dict(orient='records')
        else:
            # Only return predictions if no actuals available
            pred_df['date'] = pred_df['date'].dt.strftime('%Y-%m-%d')
            history = pred_df.to_dict(orient='records')
        
        return {
            "city": city_name,
            "history": history
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving history for {city_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Get pipeline status and metadata
# @app.get("/pipeline/status")
# async def get_pipeline_status():
#     """Get the status of the ML pipeline and metadata about runs"""
#     try:
#         # Look for the latest pipeline report
#         reports_dir = Path("/app/models/evaluation/pipeline_reports")
#         if not reports_dir.exists():
#             return {
#                 "status": "unknown",
#                 "message": "No pipeline reports found"
#             }
        
#         # Get the most recent pipeline report
#         report_files = list(reports_dir.glob("drift_pipeline_report_*.json"))
#         if not report_files:
#             return {
#                 "status": "unknown",
#                 "message": "No pipeline reports found"
#             }
        
#         latest_report = max(report_files, key=os.path.getctime)
        
#         with open(latest_report, 'r') as f:
#             report_data = json.load(f)
        
#         return {
#             "status": "active",
#             "last_run": report_data.get("timestamp"),
#             "drift_detected": report_data.get("drift_detected", False),
#             "details": report_data
#         }
#     except Exception as e:
#         logger.error(f"Error retrieving pipeline status: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# # Endpoint for model performance metrics
# @app.get("/metrics/{city_name}")
# async def get_metrics(city_name: str):
#     """Get model performance metrics for a specific city"""
#     try:
#         # Normalize city name
#         normalized_city = city_name.lower().replace(' ', '_')
        
#         # Find the latest evaluation file
#         eval_dir = Path("/app/models/evaluation")
#         if not eval_dir.exists():
#             raise HTTPException(status_code=404, detail="No evaluation data found")
        
#         eval_files = list(eval_dir.glob(f"{normalized_city}_evaluation_*.json"))
#         if not eval_files:
#             raise HTTPException(status_code=404, detail=f"No evaluation data found for {city_name}")
        
#         latest_eval = max(eval_files, key=os.path.getctime)
        
#         with open(latest_eval, 'r') as f:
#             eval_data = json.load(f)
        
#         return eval_data
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error retrieving metrics for {city_name}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
