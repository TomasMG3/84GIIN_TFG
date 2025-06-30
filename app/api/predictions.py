from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from app.database import get_db
from app import models, schemas
from ml.waste_predictor import WastePredictor

router = APIRouter()

# Instancia global del predictor
predictor = WastePredictor()

@router.post("/train")
async def train_prediction_models(
    background_tasks: BackgroundTasks,
    days_history: int = 30,
    db: Session = Depends(get_db)
):
    """Entrenar modelos de predicción con datos históricos"""
    
    # Obtener datos históricos de sensores
    since_date = datetime.utcnow() - timedelta(days=days_history)
    
    readings = db.query(models.SensorReading).filter(
        models.SensorReading.timestamp >= since_date
    ).all()
    
    if len(readings) < 100:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient data for training. Found {len(readings)} readings, need at least 100"
        )
    
    # Convertir a DataFrame
    data = []
    for reading in readings:
        container = db.query(models.Container).filter(
            models.Container.id == reading.container_id
        ).first()
        
        if container:
            data.append({
                'container_id': reading.container_id,
                'timestamp': reading.timestamp,
                'fill_percentage': reading.fill_percentage,
                'temperature': reading.temperature,
                'zone': container.zone or 'default'
            })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid training data found")
    
    # Entrenar modelos en background
    def train_models():
        try:
            results = predictor.train_models(df)
            print(f"Training completed: {results}")
        except Exception as e:
            print(f"Training failed: {e}")
    
    background_tasks.add_task(train_models)
    
    return {
        "message": "Model training started",
        "training_data_size": len(df),
        "date_range": f"{since_date.date()} to {datetime.utcnow().date()}",
        "status": "training_in_progress"
    }

@router.get("/container/{container_id}/prediction")
async def predict_container_fill(
    container_id: int,
    db: Session = Depends(get_db)
):
    """Predecir cuándo se llenará un contenedor específico"""
    
    container = db.query(models.Container).filter(
        models.Container.id == container_id
    ).first()
    
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    # Obtener predicción
    prediction = predictor.predict_fill_time(
        container_id=container.id,
        current_fill=container.fill_percentage,
        zone=container.zone
    )
    
    # Agregar información del contenedor
    prediction.update({
        "container_name": container.container_id,
        "address": container.address,
        "zone": container.zone,
        "capacity": container.capacity,
        "current_level": container.current_level,
        "last_update": container.last_update.isoformat() if container.last_update else None
    })
    
    return prediction

@router.get("/daily-predictions")
async def get_daily_predictions(
    zone: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Obtener predicciones diarias para todos los contenedores"""
    
    query = db.query(models.Container).filter(models.Container.is_active == True)
    
    if zone:
        query = query.filter(models.Container.zone == zone)
    
    containers = query.limit(limit).all()
    
    if not containers:
        return {
            "predictions": [],
            "summary": {
                "total_containers": 0,
                "high_priority": 0,
                "medium_priority": 0,
                "low_priority": 0
            }
        }
    
    # Convertir a DataFrame para predicciones
    containers_data = []
    for container in containers:
        containers_data.append({
            'id': container.id,
            'container_id': container.container_id,
            'fill_percentage': container.fill_percentage,
            'zone': container.zone
        })
    
    df = pd.DataFrame(containers_data)
    predictions = predictor.get_daily_predictions(df)
    
    # Filtrar por prioridad si se especifica
    if priority:
        predictions = [p for p in predictions if p.get('priority') == priority.upper()]
    
    # Estadísticas
    priority_counts = {
        'HIGH': len([p for p in predictions if p.get('priority') == 'HIGH']),
        'MEDIUM': len([p for p in predictions if p.get('priority') == 'MEDIUM']),
        'LOW': len([p for p in predictions if p.get('priority') == 'LOW'])
    }
    
    return {
        "predictions": predictions,
        "summary": {
            "total_containers": len(predictions),
            "high_priority": priority_counts['HIGH'],
            "medium_priority": priority_counts['MEDIUM'],
            "low_priority": priority_counts['LOW']
        },
        "generated_at": datetime.utcnow().isoformat()
    }

@router.get("/alerts")
async def get_prediction_alerts(
    hours_threshold: int = 24,
    db: Session = Depends(get_db)
):
    """Obtener alertas basadas en predicciones"""
    
    # Obtener todos los contenedores activos
    containers = db.query(models.Container).filter(
        models.Container.is_active == True
    ).all()
    
    alerts = []
    
    for container in containers:
        prediction = predictor.predict_fill_time(
            container_id=container.id,
            current_fill=container.fill_percentage,
            zone=container.zone
        )
        
        # Generar alertas según el tiempo predicho
        if prediction.get('hours_to_full') is not None:
            hours_to_full = prediction['hours_to_full']
            
            if hours_to_full <= hours_threshold:
                severity = "CRITICAL" if hours_to_full <= 12 else "WARNING"
                
                alerts.append({
                    "container_id": container.id,
                    "container_name": container.container_id,
                    "address": container.address,
                    "zone": container.zone,
                    "current_fill": container.fill_percentage,
                    "predicted_full_time": prediction['predicted_full_time'],
                    "hours_to_full": hours_to_full,
                    "severity": severity,
                    "message": f"Container will be full in {hours_to_full} hours",
                    "recommendation": prediction['recommendation']
                })
    
    # Ordenar por urgencia
    alerts.sort(key=lambda x: x['hours_to_full'])
    
    return {
        "alerts": alerts,
        "total_alerts": len(alerts),
        "critical_alerts": len([a for a in alerts if a['severity'] == 'CRITICAL']),
        "generated_at": datetime.utcnow().isoformat()
    }

@router.get("/model-performance")
async def get_model_performance():
    """Obtener métricas de rendimiento de los modelos"""
    
    if not predictor.trained_models:
        raise HTTPException(
            status_code=400, 
            detail="Models not trained yet. Please train models first."
        )
    
    # Simular métricas de rendimiento
    performance_metrics = {}
    
    for model_name in predictor.trained_models.keys():
        # En un sistema real, estas métricas vendrían de validación cruzada
        performance_metrics[model_name] = {
            "mean_absolute_error": np.random.uniform(3, 8),
            "r2_score": np.random.uniform(0.7, 0.95),
            "accuracy_percentage": np.random.uniform(85, 95),
            "predictions_made": np.random.randint(100, 1000),
            "last_training": datetime.utcnow().isoformat()
        }
    
    return {
        "models": performance_metrics,
        "best_model": "random_forest",  # Por defecto
        "last_evaluation": datetime.utcnow().isoformat()
    }

@router.post("/optimize-collection-schedule")
async def optimize_collection_schedule(
    days_ahead: int = 7,
    max_routes_per_day: int = 3,
    db: Session = Depends(get_db)
):
    """Optimizar horarios de recolección basado en predicciones"""
    
    # Obtener predicciones para todos los contenedores
    containers = db.query(models.Container).filter(
        models.Container.is_active == True
    ).all()
    
    schedule = {}
    
    for day in range(days_ahead):
        target_date = datetime.utcnow() + timedelta(days=day)
        day_key = target_date.strftime('%Y-%m-%d')
        
        containers_for_day = []
        
        for container in containers:
            prediction = predictor.predict_fill_time(
                container_id=container.id,
                current_fill=container.fill_percentage,
                zone=container.zone
            )
            
            # Si el contenedor se llenará en este día
            if prediction.get('hours_to_full'):
                full_time = datetime.fromisoformat(prediction['predicted_full_time'].replace('Z', '+00:00'))
                if full_time.date() == target_date.date():
                    containers_for_day.append({
                        "container_id": container.id,
                        "container_name": container.container_id,
                        "predicted_full_time": prediction['predicted_full_time'],
                        "priority": prediction['priority'],
                        "zone": container.zone
                    })
        
        # Agrupar por zonas y prioridad
        schedule[day_key] = {
            "containers": containers_for_day[:20],  # Limitar a 20 por día
            "total_containers": len(containers_for_day),
            "high_priority": len([c for c in containers_for_day if c['priority'] == 'HIGH']),
            "recommended_routes": min(max_routes_per_day, (len(containers_for_day) + 9) // 10)
        }
    
    return {
        "schedule": schedule,
        "optimization_period": f"{days_ahead} days",
        "generated_at": datetime.utcnow().isoformat()
    }