from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta
import random

from app.database import get_db
from app import models, schemas
from config import settings

router = APIRouter()

@router.post("/", response_model=schemas.Container)
def create_container(container: schemas.ContainerCreate, db: Session = Depends(get_db)):
    """Crear nuevo contenedor"""
    # CORREGIDO: usar model_dump() correctamente
    db_container = models.Container(**container.model_dump())
    db.add(db_container)
    db.commit()
    db.refresh(db_container)
    return db_container

@router.get("/", response_model=List[schemas.Container])
def get_containers(
    skip: int = 0,
    limit: int = 1000,
    zone: Optional[str] = None,
    min_fill: Optional[float] = None,
    active_only: bool = False,
    db: Session = Depends(get_db)
):
    """Obtener lista de contenedores con filtros"""
    try:
        query = db.query(models.Container)
        
        # Aplicar filtros solo si se especifican
        if active_only:
            query = query.filter(models.Container.is_active == True)
        
        if zone:
            query = query.filter(models.Container.zone == zone)
        
        if min_fill is not None:
            query = query.filter(models.Container.fill_percentage >= min_fill)
        
        # DEBUGGING: Agregar logs para verificar
        total_count = query.count()
        print(f"Total containers matching filters: {total_count}")
        
        containers = query.offset(skip).limit(limit).all()
        print(f"Containers returned: {len(containers)}")
        
        return containers
        
    except Exception as e:
        print(f"Error en get_containers: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving containers: {str(e)}")

@router.get("/debug/count")
def get_containers_count(db: Session = Depends(get_db)):
    """Endpoint de debug para contar contenedores"""
    try:
        total = db.query(models.Container).count()
        active = db.query(models.Container).filter(models.Container.is_active == True).count()
        inactive = db.query(models.Container).filter(models.Container.is_active == False).count()
        
        # Verificar datos específicos
        sample_containers = db.query(models.Container).limit(5).all()
        sample_data = []
        for c in sample_containers:
            sample_data.append({
                "id": c.id,
                "container_id": c.container_id,
                "is_active": c.is_active,
                "fill_percentage": c.fill_percentage,
                "zone": c.zone
            })
        
        return {
            "total_containers": total,
            "active_containers": active,
            "inactive_containers": inactive,
            "sample_data": sample_data
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/{container_id}", response_model=schemas.Container)
def get_container(container_id: int, db: Session = Depends(get_db)):
    """Obtener contenedor específico"""
    container = db.query(models.Container).filter(models.Container.id == container_id).first()
    
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    return container

@router.put("/{container_id}", response_model=schemas.Container)
def update_container(
    container_id: int,
    container_update: schemas.ContainerUpdate,
    db: Session = Depends(get_db)
):
    """Actualizar datos del contenedor"""
    container = db.query(models.Container).filter(models.Container.id == container_id).first()
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    try:
        # CORREGIDO: usar model_dump() en lugar de dict()
        update_data = container_update.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            if value is not None:
                setattr(container, field, value)
        
        # Actualizar timestamp
        container.last_update = datetime.utcnow()
        
        # Recalcular current_level basado en fill_percentage
        if container.fill_percentage is not None and container.capacity:
            container.current_level = (container.fill_percentage / 100) * container.capacity
        
        db.commit()
        db.refresh(container)
        return container
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating container: {str(e)}")

@router.get("/{container_id}/history")
def get_container_history(
    container_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Obtener historial de lecturas del contenedor"""
    container = db.query(models.Container).filter(models.Container.id == container_id).first()
    
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    since_date = datetime.utcnow() - timedelta(days=days)
    
    readings = db.query(models.SensorReading).filter(
        models.SensorReading.container_id == container_id,
        models.SensorReading.timestamp >= since_date
    ).order_by(models.SensorReading.timestamp.desc()).all()
    
    return {
        "container_id": container_id,
        "container_name": container.container_id,
        "readings_count": len(readings),
        "readings": readings[:100]
    }

@router.post("/simulate-data")
def simulate_container_data(db: Session = Depends(get_db)):
    """Simular datos IoT"""
    try:
        containers = db.query(models.Container).filter(models.Container.is_active == True).all()
        
        updated_count = 0
        for container in containers:
            # Simular incremento gradual del llenado
            current_fill = float(container.fill_percentage or 0)
            fill_increment = random.uniform(0.5, 3.0)  # Incremento más realista
            new_fill = min(100.0, current_fill + fill_increment)
            
            container.fill_percentage = new_fill
            container.current_level = (new_fill / 100) * float(container.capacity)
            container.temperature = random.uniform(15, 30)
            
            # Simular descarga lenta de batería
            current_battery = float(container.battery_level or 100)
            battery_drain = random.uniform(0.01, 0.05)
            container.battery_level = max(10.0, current_battery - battery_drain)
            
            container.last_update = datetime.utcnow()
            updated_count += 1
        
        db.commit()
        return {
            "message": "Data simulation completed successfully",
            "containers_updated": updated_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error simulating data: {str(e)}")

@router.get("/{container_id}/alerts")
def get_container_alerts(container_id: int, db: Session = Depends(get_db)):
    """Obtener alertas del contenedor"""
    container = db.query(models.Container).filter(models.Container.id == container_id).first()
    
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    alerts = []
    
    # Alerta por llenado crítico
    if container.fill_percentage and container.fill_percentage > 90:
        alerts.append({
            "type": "CRITICAL",
            "message": "Container almost full - immediate collection required",
            "value": container.fill_percentage,
            "threshold": 90,
            "priority": "HIGH"
        })
    elif container.fill_percentage and container.fill_percentage > 75:
        alerts.append({
            "type": "WARNING", 
            "message": "Container filling up - schedule collection soon",
            "value": container.fill_percentage,
            "threshold": 75,
            "priority": "MEDIUM"
        })
    
    # Alerta por batería baja
    if container.battery_level and container.battery_level < 20:
        alerts.append({
            "type": "WARNING",
            "message": "Low battery - maintenance required",
            "value": container.battery_level,
            "threshold": 20,
            "priority": "MEDIUM"
        })
    elif container.battery_level and container.battery_level < 10:
        alerts.append({
            "type": "CRITICAL",
            "message": "Critical battery level - immediate maintenance required",
            "value": container.battery_level,
            "threshold": 10,
            "priority": "HIGH"
        })
    
    # Alerta por temperatura alta
    if container.temperature and container.temperature > 35:
        alerts.append({
            "type": "WARNING",
            "message": "High temperature detected",
            "value": container.temperature,
            "threshold": 35,
            "priority": "LOW"
        })
    
    return {
        "container_id": container_id,
        "container_name": container.container_id,
        "alerts_count": len(alerts),
        "alerts": alerts,
        "last_check": datetime.utcnow().isoformat()
    }

@router.post("/bulk-create")
def create_multiple_containers(
    containers_data: List[schemas.ContainerCreate],
    db: Session = Depends(get_db)
):
    """Crear múltiples contenedores de una vez"""
    try:
        created_containers = []
        
        for container_data in containers_data:
            db_container = models.Container(**container_data.model_dump())
            db.add(db_container)
            created_containers.append(db_container)
        
        db.commit()
        
        # Refresh all containers
        for container in created_containers:
            db.refresh(container)
        
        return {
            "message": f"Successfully created {len(created_containers)} containers",
            "containers": created_containers
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating containers: {str(e)}")

@router.delete("/{container_id}")
def delete_container(container_id: int, db: Session = Depends(get_db)):
    """Eliminar contenedor (soft delete)"""
    container = db.query(models.Container).filter(models.Container.id == container_id).first()
    
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    # Soft delete - marcar como inactivo
    container.is_active = False
    db.commit()
    
    return {"message": f"Container {container.container_id} marked as inactive"}