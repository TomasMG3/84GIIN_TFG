# physics_routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List
from app import physics_simulator 
import json

from app.database import get_db
from app import models

router = APIRouter()

@router.post("/simulate-route-physics")
async def simulate_route_physics(
    route_id: int,
    db: Session = Depends(get_db)
):
    """Simular la física realista de una ruta"""
    
    # Obtener ruta de la base de datos
    route = db.query(models.Route).filter(models.Route.id == route_id).first()
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    # Obtener contenedores de la ruta
    route_containers = json.loads(route.container_ids) if route.container_ids else []
    containers = db.query(models.Container).filter(
        models.Container.id.in_(route_containers)
    ).all()
    
    # Convertir a formato para simulación
    containers_data = []
    for container in containers:
        containers_data.append({
            'id': container.id,
            'capacity': container.capacity,
            'current_level': container.current_level,
            'fill_percentage': container.fill_percentage,
            'container_type': getattr(container, 'container_type', 'medium')
        })
    
    # Datos de la ruta
    route_data = {
        'total_distance_km': route.total_distance_km or 0,
        'containers': [{'container_id': c.id} for c in containers]
    }
    
    # Ejecutar simulación física
    physics_metrics = physics_simulator.generate_physical_metrics(route_data, containers_data)
    
    return {
        'route_id': route_id,
        'simulation_results': physics_metrics,
        'simulation_timestamp': datetime.now().isoformat()
    }

@router.get("/fuel-calculator")
async def calculate_fuel_requirements(
    distance_km: float,
    expected_load_kg: float = 0
):
    """Calculadora de combustible para rutas planificadas"""
    
    fuel_required = physics_simulator.calculate_fuel_consumption(distance_km, expected_load_kg)
    trips_possible = physics_simulator.truck_fuel_capacity / fuel_required if fuel_required > 0 else 0
    
    return {
        'distance_km': distance_km,
        'expected_load_kg': expected_load_kg,
        'fuel_required_liters': round(fuel_required, 2),
        'trips_possible': int(trips_possible),
        'fuel_remaining_after_trip': max(0, physics_simulator.truck_fuel_capacity - fuel_required),
        'co2_emissions_kg': round(fuel_required * physics_simulator.co2_per_liter_diesel, 2)
    }