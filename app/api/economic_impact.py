from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app import models

router = APIRouter()

@router.get("/savings")
def calculate_savings(db: Session = Depends(get_db)):
    """Calcula ahorros económicos y reducción de CO2"""
    # Obtener métricas operativas
    total_routes = db.query(models.Route).count()
    if total_routes == 0:
        return {"message": "No hay rutas registradas para calcular ahorros."}
    
    avg_distance = db.query(func.avg(models.Route.total_distance_km)).scalar() or 0
    avg_fuel = db.query(func.avg(models.Route.fuel_consumption_liters)).scalar() or 0
    
    # Parámetros económicos (personalizables)
    fuel_price = 1.25  # USD/L
    driver_cost = 25  # USD/hora
    maintenance_saving_per_km = 0.02  # USD/km
    co2_price = 0.05  # USD/kg
    
    # Cálculo de ahorros
    fuel_savings = avg_fuel * total_routes * 0.18 * fuel_price
    time_savings = (avg_distance / 25 * 60) * total_routes * 0.21 * driver_cost
    maintenance_savings = avg_distance * total_routes * maintenance_saving_per_km
    co2_reduction = avg_fuel * 2.6 * total_routes * 0.19
    
    return {
        "projected_annual_savings": {
            "fuel": round(fuel_savings * 365, 2),
            "labor": round(time_savings * 365, 2),
            "maintenance": round(maintenance_savings * 365, 2),
            "total": round((fuel_savings + time_savings + maintenance_savings) * 365, 2)
        },
        "environmental_impact": {
            "co2_reduction_tons": round(co2_reduction * 365 / 1000, 1),
            "co2_cost_savings": round(co2_reduction * 365 * co2_price, 2)
        },
        "assumptions": {
            "fuel_price_usd_per_l": fuel_price,
            "driver_cost_usd_per_hour": driver_cost,
            "maintenance_saving_per_km": maintenance_saving_per_km,
            "co2_price_usd_per_kg": co2_price
        }
    }
    
# En economic_impact.py, ampliar cálculos
@router.get("/detailed-savings")
def calculate_detailed_savings(db: Session = Depends(get_db)):
    routes = db.query(models.Route).count()
    optimized_routes = db.query(models.Route).filter(models.Route.is_optimized == True).count()
    
    avg_distance = db.query(func.avg(models.Route.total_distance_km)).scalar() or 0
    avg_fuel = db.query(func.avg(models.Route.fuel_consumption_liters)).scalar() or 0
    
    # Cálculos detallados
    savings = {
        "annual_fuel_savings": avg_fuel * optimized_routes * 365 * 0.18 * 1.25,
        "labor_savings": (avg_distance / 25 * 60) * optimized_routes * 365 * 0.21 * 25,
        "maintenance_savings": avg_distance * optimized_routes * 365 * 0.02,
        "co2_reduction": avg_fuel * 2.6 * optimized_routes * 365 * 0.19 / 1000
    }
    
    return savings