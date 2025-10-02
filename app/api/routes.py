# routes.py - Versión actualizada con OpenRouteService

from ml.route_optimizer import RouteOptimizer, Container, Vehicle
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session
from sqlalchemy import func 
from typing import List
from datetime import datetime
import math
import requests
import os

from app.database import get_db
from app import models, schemas

router = APIRouter()

# Simple container class for optimization
class OptimizationContainer:
    def __init__(self, id: int, lat: float, lon: float, fill_percentage: float, priority: float = 1.0):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.fill_percentage = fill_percentage
        self.priority = priority

def check_openrouteservice_available():
    """Verificar si OpenRouteService está disponible"""
    try:
        openroute_api_key = os.getenv('OPENROUTESERVICE_API_KEY')
        if not openroute_api_key:
            print("OPENROUTESERVICE_API_KEY not configured.")
            return False
        
        # Hacer una prueba simple con OpenRouteService Matrix API
        url = "https://api.openrouteservice.org/v2/matrix/driving-car"
        headers = {
            'Accept': 'application/json',
            'Authorization': openroute_api_key,
            'Content-Type': 'application/json; charset=utf-8'
        }
        body = {
            "locations": [[-70.5241, -33.4119], [-70.5200, -33.4100]],
            "metrics": ["distance"]
        }
        
        response = requests.post(url, json=body, headers=headers, timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("OpenRouteService connection error.")
        return False
    except requests.exceptions.Timeout:
        print("OpenRouteService connection timed out.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred checking OpenRouteService: {e}")
        return False

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def openrouteservice_optimized_route(containers: List[OptimizationContainer]) -> dict:
    """Optimización de ruta usando OpenRouteService"""
    if not containers:
        return {
            "routes": [],
            "total_distance": 0.0,
            "containers_count": 0,
            "optimization_method": "openrouteservice"
        }
    
    try:
        # Crear optimizador de OpenRouteService
        optimizer = RouteOptimizer()
        
        # Convertir containers a formato del optimizador
        ors_containers = []
        for container in containers:
            ors_containers.append(Container(
                id=container.id,
                lat=container.lat,
                lon=container.lon,
                fill_percentage=container.fill_percentage,
                priority=container.priority
            ))
        
        # Ejecutar optimización
        result = optimizer.optimize_route(ors_containers)
        
        return result
        
    except Exception as e:
        print(f"Error en optimización con OpenRouteService: {e}")
        # Fallback a cálculo haversine
        return simple_route_optimization(containers)

def simple_route_optimization(containers: List[OptimizationContainer]) -> dict:
    """Simple nearest neighbor route optimization usando distancias Haversine"""
    if not containers:
        return {
            "routes": [],
            "total_distance": 0.0,
            "containers_count": 0,
            "optimization_method": "nearest_neighbor_haversine"
        }
    
    # Start from depot (Las Condes centro)
    depot_lat, depot_lon = -33.4119, -70.5241
    
    unvisited = containers.copy()
    route = []
    total_distance = 0.0
    current_lat, current_lon = depot_lat, depot_lon
    
    # Nearest neighbor algorithm
    while unvisited:
        nearest_container = None
        min_distance = float('inf')
        
        for container in unvisited:
            distance = calculate_distance(current_lat, current_lon, container.lat, container.lon)
            # Apply priority weighting (higher priority = shorter perceived distance)
            weighted_distance = distance / container.priority
            
            if weighted_distance < min_distance:
                min_distance = weighted_distance
                nearest_container = container
        
        if nearest_container:
            actual_distance = calculate_distance(current_lat, current_lon, nearest_container.lat, nearest_container.lon)
            total_distance += actual_distance
            route.append({
                "container_id": nearest_container.id,
                "latitude": nearest_container.lat,
                "longitude": nearest_container.lon,
                "fill_percentage": nearest_container.fill_percentage,
                "distance_from_previous": round(actual_distance, 2),
                "order": len(route) + 1
            })
            
            current_lat, current_lon = nearest_container.lat, nearest_container.lon
            unvisited.remove(nearest_container)
    
    # Return to depot
    if route:
        return_distance = calculate_distance(current_lat, current_lon, depot_lat, depot_lon)
        total_distance += return_distance
    
    # Calculate estimates
    estimated_time = int(total_distance * 2.5 + len(route) * 5)  # 2.5 min/km + 5 min per stop
    fuel_consumption = total_distance * 0.35  # 0.35 L/km
    co2_emissions = fuel_consumption * 2.6  # 2.6 kg CO2/L
    
    return {
        "routes": [{
            "route_id": 1,
            "containers": route,
            "total_distance_km": round(total_distance, 2),
            "estimated_time_minutes": estimated_time,
            "fuel_consumption_liters": round(fuel_consumption, 2),
            "co2_emissions_kg": round(co2_emissions, 2),
            "depot_location": {"lat": depot_lat, "lon": depot_lon}
        }],
        "total_distance": round(total_distance, 2),
        "containers_count": len(containers),
        "optimization_method": "nearest_neighbor_haversine"
    }

def genetic_algorithm_optimization(containers: List[OptimizationContainer], vehicles: List[Vehicle]) -> dict:
    """Algoritmo genético avanzado con OpenRouteService"""
    try:
        optimizer = RouteOptimizer()
        
        # Convertir containers a formato del optimizador
        ors_containers = []
        for container in containers:
            ors_containers.append(Container(
                id=container.id,
                lat=container.lat,
                lon=container.lon,
                fill_percentage=container.fill_percentage,
                priority=container.priority
            ))
        
        # Ejecutar algoritmo genético
        result = optimizer.genetic_algorithm_vrp(
            containers=ors_containers,
            vehicles=vehicles,
            population_size=30,
            generations=50
        )
        
        return result
        
    except Exception as e:
        print(f"Error en algoritmo genético: {e}")
        # Fallback a optimización básica con OpenRouteService
        return openrouteservice_optimized_route(containers)

@router.post("/optimize", response_model=dict)
async def optimize_routes(
    background_tasks: BackgroundTasks,
    min_fill_threshold: float = 70.0,
    use_genetic_algorithm: bool = True,
    include_nearby_overflow: bool = True,
    db: Session = Depends(get_db)
):
    """
    Optimizar rutas de recolección con priorización de desbordamientos
    
    Args:
        min_fill_threshold: Umbral mínimo de llenado para incluir contenedor (default 70%)
        use_genetic_algorithm: Si usar algoritmo genético avanzado
        include_nearby_overflow: Si incluir contenedores con desbordamiento cercanos
    """
    # Obtener TODOS los contenedores activos
    all_containers_query = db.query(models.Container).filter(
        models.Container.is_active == True
    )
    
    db_containers = all_containers_query.all()
    
    if not db_containers:
        return {
            "message": "No hay contenedores activos en el sistema",
            "routes": [],
            "total_distance": 0.0,
            "containers_count": 0
        }
    
    # Convertir a objetos del optimizador
    containers_for_opt = []
    for db_container in db_containers:
        if not db_container.latitude or not db_container.longitude:
            print(f"Container {db_container.id} missing coordinates, skipping.")
            continue
            
        # Crear contenedor con todos los datos necesarios
        container = Container(
            id=db_container.id,
            lat=db_container.latitude,
            lon=db_container.longitude,
            fill_percentage=db_container.fill_percentage,
            priority=1.0,  # Será calculado por el optimizador
            capacity=db_container.capacity if hasattr(db_container, 'capacity') else 1000.0,
            is_overflow=(db_container.fill_percentage > 100)
        )
        
        containers_for_opt.append(container)
    
    if not containers_for_opt:
        return {
            "message": "No hay contenedores con coordenadas válidas",
            "routes": [],
            "total_distance": 0.0,
            "containers_count": 0
        }
    
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO OPTIMIZACIÓN DE RUTAS")
    print(f"{'='*60}")
    print(f"📊 Contenedores totales: {len(containers_for_opt)}")
    print(f"📏 Umbral mínimo: {min_fill_threshold}%")
    print(f"🧬 Algoritmo genético: {'SÍ' if use_genetic_algorithm else 'NO'}")
    print(f"🎯 Incluir desbordamientos cercanos: {'SÍ' if include_nearby_overflow else 'NO'}")
    
    # Estadísticas previas
    total_containers = len(containers_for_opt)
    overflow_count = sum(1 for c in containers_for_opt if c.fill_percentage > 100)
    above_threshold = sum(1 for c in containers_for_opt if c.fill_percentage >= min_fill_threshold)
    below_threshold = total_containers - above_threshold
    
    print(f"\n📈 ANÁLISIS PREVIO:")
    print(f"   • Desbordamientos (>100%): {overflow_count}")
    print(f"   • Sobre umbral (≥{min_fill_threshold}%): {above_threshold}")
    print(f"   • Bajo umbral (<{min_fill_threshold}%): {below_threshold}")
    
    optimization_result = {}
    optimization_method_used = "priority_based_with_overflow"

    try:
        # Verificar disponibilidad de OpenRouteService
        if not check_openrouteservice_available():
            print("\n⚠️ OpenRouteService no está disponible. Usando optimización simple.")
            optimization_result = simple_route_optimization(
                [OptimizationContainer(c.id, c.lat, c.lon, c.fill_percentage, c.priority) 
                 for c in containers_for_opt if c.fill_percentage >= min_fill_threshold]
            )
            optimization_method_used = "nearest_neighbor_fallback"
            optimization_result["message"] = "OpenRouteService no disponible. Se usó optimización simple."
        else:
            # OpenRouteService disponible - usar optimización avanzada
            optimizer = RouteOptimizer()
            
            # Usar el método mejorado con prioridades y desbordamientos
            optimization_result = optimizer.optimize_route_with_priorities(
                containers_for_opt, 
                min_threshold=min_fill_threshold
            )
            
            optimization_method_used = optimization_result.get("optimization_method", "priority_based_with_overflow")
            
            # Mostrar estadísticas de la optimización
            print(f"\n✅ OPTIMIZACIÓN COMPLETADA:")
            print(f"   • Método: {optimization_method_used}")
            print(f"   • Contenedores en ruta: {optimization_result.get('containers_count', 0)}")
            print(f"   • Viajes innecesarios evitados: {optimization_result.get('unnecessary_trips_avoided', 0)}")
            print(f"   • Desbordamientos agregados: {optimization_result.get('overflow_containers_added', 0)}")
            print(f"   • Distancia total: {optimization_result.get('total_distance', 0):.2f} km")
                
    except Exception as e:
        print(f"\n❌ ERROR en optimización: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Fallback a método simple
        optimization_result = simple_route_optimization(
            [OptimizationContainer(c.id, c.lat, c.lon, c.fill_percentage, c.priority) 
             for c in containers_for_opt if c.fill_percentage >= min_fill_threshold]
        )
        optimization_method_used = "nearest_neighbor_fallback"
        optimization_result["message"] = f"Error en optimización: {str(e)}. Se usó optimización simple."

    # Guardar ruta optimizada en BD si se generó una ruta válida
    if optimization_result.get("routes") and len(optimization_result["routes"]) > 0:
        route_data = optimization_result["routes"][0]
        route = models.Route(
            route_name=f"Ruta_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_distance_km=optimization_result["total_distance"],
            estimated_time_minutes=route_data.get("estimated_time_minutes"),
            fuel_consumption_liters=route_data.get("fuel_consumption_liters"),
            co2_emissions_kg=route_data.get("co2_emissions_kg"),
            containers_count=optimization_result["containers_count"],
            optimization_algorithm=optimization_method_used,
            is_optimized=True,
            route_coordinates=route_data.get("route_coordinates")
        )
        
        db.add(route)
        db.commit()
        db.refresh(route)
        
        optimization_result["route_id"] = route.id
        optimization_result["optimization_method"] = optimization_method_used
        
        print(f"\n💾 Ruta guardada en BD con ID: {route.id}")
    else:
        if "message" not in optimization_result:
            optimization_result["message"] = "No se pudo generar una ruta válida."
    
    print(f"{'='*60}\n")
    
    return optimization_result

@router.post("/optimization-metrics")
def calculate_optimization_metrics(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Comparación de algoritmos de optimización"""
    # Ejecutar ambos algoritmos
    genetic_result = genetic_algorithm_optimization(Container, Vehicle)
    nn_result = simple_route_optimization(Container)
    
    return {
        "genetic_algorithm": {
            "distance_km": genetic_result["total_distance"],
            "time_minutes": genetic_result["routes"][0]["estimated_time_minutes"],
            "fuel_liters": genetic_result["routes"][0]["fuel_consumption_liters"]
        },
        "nearest_neighbor": {
            "distance_km": nn_result["total_distance"],
            "time_minutes": nn_result["routes"][0]["estimated_time_minutes"],
            "fuel_liters": nn_result["routes"][0]["fuel_consumption_liters"]
        },
        "improvement_percentage": {
            "distance": round((1 - genetic_result["total_distance"] / nn_result["total_distance"]) * 100, 1),
            "time": round((1 - genetic_result["routes"][0]["estimated_time_minutes"] / nn_result["routes"][0]["estimated_time_minutes"]) * 100, 1),
            "fuel": round((1 - genetic_result["routes"][0]["fuel_consumption_liters"] / nn_result["routes"][0]["fuel_consumption_liters"]) * 100, 1)
        }
    }

@router.get("/", response_model=List[schemas.Route])
def get_routes(
    skip: int = 0,
    limit: int = 50,
    optimized_only: bool = False,
    db: Session = Depends(get_db)
):
    """Obtener rutas guardadas"""
    query = db.query(models.Route)
    
    if optimized_only:
        query = query.filter(models.Route.is_optimized == True)
    
    routes = query.order_by(models.Route.created_at.desc()).offset(skip).limit(limit).all()
    return routes

@router.get("/{route_id}", response_model=schemas.Route)
def get_route(route_id: int, db: Session = Depends(get_db)):
    """Obtener ruta específica"""
    route = db.query(models.Route).filter(models.Route.id == route_id).first()
    
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    return route

@router.get("/{route_id}/containers")
def get_route_containers(route_id: int, db: Session = Depends(get_db)):
    """Obtener contenedores de una ruta específica"""
    route = db.query(models.Route).filter(models.Route.id == route_id).first()
    
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    # Obtener eventos de recolección de esta ruta
    collection_events = db.query(models.CollectionEvent).filter(
        models.CollectionEvent.route_id == route_id
    ).all()
    
    containers_info = []
    for event in collection_events:
        container = db.query(models.Container).filter(
            models.Container.id == event.container_id
        ).first()
        
        if container:
            containers_info.append({
                "container_id": container.id,
                "container_name": container.container_id,
                "latitude": container.latitude,
                "longitude": container.longitude,
                "fill_percentage": container.fill_percentage,
                "collected_at": event.collected_at,
                "volume_collected": event.volume_collected
            })
    
    return {
        "route_id": route_id,
        "route_name": route.route_name,
        "containers": containers_info
    }

@router.post("/{route_id}/execute")
def execute_route(
    route_id: int,
    truck_id: str,
    db: Session = Depends(get_db)
):
    """Simular ejecución de ruta"""
    route = db.query(models.Route).filter(models.Route.id == route_id).first()
    
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    # Verificar que el vehículo existe
    vehicle = db.query(models.Vehicle).filter(models.Vehicle.truck_id == truck_id).first()
    
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    # Simular recolección de contenedores con fill_percentage > 70%
    containers_to_collect = db.query(models.Container).filter(
        models.Container.is_active == True,
        models.Container.fill_percentage >= 70
    ).all()
    
    collections_made = []
    
    for container in containers_to_collect:
        # Simular recolección
        volume_collected = (container.fill_percentage / 100) * container.capacity
        
        # Crear evento de recolección
        collection_event = models.CollectionEvent(
            container_id=container.id,
            route_id=route_id,
            volume_collected=volume_collected,
            collection_time_minutes=5,  # 5 minutos por contenedor
            truck_id=truck_id
        )
        
        # Vaciar contenedor
        container.fill_percentage = 0.0
        container.current_level = 0.0
        container.last_emptied = datetime.utcnow()
        
        db.add(collection_event)
        collections_made.append({
            "container_id": container.id,
            "container_name": container.container_id,
            "volume_collected": round(volume_collected, 2)
        })
    
    db.commit()
    
    return {
        "route_id": route_id,
        "truck_id": truck_id,
        "collections_made": len(collections_made),
        "total_volume_collected": sum(c["volume_collected"] for c in collections_made),
        "collections": collections_made,
        "execution_time": datetime.utcnow().isoformat()
    }

@router.post("/optimize-with-ai")
async def optimize_routes_with_ai(
    background_tasks: BackgroundTasks,
    min_fill_threshold: float = 60.0,
    use_lstm: bool = True,
    use_genetic_algorithm: bool = True,
    db: Session = Depends(get_db)
):
    """Optimización de rutas mejorada con IA (LSTM)"""
    
    # Obtener contenedores que necesitan recolección
    containers_query = db.query(models.Container).filter(
        models.Container.is_active == True,
        models.Container.fill_percentage >= min_fill_threshold
    )
    
    db_containers = containers_query.all()
    
    if not db_containers:
        return {
            "message": f"No hay contenedores que necesiten recolección (umbral: {min_fill_threshold}%)",
            "routes": [],
            "total_distance": 0.0,
            "containers_count": 0
        }
    
    # Obtener datos históricos para LSTM
    since_date = datetime.utcnow() - timedelta(days=7)
    historical_readings = db.query(models.SensorReading).filter(
        models.SensorReading.timestamp >= since_date
    ).all()
    
    # Convertir a DataFrame
    historical_data = []
    for reading in historical_readings:
        historical_data.append({
            'container_id': reading.container_id,
            'timestamp': reading.timestamp,
            'fill_percentage': reading.fill_percentage
        })
    
    historical_df = pd.DataFrame(historical_data)
    
    # Convertir a objetos de optimización
    containers_for_opt = []
    for db_container in db_containers:
        if db_container.latitude and db_container.longitude:
            container_obj = Container(
                id=db_container.id,
                lat=db_container.latitude,
                lon=db_container.longitude,
                fill_percentage=db_container.fill_percentage,
                priority=1.0  # Será actualizado por LSTM
            )
            containers_for_opt.append(container_obj)
    
    # Crear vehículos
    vehicles = [Vehicle(
        id="TRUCK-001",
        capacity_kg=5000.0,
        current_lat=-33.4119,
        current_lon=-70.5241
    )]
    
    # Ejecutar optimización híbrida
    hybrid_optimizer = HybridRouteOptimizer()
    
    if use_lstm and historical_df.empty:
        use_lstm = False
        print("⚠️ No hay datos históricos, usando optimización tradicional")
    
    try:
        if use_lstm:
            optimization_result = hybrid_optimizer.optimize_with_lstm(
                containers=containers_for_opt,
                historical_data=historical_df,
                vehicles=vehicles,
                use_genetic_algorithm=use_genetic_algorithm
            )
            optimization_method = "hybrid_lstm_genetic" if use_genetic_algorithm else "hybrid_lstm_basic"
        else:
            # Fallback a optimización tradicional
            if use_genetic_algorithm:
                optimization_result = hybrid_optimizer.genetic_algorithm_vrp(
                    containers_for_opt, vehicles
                )
                optimization_method = "genetic_algorithm"
            else:
                optimization_result = hybrid_optimizer.optimize_route(containers_for_opt)
                optimization_method = "openrouteservice"
        
        optimization_result["optimization_method"] = optimization_method
        optimization_result["ai_model_used"] = "lstm" if use_lstm else "traditional"
        
        # Guardar resultados en BD
        if optimization_result.get("routes"):
            self._save_optimization_result(optimization_result, db, optimization_method)
        
        return optimization_result
        
    except Exception as e:
        print(f"Error en optimización con IA: {e}")
        # Fallback a método simple
        return await optimize_routes(background_tasks, min_fill_threshold, False, db)

@router.get("/stats/efficiency")
def get_route_efficiency_stats(db: Session = Depends(get_db)):
    """Estadísticas de eficiencia de rutas"""

    total_routes = db.query(models.Route).count()
    optimized_routes = db.query(models.Route).filter(models.Route.is_optimized == True).count()

    # Obtener rutas con datos válidos
    rutas_validas = db.query(models.Route).filter(
        models.Route.total_distance_km.isnot(None)
    ).all()

    # Calcular promedios evitando divisiones por cero
    avg_distance_val = db.query(func.avg(models.Route.total_distance_km)).scalar() or 0.0
    avg_time_val = db.query(func.avg(models.Route.estimated_time_minutes)).scalar() or 0.0
    avg_fuel_val = db.query(func.avg(models.Route.fuel_consumption_liters)).scalar() or 0.0

    return {
        "total_routes": total_routes,
        "optimized_routes": optimized_routes,
        "optimization_rate": round(optimized_routes / total_routes * 100, 1) if total_routes > 0 else 0.0,
        "avg_distance_km": round(avg_distance_val, 2),
        "avg_time_minutes": round(avg_time_val, 1),
        "avg_fuel_consumption": round(avg_fuel_val, 2),
        "estimated_co2_savings_kg": round(avg_fuel_val * 2.6 * 0.3, 2)
    }