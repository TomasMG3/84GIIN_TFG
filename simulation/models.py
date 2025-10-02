# simulator/models.py
"""
Modelos de datos para la simulación de recolección de residuos.
Define las clases principales: Container, CollectionPoint, Truck, Route, etc.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum
import math


class ContainerStatus(Enum):
    """Estados posibles de un contenedor"""
    NORMAL = "normal"
    BLOCKED = "blocked"
    DAMAGED = "damaged"
    FULL = "full"
    OVERFLOWING = "overflowing"


class TruckStatus(Enum):
    """Estados posibles del camión"""
    IDLE = "idle"
    TRAVELING = "traveling"
    COLLECTING = "collecting"
    GOING_TO_DISPOSAL = "going_to_disposal"
    AT_DISPOSAL = "at_disposal"
    RETURNING = "returning"
    REFUELING = "refueling"
    BROKEN = "broken"


@dataclass
class Container:
    """Modelo de contenedor individual"""
    id: str
    capacity_liters: int  # 240, 340, 660
    current_fill_percentage: float  # 0-150 (puede desbordar)
    latitude: float
    longitude: float
    address: str
    weight_kg: float = 0.0
    status: ContainerStatus = ContainerStatus.NORMAL
    last_collection: Optional[datetime] = None
    collection_frequency: str = "daily"  # daily, alternate, weekly
    
    def __post_init__(self):
        """Calcula el peso basado en el llenado y capacidad"""
        self.weight_kg = self._calculate_weight()
    
    def _calculate_weight(self) -> float:
        """
        Calcula el peso del contenedor basado en su capacidad y llenado.
        Usa densidad promedio de residuos urbanos: 200-250 kg/m³
        """
        # Densidad promedio de residuos domésticos sin compactar
        density_kg_per_m3 = 225
        
        # Convertir litros a m³ y calcular volumen ocupado
        volume_m3 = (self.capacity_liters / 1000) * (self.current_fill_percentage / 100)
        
        # Peso base del contenedor vacío (estimación)
        container_weights = {240: 12, 340: 18, 660: 25}
        empty_weight = container_weights.get(self.capacity_liters, 15)
        
        # Peso total = contenedor vacío + residuos
        waste_weight = volume_m3 * density_kg_per_m3
        return empty_weight + waste_weight
    
    @property
    def needs_collection(self) -> bool:
        """Determina si el contenedor necesita recolección"""
        return self.current_fill_percentage >= 70.0
    
    @property
    def is_critical(self) -> bool:
        """Determina si el contenedor está en estado crítico"""
        return self.current_fill_percentage >= 90.0
    
    @property
    def is_overflowing(self) -> bool:
        """Determina si el contenedor está desbordado"""
        return self.current_fill_percentage > 100.0


@dataclass
class CollectionPoint:
    """
    Punto de recolección que puede contener múltiples contenedores.
    Representa una dirección física donde hay uno o más contenedores.
    """
    id: str
    address: str
    latitude: float
    longitude: float
    containers: List[Container] = field(default_factory=list)
    access_difficulty: float = 1.0  # Factor que afecta tiempo de servicio (1.0 = normal)
    
    @property
    def total_containers(self) -> int:
        """Número total de contenedores en este punto"""
        return len(self.containers)
    
    @property
    def needs_collection(self) -> bool:
        """Determina si este punto necesita recolección"""
        return any(container.needs_collection for container in self.containers)
    
    @property
    def containers_to_collect(self) -> List[Container]:
        """Contenedores que deben ser recolectados en este punto"""
        if not self.needs_collection:
            return []
        
        # Si algún contenedor supera el umbral, se recogen todos
        return self.containers.copy()
    
    @property
    def total_weight_to_collect(self) -> float:
        """Peso total de los contenedores a recolectar"""
        return sum(container.weight_kg for container in self.containers_to_collect)
    
    @property
    def total_volume_to_collect(self) -> float:
        """Volumen total a recolectar (en m³, sin compactar)"""
        total_volume = 0
        for container in self.containers_to_collect:
            volume_liters = (container.capacity_liters * 
                           container.current_fill_percentage / 100)
            total_volume += volume_liters / 1000  # Convertir a m³
        return total_volume
    
    @property
    def service_time_minutes(self) -> float:
        """Tiempo estimado de servicio en este punto"""
        base_time_per_container = {240: 1.5, 340: 2.0, 660: 2.5}
        
        total_time = 0
        containers_to_collect = self.containers_to_collect
        
        for i, container in enumerate(containers_to_collect):
            base_time = base_time_per_container.get(container.capacity_liters, 2.0)
            
            # Factor de reducción por contenedores múltiples (eficiencia)
            if i > 0:
                base_time *= 0.8
            
            # Factor de dificultad de acceso
            base_time *= self.access_difficulty
            
            total_time += base_time
        
        # Tiempo mínimo de maniobra si hay múltiples contenedores
        if len(containers_to_collect) > 1:
            total_time += 1.0  # 1 minuto extra por maniobras
        
        return total_time
    
    @property
    def has_critical_containers(self) -> bool:
        """Verifica si hay contenedores críticos en este punto"""
        return any(container.is_critical for container in self.containers)
    
    @property
    def has_overflowing_containers(self) -> bool:
        """Verifica si hay contenedores desbordados en este punto"""
        return any(container.is_overflowing for container in self.containers)


@dataclass
class Truck:
    """Modelo del camión recolector"""
    id: str
    fuel_capacity_l: float = 270.0
    current_fuel_l: float = 270.0
    cargo_capacity_m3: float = 19.0
    current_cargo_m3: float = 0.0
    current_cargo_weight_kg: float = 0.0
    compaction_ratio: float = 3.0  # 3:1
    max_weight_kg: float = 12000.0
    status: TruckStatus = TruckStatus.IDLE
    current_latitude: float = -33.4119  # Depot por defecto
    current_longitude: float = -70.5241
    
    # Configuración de consumo
    fuel_consumption_urban_l_per_km: float = 0.40  # Consumo en recolección
    fuel_consumption_highway_l_per_km: float = 0.28  # Consumo en traslado
    
    # Métricas de la jornada actual
    total_distance_km: float = 0.0
    total_fuel_consumed_l: float = 0.0
    total_service_time_min: float = 0.0
    containers_collected: int = 0
    collection_points_visited: int = 0
    
    @property
    def fuel_percentage(self) -> float:
        """Porcentaje de combustible restante"""
        return (self.current_fuel_l / self.fuel_capacity_l) * 100
    
    @property
    def cargo_percentage(self) -> float:
        """Porcentaje de carga actual (por volumen compactado)"""
        return (self.current_cargo_m3 / self.cargo_capacity_m3) * 100
    
    @property
    def weight_percentage(self) -> float:
        """Porcentaje de peso actual"""
        return (self.current_cargo_weight_kg / self.max_weight_kg) * 100
    
    @property
    def is_nearly_full(self) -> bool:
        """Determina si el camión está cerca de su capacidad máxima"""
        return (self.cargo_percentage >= 80 or 
                self.weight_percentage >= 85)
    
    @property
    def is_full(self) -> bool:
        """Determina si el camión está lleno"""
        return (self.cargo_percentage >= 95 or 
                self.weight_percentage >= 95)
    
    @property
    def needs_refuel(self) -> bool:
        """Determina si necesita recargar combustible"""
        return self.fuel_percentage < 20
    
    def can_collect_point(self, point: CollectionPoint) -> bool:
        """
        Determina si el camión puede recolectar un punto específico
        considerando capacidad de carga y peso.
        """
        if self.is_full:
            return False
        
        # Volumen después de compactación
        compacted_volume = point.total_volume_to_collect / self.compaction_ratio
        weight_to_add = point.total_weight_to_collect
        
        # Verificar si cabe en volumen y peso
        volume_fits = (self.current_cargo_m3 + compacted_volume) <= self.cargo_capacity_m3
        weight_fits = (self.current_cargo_weight_kg + weight_to_add) <= self.max_weight_kg
        
        return volume_fits and weight_fits
    
    def collect_point(self, point: CollectionPoint) -> Dict[str, float]:
        """
        Simula la recolección de un punto y actualiza el estado del camión.
        Retorna métricas de la operación.
        """
        if not self.can_collect_point(point):
            return {"success": False, "reason": "capacity_exceeded"}
        
        # Calcular volumen y peso a agregar
        volume_uncompacted = point.total_volume_to_collect
        volume_compacted = volume_uncompacted / self.compaction_ratio
        weight = point.total_weight_to_collect
        containers = len(point.containers_to_collect)
        service_time = point.service_time_minutes
        
        # Actualizar estado del camión
        self.current_cargo_m3 += volume_compacted
        self.current_cargo_weight_kg += weight
        self.containers_collected += containers
        self.collection_points_visited += 1
        self.total_service_time_min += service_time
        
        # Vaciar contenedores
        for container in point.containers_to_collect:
            container.current_fill_percentage = 0.0
            container.last_collection = datetime.now()
            container.weight_kg = container._calculate_weight()
        
        return {
            "success": True,
            "containers_collected": containers,
            "volume_collected_m3": volume_uncompacted,
            "weight_collected_kg": weight,
            "service_time_min": service_time
        }
    
    def consume_fuel(self, distance_km: float, is_highway: bool = False) -> float:
        """
        Consume combustible y actualiza métricas.
        Retorna el combustible consumido.
        """
        # Seleccionar tasa de consumo
        if is_highway:
            consumption_rate = self.fuel_consumption_highway_l_per_km
        else:
            consumption_rate = self.fuel_consumption_urban_l_per_km
        
        # Factor de carga (más carga = más consumo)
        load_factor = 1.0 + (self.cargo_percentage / 100) * 0.15
        
        # Calcular consumo
        fuel_consumed = distance_km * consumption_rate * load_factor
        
        # Actualizar estado
        self.current_fuel_l = max(0, self.current_fuel_l - fuel_consumed)
        self.total_fuel_consumed_l += fuel_consumed
        self.total_distance_km += distance_km
        
        return fuel_consumed
    
    def move_to(self, latitude: float, longitude: float):
        """Actualiza la posición del camión"""
        self.current_latitude = latitude
        self.current_longitude = longitude
    
    def empty_cargo(self):
        """Vacía la carga del camión (en el vertedero)"""
        self.current_cargo_m3 = 0.0
        self.current_cargo_weight_kg = 0.0
    
    def refuel(self):
        """Recarga el tanque de combustible"""
        self.current_fuel_l = self.fuel_capacity_l


@dataclass 
class RouteSegment:
    """Segmento de una ruta entre dos puntos"""
    from_point: Tuple[float, float]  # (lat, lon)
    to_point: Tuple[float, float]    # (lat, lon)
    distance_km: float
    estimated_time_min: float
    segment_type: str = "collection"  # collection, disposal, return


@dataclass
class Route:
    """Ruta completa para una jornada de recolección"""
    id: str
    collection_points: List[CollectionPoint] = field(default_factory=list)
    segments: List[RouteSegment] = field(default_factory=list)
    depot_location: Tuple[float, float] = (-33.4119, -70.5241)
    disposal_location: Tuple[float, float] = (-33.3667, -70.7500)
    
    # Métricas calculadas
    total_distance_km: float = 0.0
    total_estimated_time_min: float = 0.0
    total_service_time_min: float = 0.0
    estimated_fuel_consumption_l: float = 0.0
    estimated_co2_emissions_kg: float = 0.0
    
    @property
    def total_containers(self) -> int:
        """Total de contenedores en la ruta"""
        return sum(len(point.containers_to_collect) for point in self.collection_points)
    
    @property
    def total_weight_kg(self) -> float:
        """Peso total a recolectar"""
        return sum(point.total_weight_to_collect for point in self.collection_points)
    
    @property
    def total_volume_m3(self) -> float:
        """Volumen total a recolectar (sin compactar)"""
        return sum(point.total_volume_to_collect for point in self.collection_points)
    
    def calculate_metrics(self, truck: Truck):
        """Calcula las métricas estimadas de la ruta"""
        # Tiempo de servicio
        self.total_service_time_min = sum(
            point.service_time_minutes for point in self.collection_points
        )
        
        # Distancia y tiempo de viaje
        self.total_distance_km = sum(segment.distance_km for segment in self.segments)
        travel_time = sum(segment.estimated_time_min for segment in self.segments)
        
        # Tiempo total
        self.total_estimated_time_min = travel_time + self.total_service_time_min
        
        # Consumo de combustible estimado
        load_factor = min(1.5, 1.0 + (self.total_volume_m3 / (truck.cargo_capacity_m3 * truck.compaction_ratio)))
        self.estimated_fuel_consumption_l = self.total_distance_km * truck.fuel_consumption_urban_l_per_km * load_factor
        
        # Emisiones CO2 estimadas
        co2_factor = 2.64  # kg CO2 per liter diesel
        self.estimated_co2_emissions_kg = self.estimated_fuel_consumption_l * co2_factor


@dataclass
class SimulationState:
    """Estado actual de la simulación"""
    current_time: datetime
    truck: Truck
    remaining_points: List[CollectionPoint] = field(default_factory=list)
    completed_points: List[CollectionPoint] = field(default_factory=list)
    current_route: Optional[Route] = None
    
    # Eventos y condiciones
    weather_condition: str = "normal"  # normal, rain, wind
    traffic_factor: float = 1.0  # 1.0 = normal, >1.0 = congested
    
    # Métricas acumuladas
    total_trips_to_disposal: int = 0
    total_operating_time_min: float = 0.0
    total_unnecessary_trips: int = 0  # Viajes con contenedores <70%
    total_overflow_penalties: int = 0
    
    @property
    def work_hours_remaining(self) -> float:
        """Horas de trabajo restantes en la jornada"""
        end_time = self.current_time.replace(hour=15, minute=0, second=0, microsecond=0)
        remaining = end_time - self.current_time
        return max(0, remaining.total_seconds() / 3600)
    
    @property
    def can_continue_working(self) -> bool:
        """Determina si puede continuar trabajando"""
        return self.work_hours_remaining > 0.5  # Al menos 30 min restantes