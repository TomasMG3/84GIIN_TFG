# simulator/events.py
"""
Sistema de eventos aleatorios para la simulación de recolección de residuos.
Maneja eventos que pueden ocurrir durante la jornada y afectan la operación.
"""

import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from .models import Truck, CollectionPoint, Container, SimulationState, ContainerStatus
from .config import EVENT_PROBABILITIES, ADJUSTMENT_FACTORS


class EventType(Enum):
    """Tipos de eventos que pueden ocurrir"""
    TRAFFIC_JAM = "traffic_jam"
    WEATHER_CHANGE = "weather_change"
    CONTAINER_BLOCKED = "container_blocked"
    CONTAINER_DAMAGED = "container_damaged"
    TRUCK_BREAKDOWN = "truck_breakdown"
    EMERGENCY_COLLECTION = "emergency_collection"
    ROAD_CLOSURE = "road_closure"
    FUEL_STATION_CLOSED = "fuel_station_closed"
    EXTRA_WASTE = "extra_waste"
    MISSING_CONTAINER = "missing_container"


class EventSeverity(Enum):
    """Severidad de los eventos"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SimulationEvent:
    """Representa un evento que ocurre durante la simulación"""
    event_type: EventType
    severity: EventSeverity
    description: str
    duration_minutes: float = 0.0
    fuel_impact: float = 0.0
    time_impact: float = 0.0
    affects_traffic: bool = False
    affects_collection: bool = False
    location: Optional[tuple] = None  # (lat, lon)
    target_point: Optional[CollectionPoint] = None
    target_container: Optional[Container] = None
    
    # Efectos en factores
    traffic_factor_change: float = 0.0
    speed_factor_change: float = 0.0
    fuel_consumption_change: float = 0.0
    service_time_change: float = 0.0


class EventManager:
    """Gestor de eventos aleatorios durante la simulación"""
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed:
            random.seed(random_seed)
        
        self.active_events: List[SimulationEvent] = []
        self.event_history: List[SimulationEvent] = []
        self.event_probabilities = EVENT_PROBABILITIES.copy()
        
        # Factores que afectan la probabilidad de eventos
        self.weather_factor = 1.0
        self.time_factor = 1.0
        
    def check_for_events(self, state: SimulationState) -> List[SimulationEvent]:
        """
        Verifica si ocurren eventos aleatorios en el estado actual.
        
        Args:
            state: Estado actual de la simulación
            
        Returns:
            Lista de eventos que ocurrieron
        """
        new_events = []
        current_time = state.current_time
        
        # Ajustar probabilidades según contexto
        self._adjust_probabilities(state)
        
        # Verificar cada tipo de evento
        for event_type, base_probability in self.event_probabilities.items():
            adjusted_probability = self._get_adjusted_probability(event_type, state)
            
            if random.random() < adjusted_probability:
                event = self._generate_event(event_type, state)
                if event:
                    new_events.append(event)
                    self.active_events.append(event)
                    self.event_history.append(event)
        
        return new_events
    
    def _adjust_probabilities(self, state: SimulationState):
        """Ajusta las probabilidades base según el contexto"""
        current_hour = state.current_time.hour
        
        # Factor de hora del día
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            self.time_factor = 1.5  # Más eventos en horas punta
        else:
            self.time_factor = 1.0
        
        # Factor climático
        if state.weather_condition == "rain":
            self.weather_factor = 2.0
        elif state.weather_condition == "wind":
            self.weather_factor = 1.3
        else:
            self.weather_factor = 1.0
    
    def _get_adjusted_probability(self, event_type: str, state: SimulationState) -> float:
        """Obtiene la probabilidad ajustada para un tipo de evento"""
        base_prob = self.event_probabilities[event_type]
        
        # Ajustes específicos por tipo de evento
        if event_type == "traffic_jam":
            return base_prob * self.time_factor
        elif event_type == "weather_delay":
            return base_prob * self.weather_factor
        elif event_type == "minor_breakdown":
            # Mayor probabilidad si el camión ha trabajado mucho
            usage_factor = 1.0 + (state.truck.total_distance_km / 100) * 0.1
            return base_prob * usage_factor
        elif event_type == "emergency_collection":
            # Menor probabilidad si ya hay muchos puntos pendientes
            points_factor = max(0.5, 1.0 - len(state.remaining_points) / 50)
            return base_prob * points_factor
        
        return base_prob
    
    def _generate_event(self, event_type: str, state: SimulationState) -> Optional[SimulationEvent]:
        """Genera un evento específico según su tipo"""
        
        if event_type == "traffic_jam":
            return self._generate_traffic_jam(state)
        elif event_type == "weather_delay":
            return self._generate_weather_event(state)
        elif event_type == "container_blocked":
            return self._generate_blocked_container(state)
        elif event_type == "minor_breakdown":
            return self._generate_breakdown(state)
        elif event_type == "emergency_collection":
            return self._generate_emergency_collection(state)
        else:
            return self._generate_generic_event(event_type, state)
    
    def _generate_traffic_jam(self, state: SimulationState) -> SimulationEvent:
        """Genera un evento de atasco de tráfico"""
        severity = random.choices(
            [EventSeverity.LOW, EventSeverity.MEDIUM, EventSeverity.HIGH],
            weights=[0.5, 0.3, 0.2]
        )[0]
        
        duration_map = {
            EventSeverity.LOW: random.uniform(10, 20),
            EventSeverity.MEDIUM: random.uniform(20, 45),
            EventSeverity.HIGH: random.uniform(45, 90)
        }
        
        traffic_impact_map = {
            EventSeverity.LOW: 1.2,
            EventSeverity.MEDIUM: 1.5,
            EventSeverity.HIGH: 2.0
        }
        
        duration = duration_map[severity]
        traffic_impact = traffic_impact_map[severity]
        
        return SimulationEvent(
            event_type=EventType.TRAFFIC_JAM,
            severity=severity,
            description=f"Atasco de tráfico {severity.value} - aumenta tiempos de viaje",
            duration_minutes=duration,
            affects_traffic=True,
            traffic_factor_change=traffic_impact - 1.0,
            fuel_consumption_change=0.1 * (traffic_impact - 1.0)
        )
    
    def _generate_weather_event(self, state: SimulationState) -> SimulationEvent:
        """Genera un evento climático"""
        weather_types = ["rain", "heavy_rain", "wind", "fog"]
        weather_type = random.choice(weather_types)
        
        severity_map = {
            "rain": EventSeverity.LOW,
            "heavy_rain": EventSeverity.MEDIUM,
            "wind": EventSeverity.LOW,
            "fog": EventSeverity.MEDIUM
        }
        
        duration_map = {
            "rain": random.uniform(30, 120),
            "heavy_rain": random.uniform(45, 90),
            "wind": random.uniform(60, 180),
            "fog": random.uniform(20, 60)
        }
        
        severity = severity_map[weather_type]
        duration = duration_map[weather_type]
        
        # Efectos en la operación
        speed_reduction = 0.15 if weather_type in ["rain", "heavy_rain"] else 0.1
        service_time_increase = 0.2 if weather_type == "heavy_rain" else 0.1
        
        return SimulationEvent(
            event_type=EventType.WEATHER_CHANGE,
            severity=severity,
            description=f"Cambio climático: {weather_type.replace('_', ' ')}",
            duration_minutes=duration,
            affects_traffic=True,
            affects_collection=True,
            speed_factor_change=-speed_reduction,
            service_time_change=service_time_increase
        )
    
    def _generate_blocked_container(self, state: SimulationState) -> Optional[SimulationEvent]:
        """Genera un evento de contenedor bloqueado"""
        if not state.remaining_points:
            return None
        
        # Seleccionar un punto aleatorio
        target_point = random.choice(state.remaining_points)
        if not target_point.containers:
            return None
        
        target_container = random.choice(target_point.containers)
        
        # Bloquear el contenedor
        target_container.status = ContainerStatus.BLOCKED
        
        return SimulationEvent(
            event_type=EventType.CONTAINER_BLOCKED,
            severity=EventSeverity.LOW,
            description=f"Contenedor {target_container.id} bloqueado por vehículo estacionado",
            duration_minutes=random.uniform(15, 45),
            affects_collection=True,
            target_point=target_point,
            target_container=target_container,
            service_time_change=0.5  # 50% más tiempo de servicio
        )
    
    def _generate_breakdown(self, state: SimulationState) -> SimulationEvent:
        """Genera un evento de avería menor"""
        breakdown_types = [
            "hydraulic_slow", "compactor_jam", "engine_warning", 
            "tire_pressure", "hydraulic_leak"
        ]
        breakdown_type = random.choice(breakdown_types)
        
        severity_map = {
            "hydraulic_slow": EventSeverity.LOW,
            "compactor_jam": EventSeverity.MEDIUM,
            "engine_warning": EventSeverity.LOW,
            "tire_pressure": EventSeverity.LOW,
            "hydraulic_leak": EventSeverity.MEDIUM
        }
        
        repair_time_map = {
            "hydraulic_slow": random.uniform(5, 15),
            "compactor_jam": random.uniform(20, 45),
            "engine_warning": random.uniform(10, 25),
            "tire_pressure": random.uniform(15, 30),
            "hydraulic_leak": random.uniform(30, 60)
        }
        
        severity = severity_map[breakdown_type]
        repair_time = repair_time_map[breakdown_type]
        
        return SimulationEvent(
            event_type=EventType.TRUCK_BREAKDOWN,
            severity=severity,
            description=f"Avería menor: {breakdown_type.replace('_', ' ')}",
            duration_minutes=repair_time,
            affects_collection=True,
            time_impact=repair_time
        )
    
    def _generate_emergency_collection(self, state: SimulationState) -> SimulationEvent:
        """Genera una recolección de emergencia"""
        # Crear un nuevo punto de emergencia cerca del camión
        truck = state.truck
        
        # Generar ubicación cerca del camión (radio de 2km)
        offset = random.uniform(-0.02, 0.02)  # Aproximadamente ±2km
        emergency_lat = truck.current_latitude + offset
        emergency_lon = truck.current_longitude + offset
        
        # Crear contenedor de emergencia (siempre desbordado)
        emergency_container = Container(
            id=f"EMERG_{datetime.now().strftime('%H%M%S')}",
            capacity_liters=660,
            current_fill_percentage=random.uniform(120, 150),
            latitude=emergency_lat,
            longitude=emergency_lon,
            address="Recolección de Emergencia",
            status=ContainerStatus.OVERFLOWING
        )
        
        # Crear punto de emergencia
        emergency_point = CollectionPoint(
            id=f"EMERGENCY_POINT_{datetime.now().strftime('%H%M%S')}",
            address="Punto de Emergencia",
            latitude=emergency_lat,
            longitude=emergency_lon,
            containers=[emergency_container],
            access_difficulty=1.3  # Más difícil de acceder
        )
        
        return SimulationEvent(
            event_type=EventType.EMERGENCY_COLLECTION,
            severity=EventSeverity.HIGH,
            description="Recolección de emergencia - contenedor desbordado reportado",
            duration_minutes=0,  # Se maneja como punto adicional
            affects_collection=True,
            target_point=emergency_point,
            location=(emergency_lat, emergency_lon)
        )
    
    def _generate_generic_event(self, event_type: str, state: SimulationState) -> SimulationEvent:
        """Genera un evento genérico"""
        return SimulationEvent(
            event_type=EventType(event_type),
            severity=EventSeverity.LOW,
            description=f"Evento: {event_type}",
            duration_minutes=random.uniform(5, 20)
        )
    
    def apply_event_effects(self, state: SimulationState):
        """Aplica los efectos de los eventos activos al estado de la simulación"""
        total_traffic_change = 0.0
        total_speed_change = 0.0
        total_fuel_change = 0.0
        total_service_change = 0.0
        
        # Sumar efectos de todos los eventos activos
        for event in self.active_events:
            total_traffic_change += event.traffic_factor_change
            total_speed_change += event.speed_factor_change
            total_fuel_change += event.fuel_consumption_change
            total_service_change += event.service_time_change
        
        # Aplicar efectos acumulados
        state.traffic_factor = max(0.5, 1.0 + total_traffic_change)
        
        # Los efectos de velocidad y combustible se aplican en los cálculos específicos
        # Aquí solo los almacenamos en el estado para referencia
        if not hasattr(state, 'speed_adjustment'):
            state.speed_adjustment = 0.0
        if not hasattr(state, 'fuel_adjustment'):
            state.fuel_adjustment = 0.0
        if not hasattr(state, 'service_adjustment'):
            state.service_adjustment = 0.0
            
        state.speed_adjustment = total_speed_change
        state.fuel_adjustment = total_fuel_change
        state.service_adjustment = total_service_change
    
    def update_events(self, state: SimulationState, elapsed_minutes: float):
        """
        Actualiza los eventos activos, eliminando los que han expirado.
        
        Args:
            state: Estado actual de la simulación
            elapsed_minutes: Minutos transcurridos desde la última actualización
        """
        expired_events = []
        
        for event in self.active_events:
            if event.duration_minutes > 0:
                event.duration_minutes -= elapsed_minutes
                
                if event.duration_minutes <= 0:
                    expired_events.append(event)
        
        # Remover eventos expirados
        for event in expired_events:
            self.active_events.remove(event)
            self._resolve_event(event, state)
    
    def _resolve_event(self, event: SimulationEvent, state: SimulationState):
        """Resuelve un evento que ha expirado"""
        if event.event_type == EventType.CONTAINER_BLOCKED:
            # Desbloquear contenedor
            if event.target_container:
                event.target_container.status = ContainerStatus.NORMAL
        
        elif event.event_type == EventType.EMERGENCY_COLLECTION:
            # Agregar punto de emergencia a puntos restantes
            if event.target_point and event.target_point not in state.remaining_points:
                state.remaining_points.insert(0, event.target_point)  # Alta prioridad
    
    def get_active_events_summary(self) -> List[Dict[str, Any]]:
        """Obtiene un resumen de los eventos activos"""
        summary = []
        
        for event in self.active_events:
            summary.append({
                'type': event.event_type.value,
                'severity': event.severity.value,
                'description': event.description,
                'remaining_time_min': max(0, event.duration_minutes),
                'affects_traffic': event.affects_traffic,
                'affects_collection': event.affects_collection
            })
        
        return summary
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de los eventos ocurridos"""
        stats = {
            'total_events': len(self.event_history),
            'active_events': len(self.active_events),
            'events_by_type': {},
            'events_by_severity': {},
            'total_impact_time_min': 0.0
        }
        
        for event in self.event_history:
            # Contar por tipo
            event_type = event.event_type.value
            stats['events_by_type'][event_type] = stats['events_by_type'].get(event_type, 0) + 1
            
            # Contar por severidad
            severity = event.severity.value
            stats['events_by_severity'][severity] = stats['events_by_severity'].get(severity, 0) + 1
            
            # Sumar tiempo de impacto
            stats['total_impact_time_min'] += event.time_impact
        
        return stats
    
    def force_event(self, event_type: EventType, state: SimulationState) -> Optional[SimulationEvent]:
        """
        Fuerza la ocurrencia de un evento específico (para testing o escenarios específicos).
        
        Args:
            event_type: Tipo de evento a generar
            state: Estado actual de la simulación
            
        Returns:
            El evento generado o None si no se pudo generar
        """
        event = self._generate_event(event_type.value, state)
        
        if event:
            self.active_events.append(event)
            self.event_history.append(event)
        
        return event
    
    def clear_all_events(self):
        """Limpia todos los eventos activos (para reset de simulación)"""
        self.active_events.clear()
    
    def disable_event_type(self, event_type: EventType):
        """Deshabilita un tipo de evento específico"""
        if event_type.value in self.event_probabilities:
            self.event_probabilities[event_type.value] = 0.0
    
    def enable_event_type(self, event_type: EventType, probability: float):
        """Habilita un tipo de evento con una probabilidad específica"""
        self.event_probabilities[event_type.value] = probability