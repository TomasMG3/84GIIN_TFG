# simulator/decision_engine.py
"""
Motor de decisiones para la simulación de recolección de residuos.
Contiene la lógica para determinar qué acciones debe tomar el camión.
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import math
from datetime import datetime, timedelta

from .models import (
    Truck, CollectionPoint, Container, SimulationState, 
    TruckStatus, Route, RouteSegment
)
from .physics import PhysicsCalculator
from .config import (
    TRUCK_CONFIG, COLLECTION_THRESHOLDS, WORKDAY_CONFIG, 
    INFRASTRUCTURE, ADJUSTMENT_FACTORS
)


class DecisionType(Enum):
    """Tipos de decisiones que puede tomar el motor"""
    CONTINUE_COLLECTION = "continue_collection"
    GO_TO_DISPOSAL = "go_to_disposal"
    RETURN_TO_DEPOT = "return_to_depot"
    REFUEL = "refuel"
    SKIP_POINT = "skip_point"
    END_SHIFT = "end_shift"


@dataclass
class Decision:
    """Representa una decisión tomada por el motor"""
    decision_type: DecisionType
    target_point: Optional[CollectionPoint] = None
    reason: str = ""
    priority: int = 0  # 0 = alta prioridad, mayor número = menor prioridad
    estimated_time_min: float = 0.0
    estimated_fuel_l: float = 0.0


class DecisionEngine:
    """Motor de decisiones para la simulación"""
    
    def __init__(self):
        self.physics = PhysicsCalculator()
        
        # Ubicaciones importantes
        self.depot_location = (
            INFRASTRUCTURE['depot']['latitude'],
            INFRASTRUCTURE['depot']['longitude']
        )
        self.disposal_location = (
            INFRASTRUCTURE['disposal_sites'][0]['latitude'],
            INFRASTRUCTURE['disposal_sites'][0]['longitude']
        )
        
        # Factores de ajuste
        self.traffic_factors = WORKDAY_CONFIG['traffic_factors']
        self.slope_factor = ADJUSTMENT_FACTORS['slope_consumption_factor']
        
    def make_decision(self, state: SimulationState) -> Decision:
        """
        Toma la decisión principal sobre qué hacer en el estado actual.
        
        Args:
            state: Estado actual de la simulación
            
        Returns:
            Decision: La decisión tomada con su justificación
        """
        truck = state.truck
        current_time = state.current_time
        
        # 1. Verificar condiciones críticas primero
        critical_decision = self._check_critical_conditions(state)
        if critical_decision:
            return critical_decision
        
        # 2. Verificar si debe ir al vertedero
        disposal_decision = self._should_go_to_disposal(state)
        if disposal_decision:
            return disposal_decision
        
        # 3. Seleccionar próximo punto de recolección
        collection_decision = self._select_next_collection_point(state)
        if collection_decision:
            return collection_decision
        
        # 4. No hay más puntos, regresar al depósito
        return Decision(
            decision_type=DecisionType.RETURN_TO_DEPOT,
            reason="No more collection points available",
            estimated_time_min=self._estimate_travel_time_to_depot(state),
            estimated_fuel_l=self._estimate_fuel_to_depot(state)
        )
    
    def _check_critical_conditions(self, state: SimulationState) -> Optional[Decision]:
        """Verifica condiciones críticas que requieren acción inmediata"""
        truck = state.truck
        current_time = state.current_time
        
        # 1. Verificar fin de jornada
        if not state.can_continue_working:
            return Decision(
                decision_type=DecisionType.END_SHIFT,
                reason="End of work shift",
                priority=0
            )
        
        # 2. Verificar combustible crítico
        if truck.needs_refuel:
            fuel_station = self._find_nearest_fuel_station(state)
            if fuel_station:
                return Decision(
                    decision_type=DecisionType.REFUEL,
                    reason=f"Fuel critical: {truck.fuel_percentage:.1f}%",
                    priority=0
                )
        
        # 3. Verificar si está completamente lleno
        if truck.is_full:
            return Decision(
                decision_type=DecisionType.GO_TO_DISPOSAL,
                reason="Truck is full",
                priority=0,
                estimated_time_min=self._estimate_disposal_time(state),
                estimated_fuel_l=self._estimate_fuel_to_disposal(state)
            )
        
        return None
    
    def _should_go_to_disposal(self, state: SimulationState) -> Optional[Decision]:
        """Determina si debe ir al vertedero basándose en múltiples factores"""
        truck = state.truck
        
        # Factor 1: Capacidad del camión
        if truck.is_nearly_full:
            # Verificar si puede recolectar al menos un punto más crítico
            critical_points = [p for p in state.remaining_points if p.has_critical_containers]
            
            can_collect_critical = False
            for point in critical_points[:3]:  # Verificar solo los 3 más próximos
                if truck.can_collect_point(point):
                    can_collect_critical = True
                    break
            
            if not can_collect_critical:
                return Decision(
                    decision_type=DecisionType.GO_TO_DISPOSAL,
                    reason=f"Nearly full ({truck.cargo_percentage:.1f}%) and cannot collect critical points",
                    priority=1,
                    estimated_time_min=self._estimate_disposal_time(state),
                    estimated_fuel_l=self._estimate_fuel_to_disposal(state)
                )
        
        # Factor 2: Tiempo restante vs tiempo para ir al vertedero y volver
        disposal_round_trip_time = self._estimate_disposal_round_trip_time(state)
        work_time_remaining = state.work_hours_remaining * 60  # minutos
        
        # Si queda poco tiempo y tiene carga considerable, mejor ir al vertedero
        if (work_time_remaining < disposal_round_trip_time + 60 and  # 60 min buffer
            truck.cargo_percentage > 30):
            return Decision(
                decision_type=DecisionType.GO_TO_DISPOSAL,
                reason=f"Insufficient time for disposal round trip ({work_time_remaining:.0f} min remaining)",
                priority=2,
                estimated_time_min=self._estimate_disposal_time(state),
                estimated_fuel_l=self._estimate_fuel_to_disposal(state)
            )
        
        return None
    
    def _select_next_collection_point(self, state: SimulationState) -> Optional[Decision]:
        """Selecciona el próximo punto de recolección óptimo"""
        truck = state.truck
        remaining_points = state.remaining_points
        
        if not remaining_points:
            return None
        
        # Filtrar puntos que el camión puede recolectar
        collectible_points = [
            point for point in remaining_points 
            if truck.can_collect_point(point) and point.needs_collection
        ]
        
        if not collectible_points:
            return None
        
        # Calcular prioridades para cada punto
        point_priorities = []
        for point in collectible_points:
            priority_score = self._calculate_point_priority(point, truck, state)
            point_priorities.append((point, priority_score))
        
        # Ordenar por prioridad (menor score = mayor prioridad)
        point_priorities.sort(key=lambda x: x[1])
        
        # Seleccionar el punto con mayor prioridad
        best_point, _ = point_priorities[0]
        
        return Decision(
            decision_type=DecisionType.CONTINUE_COLLECTION,
            target_point=best_point,
            reason=self._explain_point_selection(best_point, truck, state),
            priority=1,
            estimated_time_min=self._estimate_collection_time(best_point, truck, state),
            estimated_fuel_l=self._estimate_fuel_to_point(best_point, truck, state)
        )
    
    def _calculate_point_priority(
        self, 
        point: CollectionPoint, 
        truck: Truck, 
        state: SimulationState
    ) -> float:
        """
        Calcula la prioridad de un punto de recolección.
        Menor score = mayor prioridad.
        """
        score = 0.0
        
        # Factor 1: Nivel de llenado (más lleno = mayor prioridad)
        max_fill = max(container.current_fill_percentage for container in point.containers)
        avg_fill = sum(container.current_fill_percentage for container in point.containers) / len(point.containers)
        
        # Contenedores críticos tienen alta prioridad
        if point.has_overflowing_containers:
            score -= 1000  # Muy alta prioridad
        elif point.has_critical_containers:
            score -= 500
        elif avg_fill >= 80:
            score -= 200
        elif avg_fill >= 70:
            score -= 100
        else:
            score += 50  # Baja prioridad para contenedores no llenos
        
        # Factor 2: Distancia (más cerca = mayor prioridad)
        distance = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            point.latitude, point.longitude
        )
        score += distance * 10  # Penalización por distancia
        
        # Factor 3: Eficiencia (más contenedores por parada = mayor prioridad)
        containers_count = len(point.containers_to_collect)
        score -= containers_count * 20  # Bonificación por múltiples contenedores
        
        # Factor 4: Tiempo restante en la jornada
        estimated_time = self._estimate_collection_time(point, truck, state)
        if state.work_hours_remaining * 60 < estimated_time + 120:  # 2 horas buffer
            score += 100  # Penalización si queda poco tiempo
        
        # Factor 5: Capacidad remanente del camión
        utilization_after = ((truck.current_cargo_m3 + point.total_volume_to_collect / truck.compaction_ratio) / 
                           truck.cargo_capacity_m3)
        if utilization_after > 0.9:
            score += 200  # Penalización si se llena mucho
        
        return score
    
    def _explain_point_selection(
        self, 
        point: CollectionPoint, 
        truck: Truck, 
        state: SimulationState
    ) -> str:
        """Genera una explicación de por qué se seleccionó este punto"""
        reasons = []
        
        # Nivel de llenado
        max_fill = max(container.current_fill_percentage for container in point.containers)
        if point.has_overflowing_containers:
            reasons.append("overflowing containers")
        elif point.has_critical_containers:
            reasons.append("critical fill level")
        elif max_fill >= 80:
            reasons.append("high fill level")
        
        # Proximidad
        distance = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            point.latitude, point.longitude
        )
        if distance < 1.0:
            reasons.append("very close")
        elif distance < 3.0:
            reasons.append("nearby")
        
        # Eficiencia
        containers_count = len(point.containers_to_collect)
        if containers_count > 3:
            reasons.append("multiple containers")
        
        if not reasons:
            reasons.append("best available option")
        
        return f"Selected due to: {', '.join(reasons)}"
    
    def _estimate_collection_time(
        self, 
        point: CollectionPoint, 
        truck: Truck, 
        state: SimulationState
    ) -> float:
        """Estima el tiempo total para recolectar un punto"""
        # Tiempo de viaje al punto
        distance = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            point.latitude, point.longitude
        )
        
        # Factor de tráfico según hora del día
        current_hour = state.current_time.hour
        traffic_factor = self.traffic_factors.get(current_hour, 1.0) * state.traffic_factor
        
        # Factor de clima
        weather_factor = 1.2 if state.weather_condition == "rain" else 1.0
        
        travel_time = self.physics.calculate_travel_time(
            distance, 
            TRUCK_CONFIG['speed_residential_kmh'],
            traffic_factor,
            weather_factor
        )
        
        # Tiempo de servicio en el punto
        service_time = point.service_time_minutes
        
        return travel_time + service_time
    
    def _estimate_fuel_to_point(
        self, 
        point: CollectionPoint, 
        truck: Truck, 
        state: SimulationState
    ) -> float:
        """Estima el combustible necesario para llegar a un punto"""
        distance = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            point.latitude, point.longitude
        )
        
        return self.physics.calculate_fuel_consumption(
            distance,
            truck.fuel_consumption_urban_l_per_km,
            truck.cargo_percentage,
            state.traffic_factor,
            self.slope_factor
        )
    
    def _estimate_travel_time_to_depot(self, state: SimulationState) -> float:
        """Estima tiempo de viaje de regreso al depósito"""
        truck = state.truck
        distance = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            self.depot_location[0], self.depot_location[1]
        )
        
        return self.physics.calculate_travel_time(
            distance, 
            TRUCK_CONFIG['speed_urban_kmh'],
            state.traffic_factor
        )
    
    def _estimate_fuel_to_depot(self, state: SimulationState) -> float:
        """Estima combustible necesario para regresar al depósito"""
        truck = state.truck
        distance = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            self.depot_location[0], self.depot_location[1]
        )
        
        return self.physics.calculate_fuel_consumption(
            distance,
            truck.fuel_consumption_urban_l_per_km,
            truck.cargo_percentage,
            state.traffic_factor
        )
    
    def _estimate_disposal_time(self, state: SimulationState) -> float:
        """Estima tiempo total para ir al vertedero y vaciar"""
        truck = state.truck
        distance = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            self.disposal_location[0], self.disposal_location[1]
        )
        
        travel_time = self.physics.calculate_travel_time(
            distance, 
            TRUCK_CONFIG['speed_highway_kmh'],
            state.traffic_factor * 0.8  # Menos tráfico en autopista
        )
        
        return travel_time + TRUCK_CONFIG['disposal_time_min']
    
    def _estimate_fuel_to_disposal(self, state: SimulationState) -> float:
        """Estima combustible necesario para ir al vertedero"""
        truck = state.truck
        distance = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            self.disposal_location[0], self.disposal_location[1]
        )
        
        return self.physics.calculate_fuel_consumption(
            distance,
            truck.fuel_consumption_highway_l_per_km,
            truck.cargo_percentage,
            state.traffic_factor * 0.8
        )
    
    def _estimate_disposal_round_trip_time(self, state: SimulationState) -> float:
        """Estima tiempo total de ida y vuelta al vertedero"""
        truck = state.truck
        
        # Distancia al vertedero
        distance_to_disposal = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            self.disposal_location[0], self.disposal_location[1]
        )
        
        # Distancia del vertedero al depósito (para continuar operaciones)
        distance_disposal_to_depot = self.physics.calculate_distance(
            self.disposal_location[0], self.disposal_location[1],
            self.depot_location[0], self.depot_location[1]
        )
        
        # Tiempos de viaje
        time_to_disposal = self.physics.calculate_travel_time(
            distance_to_disposal,
            TRUCK_CONFIG['speed_highway_kmh'],
            state.traffic_factor * 0.8
        )
        
        time_back = self.physics.calculate_travel_time(
            distance_disposal_to_depot,
            TRUCK_CONFIG['speed_highway_kmh'],
            state.traffic_factor * 0.8
        )
        
        return time_to_disposal + TRUCK_CONFIG['disposal_time_min'] + time_back
    
    def _find_nearest_fuel_station(self, state: SimulationState) -> Optional[Dict]:
        """Encuentra la estación de combustible más cercana"""
        truck = state.truck
        fuel_stations = INFRASTRUCTURE.get('fuel_stations', [])
        
        if not fuel_stations:
            return None
        
        nearest_station = None
        min_distance = float('inf')
        
        for station in fuel_stations:
            distance = self.physics.calculate_distance(
                truck.current_latitude, truck.current_longitude,
                station['latitude'], station['longitude']
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_station = station
        
        return nearest_station
    
    def evaluate_collection_viability(
        self, 
        point: CollectionPoint, 
        truck: Truck, 
        state: SimulationState
    ) -> Dict[str, any]:
        """
        Evalúa la viabilidad de recolectar un punto específico.
        
        Returns:
            Dict con información sobre viabilidad, razones y métricas
        """
        evaluation = {
            'viable': True,
            'reasons': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Verificar capacidad física
        if not truck.can_collect_point(point):
            evaluation['viable'] = False
            evaluation['reasons'].append("Exceeds truck capacity")
        
        # Verificar umbral de recolección
        containers_needing_collection = [
            c for c in point.containers 
            if c.current_fill_percentage >= COLLECTION_THRESHOLDS['min_fill_single']
        ]
        
        if not containers_needing_collection and not point.has_critical_containers:
            evaluation['viable'] = False
            evaluation['reasons'].append("No containers meet collection threshold")
        
        # Verificar tiempo disponible
        estimated_time = self._estimate_collection_time(point, truck, state)
        if state.work_hours_remaining * 60 < estimated_time + 30:  # 30 min buffer
            evaluation['viable'] = False
            evaluation['reasons'].append("Insufficient time remaining")
        
        # Verificar combustible
        estimated_fuel = self._estimate_fuel_to_point(point, truck, state)
        fuel_for_return = self._estimate_fuel_to_depot(state)
        
        if truck.current_fuel_l < estimated_fuel + fuel_for_return + 20:  # 20L buffer
            evaluation['viable'] = False
            evaluation['reasons'].append("Insufficient fuel")
        
        # Calcular métricas
        evaluation['metrics'] = {
            'estimated_time_min': estimated_time,
            'estimated_fuel_l': estimated_fuel,
            'containers_count': len(point.containers_to_collect),
            'total_weight_kg': point.total_weight_to_collect,
            'total_volume_m3': point.total_volume_to_collect,
            'efficiency_score': self._calculate_efficiency_score(point, truck, state)
        }
        
        # Generar advertencias
        if point.has_overflowing_containers:
            evaluation['warnings'].append("Contains overflowing containers")
        
        avg_fill = sum(c.current_fill_percentage for c in point.containers) / len(point.containers)
        if avg_fill < COLLECTION_THRESHOLDS['min_fill_single']:
            evaluation['warnings'].append("Below optimal fill threshold - may be unnecessary trip")
        
        return evaluation
    
    def _calculate_efficiency_score(
        self, 
        point: CollectionPoint, 
        truck: Truck, 
        state: SimulationState
    ) -> float:
        """
        Calcula un score de eficiencia para un punto de recolección.
        Mayor score = más eficiente.
        """
        # Métricas base
        distance = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            point.latitude, point.longitude
        )
        containers_count = len(point.containers_to_collect)
        total_weight = point.total_weight_to_collect
        
        # Evitar división por cero
        if distance == 0:
            distance = 0.1
        
        # Score basado en contenedores por kilómetro
        containers_per_km = containers_count / distance
        
        # Score basado en peso por kilómetro
        weight_per_km = total_weight / distance
        
        # Bonificación por contenedores críticos
        critical_bonus = 1.5 if point.has_critical_containers else 1.0
        overflow_bonus = 2.0 if point.has_overflowing_containers else 1.0
        
        # Score final combinado
        efficiency_score = (containers_per_km * 10 + weight_per_km * 0.1) * critical_bonus * overflow_bonus
        
        return efficiency_score
    
    def optimize_collection_order(
        self, 
        points: List[CollectionPoint], 
        truck: Truck, 
        state: SimulationState
    ) -> List[CollectionPoint]:
        """
        Optimiza el orden de recolección de una lista de puntos.
        Usa un algoritmo simple de vecino más cercano con ajustes por prioridad.
        """
        if not points:
            return []
        
        optimized_order = []
        remaining_points = points.copy()
        current_position = (truck.current_latitude, truck.current_longitude)
        
        while remaining_points:
            best_point = None
            best_score = float('inf')
            
            for point in remaining_points:
                # Distancia desde posición actual
                distance = self.physics.calculate_distance(
                    current_position[0], current_position[1],
                    point.latitude, point.longitude
                )
                
                # Factor de prioridad (contenedores críticos tienen prioridad)
                priority_factor = 0.5 if point.has_critical_containers else 1.0
                overflow_factor = 0.3 if point.has_overflowing_containers else 1.0
                
                # Score combinado (menor = mejor)
                score = distance * priority_factor * overflow_factor
                
                if score < best_score:
                    best_score = score
                    best_point = point
            
            if best_point:
                optimized_order.append(best_point)
                remaining_points.remove(best_point)
                current_position = (best_point.latitude, best_point.longitude)
        
        return optimized_order