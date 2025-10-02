# simulator/metrics.py
"""
Módulo de cálculo de métricas y KPIs para la simulación de recolección de residuos.
Calcula eficiencia operacional, impacto ambiental y costos.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from .models import Truck, CollectionPoint, Container, SimulationState, Route
from .config import (
    TRUCK_CONFIG, COLLECTION_THRESHOLDS, KPI_TARGETS, 
    EMISSIONS_CONFIG, COST_CONFIG, WASTE_CONFIG
)


@dataclass
class OperationalMetrics:
    """Métricas operacionales de la simulación"""
    # Eficiencia de recolección
    total_containers_collected: int = 0
    total_points_visited: int = 0
    total_weight_collected_kg: float = 0.0
    total_volume_collected_m3: float = 0.0
    
    # Eficiencia de tiempo
    total_operation_time_min: float = 0.0
    total_travel_time_min: float = 0.0
    total_service_time_min: float = 0.0
    total_idle_time_min: float = 0.0
    
    # Eficiencia de distancia
    total_distance_km: float = 0.0
    productive_distance_km: float = 0.0  # Distancia entre puntos de recolección
    non_productive_distance_km: float = 0.0  # Viajes al vertedero, retornos
    
    # Eficiencia del camión
    average_capacity_utilization: float = 0.0
    max_capacity_reached: float = 0.0
    trips_to_disposal: int = 0
    
    # Métricas de servicio
    containers_over_90_percent: int = 0
    containers_overflowing: int = 0
    unnecessary_collections: int = 0  # Contenedores < 70%
    missed_collections: int = 0


@dataclass
class EnvironmentalMetrics:
    """Métricas de impacto ambiental"""
    total_fuel_consumed_l: float = 0.0
    co2_emissions_kg: float = 0.0
    nox_emissions_g: float = 0.0
    pm_emissions_g: float = 0.0
    
    # Métricas por unidad
    fuel_per_container_l: float = 0.0
    fuel_per_ton_l: float = 0.0
    fuel_per_km_l: float = 0.0
    co2_per_container_kg: float = 0.0
    co2_per_ton_kg: float = 0.0


@dataclass
class EconomicMetrics:
    """Métricas económicas y de costos"""
    fuel_cost_clp: float = 0.0
    labor_cost_clp: float = 0.0
    maintenance_cost_clp: float = 0.0
    disposal_cost_clp: float = 0.0
    penalty_costs_clp: float = 0.0
    
    total_operational_cost_clp: float = 0.0
    cost_per_container_clp: float = 0.0
    cost_per_ton_clp: float = 0.0
    cost_per_km_clp: float = 0.0


@dataclass
class QualityMetrics:
    """Métricas de calidad del servicio"""
    service_level_score: float = 0.0  # 0-100
    route_efficiency_score: float = 0.0  # 0-100
    time_efficiency_score: float = 0.0  # 0-100
    fuel_efficiency_score: float = 0.0  # 0-100
    
    # Penalizaciones
    overflow_penalty_points: int = 0
    unnecessary_trip_penalty_points: int = 0
    time_violation_penalty_points: int = 0
    
    overall_performance_score: float = 0.0  # 0-100


class MetricsCalculator:
    """Calculadora de métricas y KPIs de la simulación"""
    
    def __init__(self):
        self.operational_metrics = OperationalMetrics()
        self.environmental_metrics = EnvironmentalMetrics()
        self.economic_metrics = EconomicMetrics()
        self.quality_metrics = QualityMetrics()
        
        # Referencias de configuración
        self.kpi_targets = KPI_TARGETS
        self.emissions_config = EMISSIONS_CONFIG
        self.cost_config = COST_CONFIG
        
    def calculate_all_metrics(
        self, 
        truck: Truck, 
        completed_points: List[CollectionPoint],
        state: SimulationState,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Calcula todas las métricas de la simulación.
        
        Returns:
            Diccionario con todas las métricas organizadas por categorías
        """
        # Calcular métricas operacionales
        self._calculate_operational_metrics(truck, completed_points, state, start_time, end_time)
        
        # Calcular métricas ambientales
        self._calculate_environmental_metrics(truck, completed_points)
        
        # Calcular métricas económicas
        self._calculate_economic_metrics(truck, completed_points, start_time, end_time)
        
        # Calcular métricas de calidad
        self._calculate_quality_metrics(truck, completed_points, state)
        
        return {
            'operational': self.operational_metrics,
            'environmental': self.environmental_metrics,
            'economic': self.economic_metrics,
            'quality': self.quality_metrics,
            'summary': self._generate_summary(),
            'kpi_comparison': self._compare_with_targets()
        }
    
    def _calculate_operational_metrics(
        self, 
        truck: Truck, 
        completed_points: List[CollectionPoint],
        state: SimulationState,
        start_time: datetime,
        end_time: datetime
    ):
        """Calcula métricas operacionales"""
        # Contenedores y puntos
        self.operational_metrics.total_containers_collected = truck.containers_collected
        self.operational_metrics.total_points_visited = len(completed_points)
        
        # Peso y volumen total
        total_weight = 0.0
        total_volume = 0.0
        containers_over_90 = 0
        containers_overflowing = 0
        unnecessary_collections = 0
        
        for point in completed_points:
            for container in point.containers:
                if container.last_collection and container.last_collection >= start_time:
                    # Este contenedor fue recolectado en esta simulación
                    total_weight += container.weight_kg
                    total_volume += (container.capacity_liters / 1000) * (container.current_fill_percentage / 100)
                    
                    # Verificar niveles
                    if container.current_fill_percentage >= 100:
                        containers_overflowing += 1
                    elif container.current_fill_percentage >= 90:
                        containers_over_90 += 1
                    elif container.current_fill_percentage < 70:
                        unnecessary_collections += 1
        
        self.operational_metrics.total_weight_collected_kg = total_weight
        self.operational_metrics.total_volume_collected_m3 = total_volume
        self.operational_metrics.containers_over_90_percent = containers_over_90
        self.operational_metrics.containers_overflowing = containers_overflowing
        self.operational_metrics.unnecessary_collections = unnecessary_collections
        
        # Tiempos
        operation_time = (end_time - start_time).total_seconds() / 60  # minutos
        self.operational_metrics.total_operation_time_min = operation_time
        self.operational_metrics.total_service_time_min = truck.total_service_time_min
        
        # Estimar tiempo de viaje (operación total - servicio)
        self.operational_metrics.total_travel_time_min = max(0, 
            operation_time - truck.total_service_time_min)
        
        # Distancias
        self.operational_metrics.total_distance_km = truck.total_distance_km
        
        # Utilización de capacidad (promedio durante la operación)
        if truck.collection_points_visited > 0:
            self.operational_metrics.average_capacity_utilization = (
                self.operational_metrics.total_volume_collected_m3 / 
                (truck.cargo_capacity_m3 * truck.compaction_ratio)
            ) * 100
        
        self.operational_metrics.max_capacity_reached = truck.cargo_percentage
        self.operational_metrics.trips_to_disposal = state.total_trips_to_disposal
    
    def _calculate_environmental_metrics(
        self, 
        truck: Truck, 
        completed_points: List[CollectionPoint]
    ):
        """Calcula métricas de impacto ambiental"""
        # Consumo de combustible
        self.environmental_metrics.total_fuel_consumed_l = truck.total_fuel_consumed_l
        
        # Emisiones CO2
        co2_factor = EMISSIONS_CONFIG['co2_kg_per_liter_diesel']
        self.environmental_metrics.co2_emissions_kg = (
            truck.total_fuel_consumed_l * co2_factor
        )
        
        # Emisiones NOx y PM
        nox_factor = EMISSIONS_CONFIG['nox_g_per_km']
        pm_factor = EMISSIONS_CONFIG['pm_g_per_km']
        
        self.environmental_metrics.nox_emissions_g = truck.total_distance_km * nox_factor
        self.environmental_metrics.pm_emissions_g = truck.total_distance_km * pm_factor
        
        # Métricas por unidad
        if truck.containers_collected > 0:
            self.environmental_metrics.fuel_per_container_l = (
                truck.total_fuel_consumed_l / truck.containers_collected
            )
            self.environmental_metrics.co2_per_container_kg = (
                self.environmental_metrics.co2_emissions_kg / truck.containers_collected
            )
        
        if self.operational_metrics.total_weight_collected_kg > 0:
            weight_tons = self.operational_metrics.total_weight_collected_kg / 1000
            self.environmental_metrics.fuel_per_ton_l = (
                truck.total_fuel_consumed_l / weight_tons
            )
            self.environmental_metrics.co2_per_ton_kg = (
                self.environmental_metrics.co2_emissions_kg / weight_tons
            )
        
        if truck.total_distance_km > 0:
            self.environmental_metrics.fuel_per_km_l = (
                truck.total_fuel_consumed_l / truck.total_distance_km
            )
    
    def _calculate_economic_metrics(
        self, 
        truck: Truck, 
        completed_points: List[CollectionPoint],
        start_time: datetime,
        end_time: datetime
    ):
        """Calcula métricas económicas"""
        # Costo de combustible
        self.economic_metrics.fuel_cost_clp = (
            truck.total_fuel_consumed_l * COST_CONFIG['fuel_cost_per_liter']
        )
        
        # Costo laboral
        operation_hours = (end_time - start_time).total_seconds() / 3600
        self.economic_metrics.labor_cost_clp = (
            operation_hours * COST_CONFIG['driver_hourly_cost']
        )
        
        # Costo de mantenimiento
        self.economic_metrics.maintenance_cost_clp = (
            truck.total_distance_km * COST_CONFIG['truck_maintenance_per_km']
        )
        
        # Costo de disposición
        weight_tons = self.operational_metrics.total_weight_collected_kg / 1000
        self.economic_metrics.disposal_cost_clp = (
            weight_tons * COST_CONFIG['disposal_cost_per_ton']
        )
        
        # Penalizaciones
        overflow_penalties = (
            self.operational_metrics.containers_overflowing * 
            COST_CONFIG['penalty_overflow_clp']
        )
        
        # Penalización por viajes innecesarios (contenedores < 70%)
        unnecessary_penalties = (
            self.operational_metrics.unnecessary_collections * 
            COST_CONFIG.get('penalty_unnecessary_trip_clp', 5000)
        )
        
        self.economic_metrics.penalty_costs_clp = overflow_penalties + unnecessary_penalties
        
        # Costo total
        self.economic_metrics.total_operational_cost_clp = (
            self.economic_metrics.fuel_cost_clp +
            self.economic_metrics.labor_cost_clp +
            self.economic_metrics.maintenance_cost_clp +
            self.economic_metrics.disposal_cost_clp +
            self.economic_metrics.penalty_costs_clp
        )
        
        # Costos por unidad
        if truck.containers_collected > 0:
            self.economic_metrics.cost_per_container_clp = (
                self.economic_metrics.total_operational_cost_clp / truck.containers_collected
            )
        
        if weight_tons > 0:
            self.economic_metrics.cost_per_ton_clp = (
                self.economic_metrics.total_operational_cost_clp / weight_tons
            )
        
        if truck.total_distance_km > 0:
            self.economic_metrics.cost_per_km_clp = (
                self.economic_metrics.total_operational_cost_clp / truck.total_distance_km
            )
    
    def _calculate_quality_metrics(
        self, 
        truck: Truck, 
        completed_points: List[CollectionPoint],
        state: SimulationState
    ):
        """Calcula métricas de calidad del servicio"""
        # Score de nivel de servicio (basado en contenedores atendidos)
        total_containers_needing_service = 0
        containers_serviced = 0
        
        for point in completed_points:
            for container in point.containers:
                if container.current_fill_percentage >= 70:  # Necesitaba servicio
                    total_containers_needing_service += 1
                    if container.last_collection:  # Fue atendido
                        containers_serviced += 1
        
        if total_containers_needing_service > 0:
            self.quality_metrics.service_level_score = (
                containers_serviced / total_containers_needing_service
            ) * 100
        else:
            self.quality_metrics.service_level_score = 100
        
        # Score de eficiencia de ruta
        if truck.total_distance_km > 0 and truck.containers_collected > 0:
            containers_per_km = truck.containers_collected / truck.total_distance_km
            target_containers_per_km = 2.0  # Target: 2 contenedores por km
            self.quality_metrics.route_efficiency_score = min(100,
                (containers_per_km / target_containers_per_km) * 100
            )
        
        # Score de eficiencia de tiempo
        if self.operational_metrics.total_operation_time_min > 0:
            containers_per_hour = (
                truck.containers_collected / 
                (self.operational_metrics.total_operation_time_min / 60)
            )
            target_containers_per_hour = 15  # Target: 15 contenedores por hora
            self.quality_metrics.time_efficiency_score = min(100,
                (containers_per_hour / target_containers_per_hour) * 100
            )
        
        # Score de eficiencia de combustible
        if (self.environmental_metrics.fuel_per_ton_l > 0 and 
            KPI_TARGETS['max_fuel_per_ton'] > 0):
            fuel_efficiency_ratio = (
                KPI_TARGETS['max_fuel_per_ton'] / 
                self.environmental_metrics.fuel_per_ton_l
            )
            self.quality_metrics.fuel_efficiency_score = min(100,
                fuel_efficiency_ratio * 100
            )
        
        # Penalizaciones
        self.quality_metrics.overflow_penalty_points = (
            self.operational_metrics.containers_overflowing * 
            COLLECTION_THRESHOLDS['overflow_penalty']
        )
        
        self.quality_metrics.unnecessary_trip_penalty_points = (
            self.operational_metrics.unnecessary_collections * 
            COLLECTION_THRESHOLDS['unnecessary_trip_penalty']
        )
        
        # Score general de rendimiento
        scores = [
            self.quality_metrics.service_level_score,
            self.quality_metrics.route_efficiency_score,
            self.quality_metrics.time_efficiency_score,
            self.quality_metrics.fuel_efficiency_score
        ]
        
        valid_scores = [s for s in scores if s > 0]
        if valid_scores:
            base_score = sum(valid_scores) / len(valid_scores)
            
            # Aplicar penalizaciones
            penalty_deduction = (
                self.quality_metrics.overflow_penalty_points +
                self.quality_metrics.unnecessary_trip_penalty_points
            )
            
            self.quality_metrics.overall_performance_score = max(0,
                base_score - penalty_deduction
            )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Genera un resumen ejecutivo de las métricas"""
        return {
            'containers_collected': self.operational_metrics.total_containers_collected,
            'points_visited': self.operational_metrics.total_points_visited,
            'total_distance_km': round(self.operational_metrics.total_distance_km, 2),
            'operation_time_hours': round(self.operational_metrics.total_operation_time_min / 60, 2),
            'fuel_consumed_l': round(self.environmental_metrics.total_fuel_consumed_l, 2),
            'co2_emissions_kg': round(self.environmental_metrics.co2_emissions_kg, 2),
            'total_cost_clp': round(self.economic_metrics.total_operational_cost_clp, 0),
            'performance_score': round(self.quality_metrics.overall_performance_score, 1),
            'capacity_utilization_avg': round(self.operational_metrics.average_capacity_utilization, 1),
            'service_level': round(self.quality_metrics.service_level_score, 1)
        }
    
    def _compare_with_targets(self) -> Dict[str, Dict[str, Any]]:
        """Compara las métricas obtenidas con los objetivos establecidos"""
        comparisons = {}
        
        # Eficiencia de combustible por tonelada
        if self.environmental_metrics.fuel_per_ton_l > 0:
            comparisons['fuel_efficiency'] = {
                'actual': self.environmental_metrics.fuel_per_ton_l,
                'target': KPI_TARGETS['max_fuel_per_ton'],
                'status': 'good' if self.environmental_metrics.fuel_per_ton_l <= KPI_TARGETS['max_fuel_per_ton'] else 'warning',
                'percentage': (self.environmental_metrics.fuel_per_ton_l / KPI_TARGETS['max_fuel_per_ton']) * 100
            }
        
        # Utilización de capacidad
        comparisons['capacity_utilization'] = {
            'actual': self.operational_metrics.average_capacity_utilization,
            'target': KPI_TARGETS['min_capacity_utilization'] * 100,
            'status': 'good' if self.operational_metrics.average_capacity_utilization >= KPI_TARGETS['min_capacity_utilization'] * 100 else 'warning',
            'percentage': self.operational_metrics.average_capacity_utilization / (KPI_TARGETS['min_capacity_utilization'] * 100) * 100
        }
        
        # Emisiones CO2 por tonelada
        if self.environmental_metrics.co2_per_ton_kg > 0:
            comparisons['co2_emissions'] = {
                'actual': self.environmental_metrics.co2_per_ton_kg,
                'target': KPI_TARGETS['max_co2_per_ton'],
                'status': 'good' if self.environmental_metrics.co2_per_ton_kg <= KPI_TARGETS['max_co2_per_ton'] else 'warning',
                'percentage': (self.environmental_metrics.co2_per_ton_kg / KPI_TARGETS['max_co2_per_ton']) * 100
            }
        
        return comparisons
    
    def generate_detailed_report(self) -> str:
        """Genera un reporte detallado en texto de las métricas"""
        report = []
        report.append("=== REPORTE DE MÉTRICAS DE SIMULACIÓN ===\n")
        
        # Resumen ejecutivo
        summary = self._generate_summary()
        report.append("RESUMEN EJECUTIVO:")
        report.append(f"• Contenedores recolectados: {summary['containers_collected']}")
        report.append(f"• Puntos visitados: {summary['points_visited']}")
        report.append(f"• Distancia total: {summary['total_distance_km']} km")
        report.append(f"• Tiempo de operación: {summary['operation_time_hours']} horas")
        report.append(f"• Score de rendimiento: {summary['performance_score']}/100\n")
        
        # Métricas operacionales
        report.append("MÉTRICAS OPERACIONALES:")
        report.append(f"• Peso total recolectado: {self.operational_metrics.total_weight_collected_kg:.1f} kg")
        report.append(f"• Volumen total recolectado: {self.operational_metrics.total_volume_collected_m3:.2f} m³")
        report.append(f"• Utilización promedio de capacidad: {self.operational_metrics.average_capacity_utilization:.1f}%")
        report.append(f"• Contenedores críticos (>90%): {self.operational_metrics.containers_over_90_percent}")
        report.append(f"• Contenedores desbordados: {self.operational_metrics.containers_overflowing}")
        report.append(f"• Recolecciones innecesarias (<70%): {self.operational_metrics.unnecessary_collections}\n")
        
        # Métricas ambientales
        report.append("MÉTRICAS AMBIENTALES:")
        report.append(f"• Combustible consumido: {self.environmental_metrics.total_fuel_consumed_l:.2f} L")
        report.append(f"• Emisiones CO₂: {self.environmental_metrics.co2_emissions_kg:.2f} kg")
        report.append(f"• Combustible por contenedor: {self.environmental_metrics.fuel_per_container_l:.2f} L")
        report.append(f"• Combustible por tonelada: {self.environmental_metrics.fuel_per_ton_l:.1f} L/ton")
        report.append(f"• CO₂ por tonelada: {self.environmental_metrics.co2_per_ton_kg:.1f} kg/ton\n")
        
        # Métricas económicas
        report.append("MÉTRICAS ECONÓMICAS:")
        report.append(f"• Costo total de operación: ${self.economic_metrics.total_operational_cost_clp:,.0f} CLP")
        report.append(f"• Costo de combustible: ${self.economic_metrics.fuel_cost_clp:,.0f} CLP")
        report.append(f"• Costo laboral: ${self.economic_metrics.labor_cost_clp:,.0f} CLP")
        report.append(f"• Costo por contenedor: ${self.economic_metrics.cost_per_container_clp:,.0f} CLP")
        report.append(f"• Penalizaciones: ${self.economic_metrics.penalty_costs_clp:,.0f} CLP\n")
        
        return "\n".join(report)

def calculate_route_optimization_score(
    original_route: Route,
    optimized_route: Route,
    actual_metrics: OperationalMetrics
) -> Dict[str, float]:
    """
    Calcula un score de qué tan bien funcionó la optimización de rutas.
    
    Args:
        original_route: Ruta sin optimizar
        optimized_route: Ruta optimizada
        actual_metrics: Métricas reales de la ejecución
        
    Returns:
        Dict con scores de optimización
    """
    scores = {
        'distance_improvement': 0.0,
        'time_improvement': 0.0,
        'fuel_improvement': 0.0,
        'overall_optimization_score': 0.0
    }
    
    if original_route.total_distance_km > 0:
        distance_reduction = ((original_route.total_distance_km - optimized_route.total_distance_km) / 
                            original_route.total_distance_km) * 100
        scores['distance_improvement'] = max(0, distance_reduction)
    
    if original_route.total_estimated_time_min > 0:
        time_reduction = ((original_route.total_estimated_time_min - optimized_route.total_estimated_time_min) / 
                         original_route.total_estimated_time_min) * 100
        scores['time_improvement'] = max(0, time_reduction)
    
    if original_route.estimated_fuel_consumption_l > 0:
        fuel_reduction = ((original_route.estimated_fuel_consumption_l - optimized_route.estimated_fuel_consumption_l) / 
                         original_route.estimated_fuel_consumption_l) * 100
        scores['fuel_improvement'] = max(0, fuel_reduction)
    
    # Score general de optimización
    improvements = [
        scores['distance_improvement'],
        scores['time_improvement'],
        scores['fuel_improvement']
    ]
    
    valid_improvements = [imp for imp in improvements if imp > 0]
    if valid_improvements:
        scores['overall_optimization_score'] = sum(valid_improvements) / len(valid_improvements)
    
    return scores