# simulator/main_simulation.py
"""
Orquestador principal de la simulación de recolección de residuos.
Coordina todos los componentes y ejecuta la simulación paso a paso.
"""
"https://claude.ai/share/6f0e23e3-e066-4f5d-9682-a761d20fd2d3"

import random
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import os

from .models import (
    Truck, CollectionPoint, Container, SimulationState, 
    TruckStatus, Route, RouteSegment
)
from .decision_engine import DecisionEngine, DecisionType
from .events import EventManager, SimulationEvent
from .metrics import MetricsCalculator, OperationalMetrics
from .physics import PhysicsCalculator
from .visualizer import SimulationVisualizer
from .config import (
    TRUCK_CONFIG, SIMULATION_CONFIG, WORKDAY_CONFIG, 
    INFRASTRUCTURE, COLLECTION_THRESHOLDS
)


@dataclass
class SimulationConfig:
    """Configuración específica para una simulación"""
    start_time: datetime = field(default_factory=lambda: datetime.now().replace(hour=7, minute=0, second=0, microsecond=0))
    end_time: datetime = field(default_factory=lambda: datetime.now().replace(hour=15, minute=0, second=0, microsecond=0))
    time_step_minutes: int = 5
    enable_events: bool = True
    enable_traffic_simulation: bool = True
    random_seed: Optional[int] = None
    min_fill_threshold: float = 70.0
    weather_condition: str = "normal"  # normal, rain, wind
    traffic_base_factor: float = 1.0


@dataclass
class SimulationResults:
    """Resultados completos de una simulación"""
    config: SimulationConfig
    truck_final_state: Truck
    completed_points: List[CollectionPoint]
    remaining_points: List[CollectionPoint]
    events_occurred: List[SimulationEvent]
    metrics: Dict[str, Any]
    truck_history: List[Dict[str, Any]]
    execution_log: List[str]
    success: bool
    total_runtime_seconds: float
    
    def save_to_file(self, filepath: str):
        """Guarda los resultados a un archivo JSON"""
        # Convertir objetos complejos a diccionarios
        data = {
            'config': {
                'start_time': self.config.start_time.isoformat(),
                'end_time': self.config.end_time.isoformat(),
                'time_step_minutes': self.config.time_step_minutes,
                'enable_events': self.config.enable_events,
                'min_fill_threshold': self.config.min_fill_threshold,
                'weather_condition': self.config.weather_condition
            },
            'summary': {
                'success': self.success,
                'total_runtime_seconds': self.total_runtime_seconds,
                'containers_collected': self.truck_final_state.containers_collected,
                'points_visited': len(self.completed_points),
                'total_distance_km': self.truck_final_state.total_distance_km,
                'fuel_consumed_l': self.truck_final_state.total_fuel_consumed_l,
                'events_count': len(self.events_occurred)
            },
            'metrics': self.metrics,
            'truck_history': self.truck_history,
            'execution_log': self.execution_log[-50:]  # Últimas 50 entradas
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)


class WasteCollectionSimulator:
    """Simulador principal de recolección de residuos"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        
        # Configurar logging
        self._setup_logging()
        
        # Componentes del simulador
        self.decision_engine = DecisionEngine()
        self.event_manager = EventManager(self.config.random_seed)
        self.metrics_calculator = MetricsCalculator()
        self.physics = PhysicsCalculator()
        self.visualizer = SimulationVisualizer()
        
        # Estado de la simulación
        self.current_state: Optional[SimulationState] = None
        self.truck_history: List[Dict[str, Any]] = []
        self.execution_log: List[str] = []
        
        # Configurar semilla aleatoria
        if self.config.random_seed:
            random.seed(self.config.random_seed)
    
    def _setup_logging(self):
        """Configura el sistema de logging"""
        log_level = getattr(logging, SIMULATION_CONFIG.get('log_level', 'INFO'))
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('simulation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_simulation(
        self, 
        collection_points: List[CollectionPoint],
        initial_truck_state: Optional[Truck] = None
    ) -> SimulationResults:
        """
        Ejecuta una simulación completa de recolección de residuos.
        
        Args:
            collection_points: Lista de puntos de recolección disponibles
            initial_truck_state: Estado inicial del camión (opcional)
            
        Returns:
            Resultados completos de la simulación
        """
        start_execution_time = datetime.now()
        
        try:
            self.logger.info("=== INICIANDO SIMULACIÓN DE RECOLECCIÓN ===")
            self._log(f"Configuración: {self.config.min_fill_threshold}% umbral mínimo")
            
            # Inicializar estado
            self._initialize_simulation(collection_points, initial_truck_state)
            
            # Ejecutar bucle principal
            self._run_main_loop()
            
            # Calcular métricas finales
            final_metrics = self._calculate_final_metrics(start_execution_time)
            
            # Crear resultados
            results = SimulationResults(
                config=self.config,
                truck_final_state=self.current_state.truck,
                completed_points=self.current_state.completed_points,
                remaining_points=self.current_state.remaining_points,
                events_occurred=self.event_manager.event_history,
                metrics=final_metrics,
                truck_history=self.truck_history,
                execution_log=self.execution_log,
                success=True,
                total_runtime_seconds=(datetime.now() - start_execution_time).total_seconds()
            )
            
            self.logger.info("=== SIMULACIÓN COMPLETADA EXITOSAMENTE ===")
            return results
            
        except Exception as e:
            self.logger.error(f"Error durante la simulación: {str(e)}")
            
            # Crear resultados de error
            return SimulationResults(
                config=self.config,
                truck_final_state=self.current_state.truck if self.current_state else Truck("ERROR"),
                completed_points=[],
                remaining_points=collection_points,
                events_occurred=self.event_manager.event_history,
                metrics={},
                truck_history=self.truck_history,
                execution_log=self.execution_log + [f"ERROR: {str(e)}"],
                success=False,
                total_runtime_seconds=(datetime.now() - start_execution_time).total_seconds()
            )
    
    def _initialize_simulation(
        self, 
        collection_points: List[CollectionPoint],
        initial_truck_state: Optional[Truck]
    ):
        """Inicializa el estado de la simulación"""
        # Crear o usar camión inicial
        if initial_truck_state:
            truck = initial_truck_state
        else:
            truck = Truck(
                id="TRUCK_001",
                fuel_capacity_l=TRUCK_CONFIG['fuel_capacity_l'],
                current_fuel_l=TRUCK_CONFIG['fuel_capacity_l'],
                cargo_capacity_m3=TRUCK_CONFIG['capacity_m3'],
                compaction_ratio=TRUCK_CONFIG['compaction_ratio']
            )
        
        # Filtrar puntos que requieren recolección
        points_needing_collection = [
            point for point in collection_points 
            if point.needs_collection and any(
                container.current_fill_percentage >= self.config.min_fill_threshold 
                for container in point.containers
            )
        ]
        
        self._log(f"Puntos que requieren recolección: {len(points_needing_collection)} de {len(collection_points)}")
        
        # Optimizar orden inicial de puntos
        optimized_points = self.decision_engine.optimize_collection_order(
            points_needing_collection, truck, None
        )
        
        # Crear estado inicial
        self.current_state = SimulationState(
            current_time=self.config.start_time,
            truck=truck,
            remaining_points=optimized_points,
            completed_points=[],
            weather_condition=self.config.weather_condition,
            traffic_factor=self.config.traffic_base_factor
        )
        
        self._log(f"Estado inicial creado - Camión en ({truck.current_latitude}, {truck.current_longitude})")
        self._record_truck_state()
    
    def _run_main_loop(self):
        """Ejecuta el bucle principal de la simulación"""
        step_count = 0
        max_steps = (8 * 60) // self.config.time_step_minutes  # Máximo 8 horas
        
        while (self.current_state.current_time < self.config.end_time and 
               step_count < max_steps and
               self.current_state.can_continue_working):
            
            step_count += 1
            self._log(f"=== PASO {step_count} - {self.current_state.current_time.strftime('%H:%M')} ===")
            
            # Verificar y aplicar eventos aleatorios
            if self.config.enable_events:
                new_events = self.event_manager.check_for_events(self.current_state)
                for event in new_events:
                    self._log(f"EVENTO: {event.description}")
                
                self.event_manager.apply_event_effects(self.current_state)
            
            # Tomar decisión principal
            decision = self.decision_engine.make_decision(self.current_state)
            self._log(f"DECISIÓN: {decision.decision_type.value} - {decision.reason}")
            
            # Ejecutar la decisión
            self._execute_decision(decision)
            
            # Actualizar eventos activos
            self.event_manager.update_events(self.current_state, self.config.time_step_minutes)
            
            # Avanzar tiempo
            self.current_state.current_time += timedelta(minutes=self.config.time_step_minutes)
            
            # Registrar estado del camión
            self._record_truck_state()
            
            # Verificar condiciones de parada
            if not self.current_state.remaining_points and self.current_state.truck.current_cargo_m3 == 0:
                self._log("Todos los puntos completados y camión vacío - Finalizando")
                break
        
        self._log(f"Bucle principal terminado después de {step_count} pasos")
    
    def _execute_decision(self, decision):
        """Ejecuta una decisión tomada por el motor de decisiones"""
        truck = self.current_state.truck
        
        if decision.decision_type == DecisionType.CONTINUE_COLLECTION:
            self._execute_collection(decision.target_point)
            
        elif decision.decision_type == DecisionType.GO_TO_DISPOSAL:
            self._execute_disposal_trip()
            
        elif decision.decision_type == DecisionType.RETURN_TO_DEPOT:
            self._execute_return_to_depot()
            
        elif decision.decision_type == DecisionType.REFUEL:
            self._execute_refueling()
            
        elif decision.decision_type == DecisionType.END_SHIFT:
            self._log("Fin de jornada - Regresando al depósito")
            self._execute_return_to_depot()
            
        elif decision.decision_type == DecisionType.SKIP_POINT:
            if decision.target_point in self.current_state.remaining_points:
                self.current_state.remaining_points.remove(decision.target_point)
                self._log(f"Punto omitido: {decision.target_point.address}")
    
    def _execute_collection(self, point: CollectionPoint):
        """Ejecuta la recolección de un punto específico"""
        truck = self.current_state.truck
        
        if not point or point not in self.current_state.remaining_points:
            self._log("Error: Punto no válido para recolección")
            return
        
        # Viajar al punto
        travel_distance = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            point.latitude, point.longitude
        )
        
        # Consumir combustible por el viaje
        fuel_consumed = truck.consume_fuel(travel_distance, is_highway=False)
        self._log(f"Viaje a {point.address}: {travel_distance:.2f} km, {fuel_consumed:.2f} L")
        
        # Mover camión al punto
        truck.move_to(point.latitude, point.longitude)
        truck.status = TruckStatus.COLLECTING
        
        # Ejecutar recolección
        collection_result = truck.collect_point(point)
        
        if collection_result.get('success'):
            self._log(f"Recolección exitosa: {collection_result['containers_collected']} contenedores, "
                     f"{collection_result['weight_collected_kg']:.1f} kg")
            
            # Mover punto a completados
            self.current_state.remaining_points.remove(point)
            self.current_state.completed_points.append(point)
            
            # Verificar si necesita ir al vertedero
            if truck.is_nearly_full:
                self._log(f"Camión casi lleno ({truck.cargo_percentage:.1f}%) - Considerando ir al vertedero")
        else:
            self._log(f"Error en recolección: {collection_result.get('reason', 'Desconocido')}")
        
        truck.status = TruckStatus.TRAVELING
    
    def _execute_disposal_trip(self):
        """Ejecuta un viaje al vertedero para vaciar la carga"""
        truck = self.current_state.truck
        disposal_location = INFRASTRUCTURE['disposal_sites'][0]
        
        self._log(f"Iniciando viaje al vertedero - Carga actual: {truck.cargo_percentage:.1f}%")
        
        # Viajar al vertedero
        travel_distance = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            disposal_location['latitude'], disposal_location['longitude']
        )
        
        fuel_consumed = truck.consume_fuel(travel_distance, is_highway=True)
        self._log(f"Viaje al vertedero: {travel_distance:.2f} km, {fuel_consumed:.2f} L")
        
        # Mover camión al vertedero
        truck.move_to(disposal_location['latitude'], disposal_location['longitude'])
        truck.status = TruckStatus.AT_DISPOSAL
        
        # Tiempo de descarga
        self.current_state.current_time += timedelta(minutes=TRUCK_CONFIG['disposal_time_min'])
        
        # Vaciar camión
        weight_disposed = truck.current_cargo_weight_kg
        truck.empty_cargo()
        self._log(f"Carga vaciada: {weight_disposed:.1f} kg dispuestos")
        
        # Incrementar contador de viajes al vertedero
        self.current_state.total_trips_to_disposal += 1
        
        truck.status = TruckStatus.TRAVELING
    
    def _execute_return_to_depot(self):
        """Ejecuta el regreso al depósito"""
        truck = self.current_state.truck
        depot_location = (
            INFRASTRUCTURE['depot']['latitude'],
            INFRASTRUCTURE['depot']['longitude']
        )
        
        travel_distance = self.physics.calculate_distance(
            truck.current_latitude, truck.current_longitude,
            depot_location[0], depot_location[1]
        )
        
        fuel_consumed = truck.consume_fuel(travel_distance, is_highway=False)
        self._log(f"Regreso al depósito: {travel_distance:.2f} km, {fuel_consumed:.2f} L")
        
        truck.move_to(depot_location[0], depot_location[1])
        truck.status = TruckStatus.IDLE
        
        self._log("Camión de regreso en el depósito")
    
    def _execute_refueling(self):
        """Ejecuta la recarga de combustible"""
        truck = self.current_state.truck
        fuel_before = truck.current_fuel_l
        
        truck.refuel()
        fuel_added = truck.current_fuel_l - fuel_before
        
        self._log(f"Reabastecimiento: {fuel_added:.1f} L agregados")
        
        # Tiempo de reabastecimiento
        self.current_state.current_time += timedelta(minutes=15)
    
    def _record_truck_state(self):
        """Registra el estado actual del camión para análisis"""
        truck = self.current_state.truck
        
        state_record = {
            'timestamp': self.current_state.current_time.isoformat(),
            'fuel_percentage': truck.fuel_percentage,
            'cargo_percentage': truck.cargo_percentage,
            'weight_percentage': truck.weight_percentage,
            'containers_collected': truck.containers_collected,
            'total_distance_km': truck.total_distance_km,
            'fuel_consumed_l': truck.total_fuel_consumed_l,
            'latitude': truck.current_latitude,
            'longitude': truck.current_longitude,
            'status': truck.status.value,
            'remaining_points': len(self.current_state.remaining_points)
        }
        
        self.truck_history.append(state_record)
    
    def _calculate_final_metrics(self, start_execution_time: datetime) -> Dict[str, Any]:
        """Calcula las métricas finales de la simulación"""
        end_execution_time = datetime.now()
        
        metrics = self.metrics_calculator.calculate_all_metrics(
            self.current_state.truck,
            self.current_state.completed_points,
            self.current_state,
            self.config.start_time,
            self.current_state.current_time
        )
        
        # Agregar métricas adicionales específicas de la simulación
        metrics['simulation_info'] = {
            'start_time': self.config.start_time.isoformat(),
            'end_time': self.current_state.current_time.isoformat(),
            'execution_time_seconds': (end_execution_time - start_execution_time).total_seconds(),
            'steps_executed': len(self.truck_history),
            'events_occurred': len(self.event_manager.event_history),
            'trips_to_disposal': self.current_state.total_trips_to_disposal,
            'final_fuel_percentage': self.current_state.truck.fuel_percentage,
            'points_completed': len(self.current_state.completed_points),
            'points_remaining': len(self.current_state.remaining_points)
        }
        
        return metrics
    
    def _log(self, message: str):
        """Registra un mensaje en el log de ejecución"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        self.execution_log.append(log_entry)
        self.logger.info(message)
    
    def create_visualization_dashboard(self, results: SimulationResults):
        """Crea un dashboard de visualización de los resultados"""
        dashboard_figs = {}
        
        # Timeline del camión
        dashboard_figs['truck_timeline'] = self.visualizer.create_truck_status_timeline(
            results.truck_history,
            results.events_occurred
        )
        
        # Dashboard de rendimiento
        if 'operational' in results.metrics:
            dashboard_figs['performance'] = self.visualizer.create_route_performance_dashboard(
                results.metrics['operational'],
                results.metrics['environmental'],
                results.metrics['economic'],
                results.metrics['quality']
            )
        
        # Mapa de recolección
        dashboard_figs['collection_map'] = self.visualizer.create_collection_points_map(
            results.completed_points,
            truck_path=self._extract_truck_path(results.truck_history)
        )
        
        # Análisis de combustible
        dashboard_figs['fuel_analysis'] = self.visualizer.create_fuel_consumption_analysis(
            results.truck_history
        )
        
        return dashboard_figs
    
    def _extract_truck_path(self, truck_history: List[Dict]) -> List[Tuple[float, float]]:
        """Extrae la ruta del camión del historial"""
        path = []
        for record in truck_history:
            if 'latitude' in record and 'longitude' in record:
                path.append((record['latitude'], record['longitude']))
        return path
    
    def run_scenario_comparison(
        self,
        collection_points: List[CollectionPoint],
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, SimulationResults]:
        """
        Ejecuta múltiples escenarios y compara los resultados.
        
        Args:
            collection_points: Puntos de recolección base
            scenarios: Lista de configuraciones de escenario
            
        Returns:
            Diccionario con los resultados de cada escenario
        """
        results = {}
        
        for scenario in scenarios:
            self.logger.info(f"Ejecutando escenario: {scenario['name']}")
            
            # Crear configuración específica del escenario
            scenario_config = SimulationConfig()
            for key, value in scenario.get('config', {}).items():
                setattr(scenario_config, key, value)
            
            # Crear nueva instancia del simulador para el escenario
            scenario_simulator = WasteCollectionSimulator(scenario_config)
            
            # Ejecutar simulación
            scenario_results = scenario_simulator.run_simulation(
                collection_points,
                scenario.get('initial_truck_state')
            )
            
            results[scenario['name']] = scenario_results
        
        return results


# Funciones de utilidad para crear simulaciones pre-configuradas

def create_standard_simulation(
    collection_points: List[CollectionPoint],
    min_fill_threshold: float = 70.0,
    enable_events: bool = True,
    weather: str = "normal"
) -> WasteCollectionSimulator:
    """Crea una simulación con configuración estándar"""
    config = SimulationConfig(
        min_fill_threshold=min_fill_threshold,
        enable_events=enable_events,
        weather_condition=weather
    )
    
    return WasteCollectionSimulator(config)


def create_stress_test_simulation(
    collection_points: List[CollectionPoint]
) -> WasteCollectionSimulator:
    """Crea una simulación de prueba de estrés con condiciones adversas"""
    config = SimulationConfig(
        min_fill_threshold=60.0,  # Umbral más bajo
        enable_events=True,
        weather_condition="rain",  # Condiciones climáticas adversas
        traffic_base_factor=1.5,   # Más tráfico
        time_step_minutes=2        # Mayor resolución temporal
    )
    
    return WasteCollectionSimulator(config)


def create_efficiency_test_simulation(
    collection_points: List[CollectionPoint]
) -> WasteCollectionSimulator:
    """Crea una simulación optimizada para eficiencia máxima"""
    config = SimulationConfig(
        min_fill_threshold=80.0,   # Solo contenedores muy llenos
        enable_events=False,       # Sin eventos aleatorios
        weather_condition="normal",
        traffic_base_factor=0.8,   # Menos tráfico
        time_step_minutes=10       # Pasos más grandes
    )
    
    return WasteCollectionSimulator(config)


# Ejemplo de uso del simulador
if __name__ == "__main__":
    # Este código se ejecutaría si se llama directamente al archivo
    print("Simulador de Recolección de Residuos")
    print("Importe este módulo para usar las funciones de simulación")
    
    # Ejemplo básico de cómo usar el simulador
    """
    from .models import CollectionPoint, Container
    
    # Crear puntos de ejemplo
    points = [
        CollectionPoint(
            id="POINT_001",
            address="Ejemplo 1",
            latitude=-33.4119,
            longitude=-70.5241,
            containers=[
                Container(
                    id="CONT_001",
                    capacity_liters=660,
                    current_fill_percentage=85.0,
                    latitude=-33.4119,
                    longitude=-70.5241,
                    address="Ejemplo 1"
                )
            ]
        )
    ]
    
    # Crear y ejecutar simulación
    simulator = create_standard_simulation(points)
    results = simulator.run_simulation(points)
    
    print(f"Simulación completada: {results.success}")
    print(f"Contenedores recolectados: {results.truck_final_state.containers_collected}")
    """