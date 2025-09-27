# physics_simulator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import math

class WasteCollectionSimulator:
    def __init__(self):
        # Parámetros realistas del camión
        self.truck_fuel_capacity = 270  # Litros
        self.truck_compression_ratio = 3  # 3:1
        self.truck_max_load_kg = 10000  # 10 toneladas
        
        # Parámetros de combustible
        self.fuel_consumption_empty = 0.25  # L/km (camión vacío)
        self.fuel_consumption_loaded = 0.40  # L/km (camión lleno)
        
        # Parámetros de RSU (Residuos Sólidos Urbanos)
        self.rsu_density = 200  # kg/m³ (densidad promedio RSU compactado)
        self.rsu_moisture_content = 0.25  # 25% humedad
        
        # Tipos de contenedores (litros)
        self.container_types = {
            'small': 240,
            'medium': 340, 
            'large': 660
        }
        
        # Factores de emisión
        self.co2_per_liter_diesel = 2.68  # kg CO2/L diesel
        self.co2_per_kg_rsu = 0.5  # kg CO2/kg RSU (descomposición)
    
    def calculate_rsu_mass(self, volume_liters: float, compacted: bool = True) -> float:
        """Calcular masa de RSU considerando densidad y compactación"""
        volume_m3 = volume_liters / 1000
        
        if compacted:
            effective_density = self.rsu_density * self.truck_compression_ratio
        else:
            effective_density = self.rsu_density
            
        return volume_m3 * effective_density
    
    def simulate_container_overflow(self, container_capacity: float, current_fill: float, 
                                  hours_since_empty: int) -> Dict:
        """Simular desborde de contenedor basado en tiempo y llenado"""
        # Tasa de llenado variable por tipo de zona y hora
        base_fill_rate = 2.5  # Litros/hora base
        
        # Aumentar tasa en horas pico (mañana y tarde)
        current_hour = datetime.now().hour
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            fill_rate = base_fill_rate * 2.5
        else:
            fill_rate = base_fill_rate
        
        # Simular llenado adicional
        additional_fill = fill_rate * hours_since_empty
        new_fill = min(container_capacity, current_fill + additional_fill)
        
        # Calcular desborde
        overflow = max(0, (current_fill + additional_fill) - container_capacity)
        
        return {
            'new_fill_liters': new_fill,
            'overflow_liters': overflow,
            'fill_rate_used': fill_rate,
            'is_overflowing': overflow > 0
        }
    
    def calculate_fuel_consumption(self, distance_km: float, current_load_kg: float) -> float:
        """Calcular consumo de combustible basado en distancia y carga"""
        # Consumo variable según carga
        load_factor = current_load_kg / self.truck_max_load_kg
        consumption_rate = (self.fuel_consumption_empty * (1 - load_factor) + 
                          self.fuel_consumption_loaded * load_factor)
        
        return distance_km * consumption_rate
    
    def simulate_route_physics(self, route_data: Dict, containers_data: List[Dict]) -> Dict:
        """Simulación física completa de una ruta de recolección"""
        
        total_distance = route_data.get('total_distance_km', 0)
        route_containers = route_data.get('containers', [])
        
        # Variables de simulación
        current_fuel = self.truck_fuel_capacity
        current_load_kg = 0
        total_co2_emissions = 0
        total_rsu_collected = 0
        overflow_events = []
        
        # Simular cada parada en la ruta
        for i, stop in enumerate(route_containers):
            container_id = stop['container_id']
            container_info = next((c for c in containers_data if c['id'] == container_id), None)
            
            if not container_info:
                continue
            
            # Calcular distancia hasta esta parada
            distance_to_stop = stop.get('distance_from_previous', 0)
            
            # Consumo de combustible hasta la parada
            fuel_used = self.calculate_fuel_consumption(distance_to_stop, current_load_kg)
            current_fuel -= fuel_used
            
            # Emisiones de CO2 del tramo
            co2_emissions = fuel_used * self.co2_per_liter_diesel
            total_co2_emissions += co2_emissions
            
            # Simular recolección en esta parada
            if container_info['fill_percentage'] > 70:  # Solo recolectar si está suficientemente lleno
                fill_liters = (container_info['fill_percentage'] / 100) * container_info['capacity']
                rsu_mass = self.calculate_rsu_mass(fill_liters)
                
                # Verificar si el camión puede cargar más
                if current_load_kg + rsu_mass <= self.truck_max_load_kg:
                    current_load_kg += rsu_mass
                    total_rsu_collected += rsu_mass
                    
                    # Marcar contenedor como vaciado
                    container_info['fill_percentage'] = 0
                    container_info['current_level'] = 0
                else:
                    # Camión lleno, no puede recolectar más
                    overflow_events.append({
                        'container_id': container_id,
                        'reason': 'Truck full',
                        'missed_volume': fill_liters
                    })
            
            # Simular desbordes mientras el camión está en ruta
            for container in containers_data:
                if container['id'] != container_id:  # Otros contenedores siguen llenándose
                    overflow_result = self.simulate_container_overflow(
                        container['capacity'],
                        container['current_level'],
                        hours_since_empty=1  # Aprox 1 hora entre paradas
                    )
                    
                    if overflow_result['is_overflowing']:
                        overflow_events.append({
                            'container_id': container['id'],
                            'reason': 'Container overflow',
                            'overflow_liters': overflow_result['overflow_liters'],
                            'time_of_overflow': datetime.now()
                        })
                        
                    container['current_level'] = overflow_result['new_fill_liters']
                    container['fill_percentage'] = (container['current_level'] / container['capacity']) * 100
        
        # Consumo de regreso al depósito
        return_distance = total_distance * 0.1  # Estimado 10% extra por regreso
        return_fuel_used = self.calculate_fuel_consumption(return_distance, current_load_kg)
        current_fuel -= return_fuel_used
        
        return {
            'simulation_time': datetime.now().isoformat(),
            'initial_fuel_liters': self.truck_fuel_capacity,
            'final_fuel_liters': max(0, current_fuel),
            'fuel_used_liters': self.truck_fuel_capacity - max(0, current_fuel),
            'total_distance_km': total_distance + return_distance,
            'rsu_collected_kg': total_rsu_collected,
            'final_truck_load_kg': current_load_kg,
            'truck_load_percentage': (current_load_kg / self.truck_max_load_kg) * 100,
            'co2_emissions_kg': total_co2_emissions,
            'co2_saved_kg': total_rsu_collected * self.co2_per_kg_rsu,  # CO2 evitado por recolección
            'overflow_events': overflow_events,
            'overflow_containers': len([e for e in overflow_events if e['reason'] == 'Container overflow']),
            'missed_collections': len([e for e in overflow_events if e['reason'] == 'Truck full']),
            'efficiency_score': self.calculate_efficiency_score(
                total_rsu_collected, 
                total_distance + return_distance, 
                len(overflow_events)
            )
        }
    
    def calculate_efficiency_score(self, rsu_collected: float, distance: float, 
                                 problems: int) -> float:
        """Calcular puntaje de eficiencia de la ruta"""
        if distance == 0:
            return 0
            
        # Métrica principal: kg de RSU por km
        collection_efficiency = rsu_collected / distance
        
        # Penalizar por problemas
        problem_penalty = max(0, 1 - (problems * 0.1))
        
        # Normalizar a escala 0-100
        base_score = min(collection_efficiency * 10, 100)
        
        return max(0, base_score * problem_penalty)
    
    def generate_physical_metrics(self, route_data: Dict, containers: List[Dict]) -> Dict:
        """Generar métricas físicas para dashboard"""
        simulation = self.simulate_route_physics(route_data, containers)
        
        return {
            'fuel_metrics': {
                'initial_fuel': simulation['initial_fuel_liters'],
                'final_fuel': simulation['final_fuel_liters'],
                'fuel_used': simulation['fuel_used_liters'],
                'fuel_efficiency': simulation['fuel_used_liters'] / simulation['total_distance_km'] if simulation['total_distance_km'] > 0 else 0
            },
            'environmental_metrics': {
                'co2_emissions': simulation['co2_emissions_kg'],
                'co2_saved': simulation['co2_saved_kg'],
                'net_co2_impact': simulation['co2_saved_kg'] - simulation['co2_emissions_kg']
            },
            'operational_metrics': {
                'rsu_collected_kg': simulation['rsu_collected_kg'],
                'truck_load_percentage': simulation['truck_load_percentage'],
                'efficiency_score': simulation['efficiency_score'],
                'distance_per_kg': simulation['total_distance_km'] / simulation['rsu_collected_kg'] if simulation['rsu_collected_kg'] > 0 else 0
            },
            'problem_metrics': {
                'overflow_events': simulation['overflow_events'],
                'total_problems': simulation['overflow_containers'] + simulation['missed_collections'],
                'success_rate': (len(containers) - simulation['missed_collections']) / len(containers) * 100 if containers else 100
            }
        }

# Instancia global del simulador
physics_simulator = WasteCollectionSimulator()