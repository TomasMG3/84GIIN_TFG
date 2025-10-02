import requests
import json
import math
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os
from dataclasses import dataclass
from datetime import datetime
import polyline

@dataclass
class Container:
    id: int
    lat: float
    lon: float
    fill_percentage: float
    priority: float = 1.0
    capacity: float = 1000.0
    is_overflow: bool = False  # Nuevo campo para desbordamiento
    
@dataclass
class Vehicle:
    id: str
    capacity_kg: float
    current_lat: float
    current_lon: float
    fuel_efficiency: float = 0.35
    speed_kmh: float = 25.0

class RouteOptimizer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTESERVICE_API_KEY')
        if not self.api_key:
            print("Warning: OpenRouteService API key is not configured.")
        
        self.base_url = "https://api.openrouteservice.org"
        self.depot_lat = -33.4166986
        self.depot_lon = -70.5179074
        self.max_locations = 50
        
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula distancia haversine en km"""
        R = 6371
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def filter_unnecessary_trips(self, containers: List[Container], min_threshold: float = 70.0) -> Tuple[List[Container], List[Container]]:
        """
        Filtra contenedores según umbral y clasifica viajes innecesarios
        PRIORIZA contenedores con desbordamiento (>100%) si están cerca de la ruta
        
        Returns:
            Tuple de (contenedores_necesarios, contenedores_innecesarios)
        """
        necessary = []
        unnecessary = []
        overflow_containers = []
        
        for container in containers:
            # Clasificar por desbordamiento primero
            if container.fill_percentage > 100:
                container.is_overflow = True
                container.priority = 5.0  # Máxima prioridad - CRÍTICO
                overflow_containers.append(container)
                necessary.append(container)
            elif container.fill_percentage >= min_threshold:
                # Contenedores normales sobre umbral
                if container.fill_percentage >= 95:
                    container.priority = 3.0  # Muy alta prioridad
                elif container.fill_percentage >= 90:
                    container.priority = 2.5  # Alta prioridad
                elif container.fill_percentage >= 80:
                    container.priority = 2.0  # Prioridad media-alta
                else:
                    container.priority = 1.0  # Prioridad normal
                
                necessary.append(container)
            else:
                # Viaje innecesario - bajo umbral
                unnecessary.append(container)
        
        # Log para debugging
        print(f"✅ Contenedores necesarios: {len(necessary)} (Desbordamientos: {len(overflow_containers)})")
        print(f"❌ Viajes innecesarios evitados: {len(unnecessary)}")
        
        return necessary, unnecessary
    
    def optimize_route_with_priorities(self, containers: List[Container], min_threshold: float = 70.0) -> Dict:
        """
        Optimiza ruta considerando prioridades y omitiendo viajes innecesarios
        """
        if not containers:
            return {
                "routes": [],
                "total_distance": 0.0,
                "containers_count": 0,
                "unnecessary_trips_avoided": 0,
                "optimization_method": "priority_based"
            }
        
        # Filtrar viajes innecesarios
        necessary_containers, unnecessary_containers = self.filter_unnecessary_trips(
            containers, min_threshold
        )
        
        print(f"✅ Contenedores necesarios: {len(necessary_containers)}")
        print(f"❌ Viajes innecesarios evitados: {len(unnecessary_containers)}")
        
        if not necessary_containers:
            return {
                "routes": [],
                "total_distance": 0.0,
                "containers_count": 0,
                "unnecessary_trips_avoided": len(unnecessary_containers),
                "message": "No hay contenedores que requieran recolección",
                "optimization_method": "priority_based"
            }
        
        # Ordenar por prioridad (desbordamiento primero)
        sorted_containers = sorted(
            necessary_containers, 
            key=lambda c: (c.is_overflow, c.priority, c.fill_percentage), 
            reverse=True
        )
        
        # Obtener matriz de distancias
        distance_matrix = self.get_distance_matrix(sorted_containers)
        
        if not distance_matrix:
            print("⚠️ No se pudo obtener matriz de distancias, usando Haversine")
            distance_matrix = self._create_haversine_matrix(sorted_containers)
        
        # Aplicar algoritmo genético con prioridades
        route_order = self._genetic_algorithm_with_priorities(
            sorted_containers, 
            distance_matrix
        )
        
        # Calcular distancia total
        total_distance = self._calculate_route_distance_from_matrix(route_order, distance_matrix)
        
        # Construir respuesta con geometría
        route_containers = []
        for i, container_idx in enumerate(route_order, 1):
            container = sorted_containers[container_idx]
            
            if i == 1:
                distance_from_previous = self.calculate_distance(
                    self.depot_lat, self.depot_lon, container.lat, container.lon
                )
            else:
                prev_container = sorted_containers[route_order[i-2]]
                distance_from_previous = self.calculate_distance(
                    prev_container.lat, prev_container.lon, container.lat, container.lon
                )
            
            route_containers.append({
                "container_id": container.id,
                "latitude": container.lat,
                "longitude": container.lon,
                "fill_percentage": container.fill_percentage,
                "priority": container.priority,
                "is_overflow": container.is_overflow,
                "distance_from_previous": round(distance_from_previous, 2),
                "order": i
            })
        
        # Obtener geometría de ruta real
        route_geometry = None
        if len(sorted_containers) + 2 <= self.max_locations:
            route_geometry = self.get_route_geometry(sorted_containers, route_order)
        
        # Estimaciones
        estimated_time = int(total_distance * 2.5 + len(route_containers) * 5)
        fuel_consumption = total_distance * 0.35
        co2_emissions = fuel_consumption * 2.6
        
        route_data = {
            "route_id": 1,
            "containers": route_containers,
            "total_distance_km": round(total_distance, 2),
            "estimated_time_minutes": estimated_time,
            "fuel_consumption_liters": round(fuel_consumption, 2),
            "co2_emissions_kg": round(co2_emissions, 2),
            "depot_location": {"lat": self.depot_lat, "lon": self.depot_lon},
            "overflow_containers": sum(1 for c in route_containers if c["is_overflow"]),
            "high_priority_containers": sum(1 for c in route_containers if c["priority"] >= 2.0)
        }
        
        if route_geometry:
            route_data["route_coordinates"] = route_geometry
        
        return {
            "routes": [route_data],
            "total_distance": round(total_distance, 2),
            "containers_count": len(necessary_containers),
            "unnecessary_trips_avoided": len(unnecessary_containers),
            "optimization_method": "genetic_algorithm_with_priorities"
        }
    
    def _genetic_algorithm_with_priorities(self, containers: List[Container], 
                                          distance_matrix: List[List[float]],
                                          population_size: int = 50,
                                          generations: int = 100) -> List[int]:
        """
        Algoritmo genético optimizado con prioridades
        """
        n = len(containers)
        if n == 0:
            return []
        
        # Crear población inicial considerando prioridades
        population = []
        for _ in range(population_size):
            # Crear individuo con bias hacia contenedores de alta prioridad al inicio
            individual = list(range(n))
            
            # Ordenar parcialmente por prioridad
            high_priority_indices = [i for i, c in enumerate(containers) if c.is_overflow]
            medium_priority_indices = [i for i, c in enumerate(containers) if c.priority >= 2.0 and not c.is_overflow]
            normal_priority_indices = [i for i in range(n) if i not in high_priority_indices + medium_priority_indices]
            
            # Mezclar cada grupo
            random.shuffle(high_priority_indices)
            random.shuffle(medium_priority_indices)
            random.shuffle(normal_priority_indices)
            
            # Combinar con cierta aleatoriedad
            if random.random() < 0.7:  # 70% mantiene prioridades
                individual = high_priority_indices + medium_priority_indices + normal_priority_indices
            else:  # 30% completamente aleatorio
                random.shuffle(individual)
            
            population.append(individual)
        
        best_solution = None
        best_fitness = float('inf')
        
        for generation in range(generations):
            # Evaluar fitness
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness_with_priority(
                    individual, distance_matrix, containers
                )
                fitness_scores.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()
            
            # Selección y reproducción
            new_population = []
            
            # Elitismo
            elite_count = max(2, population_size // 10)
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:elite_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generar nueva población
            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population, fitness_scores, 3)
                parent2 = self._tournament_selection(population, fitness_scores, 3)
                
                child1, child2 = self._pmx_crossover(parent1, parent2)
                
                if random.random() < 0.15:
                    child1 = self._mutate_swap(child1)
                if random.random() < 0.15:
                    child2 = self._mutate_swap(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
            
            if generation % 20 == 0:
                print(f"Generación {generation}: Mejor fitness = {best_fitness:.2f}")
        
        return best_solution if best_solution else list(range(n))
    
    def _evaluate_fitness_with_priority(self, individual: List[int], 
                                       distance_matrix: List[List[float]],
                                       containers: List[Container]) -> float:
        """
        Evalúa fitness considerando distancia Y penalizaciones por prioridad
        """
        total_distance = 0.0
        priority_penalty = 0.0
        
        # Distancia desde depot al primer contenedor
        total_distance += distance_matrix[0][individual[0] + 1]
        
        # Distancias entre contenedores consecutivos
        for i in range(len(individual) - 1):
            total_distance += distance_matrix[individual[i] + 1][individual[i + 1] + 1]
        
        # Distancia desde último contenedor al depot
        total_distance += distance_matrix[individual[-1] + 1][0]
        
        # Penalización: contenedores de alta prioridad deben estar temprano en la ruta
        for position, container_idx in enumerate(individual):
            container = containers[container_idx]
            
            if container.is_overflow:
                # Penalización alta si desbordamiento está tarde en la ruta
                priority_penalty += position * 2.0
            elif container.priority >= 2.0:
                # Penalización media para alta prioridad
                priority_penalty += position * 0.5
        
        return total_distance + priority_penalty
    
    def _tournament_selection(self, population: List[List[int]], 
                             fitness_scores: List[float],
                             tournament_size: int = 3) -> List[int]:
        """Selección por torneo"""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _pmx_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Partially Mapped Crossover para permutaciones"""
        size = len(parent1)
        if size < 2:
            return parent1.copy(), parent2.copy()
        
        # Seleccionar puntos de corte
        cx_point1 = random.randint(0, size - 1)
        cx_point2 = random.randint(cx_point1 + 1, size)
        
        # Inicializar hijos
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copiar segmento medio
        child1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
        child2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]
        
        # Mapear el resto
        def fill_child(child, parent_source, parent_target, cx1, cx2):
            for i in range(size):
                if i >= cx1 and i < cx2:
                    continue
                
                value = parent_source[i]
                while value in child[cx1:cx2]:
                    idx = parent_source.index(value)
                    value = parent_target[idx]
                
                child[i] = value
        
        fill_child(child1, parent2, parent1, cx_point1, cx_point2)
        fill_child(child2, parent1, parent2, cx_point1, cx_point2)
        
        return child1, child2
    
    def _mutate_swap(self, individual: List[int]) -> List[int]:
        """Mutación por intercambio de dos elementos"""
        mutated = individual.copy()
        if len(mutated) > 1:
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    
    def get_distance_matrix(self, containers: List[Container]) -> Optional[List[List[float]]]:
        """Obtiene matriz de distancias usando OpenRouteService Matrix API"""
        if not self.api_key:
            return None
        
        locations = [[self.depot_lon, self.depot_lat]]
        locations.extend([[c.lon, c.lat] for c in containers])
        
        if len(locations) > self.max_locations:
            print(f"Demasiadas ubicaciones ({len(locations)}). Límite: {self.max_locations}")
            return None
        
        url = f"{self.base_url}/v2/matrix/driving-car"
        headers = {
            'Accept': 'application/json',
            'Authorization': self.api_key,
            'Content-Type': 'application/json; charset=utf-8'
        }
        body = {
            "locations": locations,
            "metrics": ["distance"],
            "units": "km"
        }
        
        try:
            response = requests.post(url, json=body, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data.get('distances')
            else:
                print(f"Error OpenRouteService: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error de conexión: {e}")
            return None
    
    def _create_haversine_matrix(self, containers: List[Container]) -> List[List[float]]:
        """Crea matriz de distancias usando Haversine"""
        n = len(containers) + 1
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i, container in enumerate(containers, 1):
            dist = self.calculate_distance(self.depot_lat, self.depot_lon, container.lat, container.lon)
            matrix[0][i] = dist
            matrix[i][0] = dist
        
        for i, container1 in enumerate(containers, 1):
            for j, container2 in enumerate(containers, 1):
                if i != j:
                    dist = self.calculate_distance(container1.lat, container1.lon, container2.lat, container2.lon)
                    matrix[i][j] = dist
        
        return matrix
    
    def _calculate_route_distance_from_matrix(self, route_order: List[int], 
                                            distance_matrix: List[List[float]]) -> float:
        """Calcula distancia total usando matriz"""
        total_distance = 0.0
        current_pos = 0
        
        for container_idx in route_order:
            total_distance += distance_matrix[current_pos][container_idx + 1]
            current_pos = container_idx + 1
        
        total_distance += distance_matrix[current_pos][0]
        
        return total_distance
    
    def get_route_geometry(self, containers: List[Container], route_order: List[int]) -> Optional[List[List[float]]]:
        """Obtiene geometría de ruta real"""
        if not self.api_key:
            return None
        
        coords = [[self.depot_lon, self.depot_lat]]
        
        for idx in route_order:
            c = containers[idx]
            coords.append([c.lon, c.lat])
        
        coords.append([self.depot_lon, self.depot_lat])
        
        url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }
        body = {
            "coordinates": coords,
            "geometry": True,
            "instructions": False,
        }
        
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            if "features" in data and len(data["features"]) > 0:
                feat = data["features"][0]["geometry"]
                if feat["type"] == "LineString":
                    return feat["coordinates"]
            return None
        except Exception as e:
            print(f"Error al obtener geometría: {e}")
            return None