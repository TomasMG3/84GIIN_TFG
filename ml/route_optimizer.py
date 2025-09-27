# route_optimizer.py - Optimizador completo con OpenRouteService y Algoritmo Genético

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
import openrouteservice
from openrouteservice import convert
import polyline

@dataclass
class Container:
    id: int
    lat: float
    lon: float
    fill_percentage: float
    priority: float = 1.0
    capacity: float = 1000.0  # Capacidad en litros
    
@dataclass
class Vehicle:
    id: str
    capacity_kg: float
    current_lat: float
    current_lon: float
    fuel_efficiency: float = 0.35  # L/km
    speed_kmh: float = 25.0  # km/h promedio en ciudad

class RouteOptimizer:
    def __init__(self, api_key: Optional[str] = None):
        # Obtener API key desde variable de entorno o parámetro
        self.api_key = api_key or os.getenv('OPENROUTESERVICE_API_KEY')
        if not self.api_key:
            print("Warning: OpenRouteService API key is not configured. Using fallback methods.")
        
        self.base_url = "https://api.openrouteservice.org"
        self.depot_lat = -33.4166986
        self.depot_lon = -70.5179074
        self.max_locations = 50  # Límite de OpenRouteService gratuito
        
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
    
    def is_openrouteservice_available(self) -> bool:
        """Verifica si OpenRouteService está disponible"""
        if not self.api_key:
            return False
        
        try:
            url = f"{self.base_url}/v2/matrix/driving-car"
            headers = {
                'Accept': 'application/json',
                'Authorization': self.api_key,
                'Content-Type': 'application/json; charset=utf-8'
            }
            body = {
                "locations": [[self.depot_lon, self.depot_lat], [self.depot_lon + 0.001, self.depot_lat + 0.001]],
                "metrics": ["distance"]
            }
            
            response = requests.post(url, json=body, headers=headers, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_distance_matrix(self, containers: List[Container]) -> Optional[List[List[float]]]:
        """Obtiene matriz de distancias usando OpenRouteService Matrix API"""
        
        if not self.api_key:
            return None
        
        # Preparar coordenadas [lon, lat] (OpenRouteService usa este formato)
        locations = [[self.depot_lon, self.depot_lat]]  # Depot primero
        locations.extend([[c.lon, c.lat] for c in containers])
        
        # Verificar límite
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
            "metrics": ["distance", "duration"],
            "units": "km"
        }
        
        try:
            response = requests.post(url, json=body, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'distances' in data:
                    return data['distances']
                else:
                    print(f"OpenRouteService no devolvió 'distances': {data}")
                    return None
            else:
                print(f"Error OpenRouteService Matrix: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión con OpenRouteService: {e}")
            return None
    
    def get_route_geometry(self, containers: List[Container], route_order: List[int]) -> Optional[List[List[float]]]:
        # 1) Construye lista de coordenadas con depósito al inicio y al final
        coords = [[self.depot_lon, self.depot_lat]]  # depot como [lon, lat]
        for idx in route_order:
            c = containers[idx]
            coords.append([c.lon, c.lat])
        coords.append([self.depot_lon, self.depot_lat])  # regreso al depot

        # 2) Llama a ORS pidiendo geojson directamente
        url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
        headers = {
            "Authorization": "5b3ce3597851110001cf6248ec92228b5c8548f2a2ef9b6c05322733",
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
            feat = resp.json()["features"][0]["geometry"]
            if feat["type"] == "LineString":
                # devuelve lista de [lon, lat] ya en el orden que espera tu map
                return feat["coordinates"]
        except Exception as e:
            print("Error al obtener geometría ORS:", e)
        return None

    
    def optimize_route(self, containers: List[Container]) -> Dict:
        """Optimiza ruta usando OpenRouteService con fallback a Haversine"""
        
        if not containers:
            return {
                "routes": [],
                "total_distance": 0.0,
                "containers_count": 0,
                "optimization_method": "openrouteservice"
            }
        
        # Intentar obtener matriz de distancias real
        distance_matrix = self.get_distance_matrix(containers)
        
        if not distance_matrix:
            raise RuntimeError("No se pudo obtener la matriz de distancias desde OpenRouteService. Verifica tu API Key y conexión.")
            
        # Usar distancias reales para optimización
        route_order = self._nearest_neighbor_with_matrix(containers, distance_matrix)
        total_distance = self._calculate_route_distance_from_matrix(route_order, distance_matrix)
        optimization_method = "openrouteservice_matrix"
        
        if len(containers) + 2 <= self.max_locations:  # +2 for depot start/end
            route_geometry = self.get_route_geometry(containers, route_order)
        else:
            print(f"Too many locations ({len(containers)}), skipping detailed geometry")
            route_geometry = None

        
        # Construir respuesta
        route_containers = []
        for i, container_idx in enumerate(route_order, 1):
            container = containers[container_idx - 1]
            
            # Calcular distancia desde el punto anterior
            if i == 1:
                # Distancia desde depot
                distance_from_previous = self.calculate_distance(
                    self.depot_lat, self.depot_lon, container.lat, container.lon
                )
            else:
                prev_container = containers[route_order[i-2] - 1]
                distance_from_previous = self.calculate_distance(
                    prev_container.lat, prev_container.lon, container.lat, container.lon
                )
            
            route_containers.append({
                "container_id": container.id,
                "latitude": container.lat,
                "longitude": container.lon,
                "fill_percentage": container.fill_percentage,
                "distance_from_previous": round(distance_from_previous, 2),
                "order": i
            })
        
        # Estimaciones
        estimated_time = int(total_distance * 2.5 + len(route_containers))
        fuel_consumption = total_distance * 0.35
        co2_emissions = fuel_consumption * 2.6
        
        route_data = {
            "route_id": 1,
            "containers": route_containers,
            "total_distance_km": round(total_distance, 2),
            "estimated_time_minutes": estimated_time,
            "fuel_consumption_liters": round(fuel_consumption, 2),
            "co2_emissions_kg": round(co2_emissions, 2),
            "depot_location": {"lat": self.depot_lat, "lon": self.depot_lon}
        }
        
        # Agregar geometría si está disponible
        if route_geometry:
            route_data["route_coordinates"] = route_geometry
        
        return {
            "routes": [route_data],
            "total_distance": round(total_distance, 2),
            "containers_count": len(containers),
            "optimization_method": optimization_method
        }
    
    def genetic_algorithm_vrp(self, containers: List[Container], vehicles: List[Vehicle], 
                            population_size: int = 30, generations: int = 50) -> Dict:
        """
        Algoritmo genético para Vehicle Routing Problem (VRP)
        
        Args:
            containers: Lista de contenedores
            vehicles: Lista de vehículos disponibles
            population_size: Tamaño de la población
            generations: Número de generaciones
        
        Returns:
            Diccionario con las rutas optimizadas
        """
        
        if not containers:
            return {
                "routes": [],
                "total_distance": 0.0,
                "containers_count": 0,
                "optimization_method": "genetic_algorithm"
            }
        
        # Obtener matriz de distancias
        distance_matrix = self.get_distance_matrix(containers)
        if not distance_matrix:
            # Fallback a matriz Haversine
            distance_matrix = self._create_haversine_matrix(containers)
        
        # Inicializar población
        population = self._initialize_population(containers, vehicles, population_size)
        
        best_solution = None
        best_fitness = float('inf')
        
        print(f"Iniciando algoritmo genético: {population_size} individuos, {generations} generaciones")
        
        for generation in range(generations):
            # Evaluar fitness de cada individuo
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, distance_matrix, vehicles)
                fitness_scores.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()
            
            # Selección, cruzamiento y mutación
            new_population = []
            
            # Elitismo: mantener los mejores individuos
            elite_count = max(1, population_size // 10)
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:elite_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generar nueva población
            while len(new_population) < population_size:
                # Selección por torneo
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Cruzamiento
                child1, child2 = self._crossover(parent1, parent2, containers)
                
                # Mutación
                if random.random() < 0.1:  # 10% probabilidad de mutación
                    child1 = self._mutate(child1, containers)
                if random.random() < 0.1:
                    child2 = self._mutate(child2, containers)
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
            
            if generation % 10 == 0:
                print(f"Generación {generation}: Mejor fitness = {best_fitness:.2f}")
        
        print(f"Algoritmo genético completado. Mejor fitness: {best_fitness:.2f}")
        
        if best_solution is None:
            raise ValueError("No se encontró una solución válida en el algoritmo genético.")
        
        # Convertir mejor solución a formato de respuesta
        return self._solution_to_response(best_solution, containers, vehicles, distance_matrix)
    
    def _create_haversine_matrix(self, containers: List[Container]) -> List[List[float]]:
        """Crea matriz de distancias usando fórmula Haversine"""
        n = len(containers) + 1  # +1 para depot
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # Distancias desde/hacia depot (índice 0)
        for i, container in enumerate(containers, 1):
            dist = self.calculate_distance(self.depot_lat, self.depot_lon, container.lat, container.lon)
            matrix[0][i] = dist
            matrix[i][0] = dist
        
        # Distancias entre contenedores
        for i, container1 in enumerate(containers, 1):
            for j, container2 in enumerate(containers, 1):
                if i != j:
                    dist = self.calculate_distance(container1.lat, container1.lon, container2.lat, container2.lon)
                    matrix[i][j] = dist
        
        return matrix
    
    def _initialize_population(self, containers: List[Container], vehicles: List[Vehicle], 
                            population_size: int) -> List[List[List[int]]]:
        """Inicializa población para algoritmo genético"""
        population = []
        container_indices = list(range(1, len(containers) + 1))  # 1-indexed
        
        for _ in range(population_size):
            # Crear solución aleatoria
            shuffled_containers = container_indices.copy()
            random.shuffle(shuffled_containers)
            
            # Dividir contenedores entre vehículos
            individual = [[] for _ in vehicles]
            containers_per_vehicle = len(shuffled_containers) // len(vehicles)
            
            for i, vehicle in enumerate(vehicles):
                start_idx = i * containers_per_vehicle
                if i == len(vehicles) - 1:  # Último vehículo toma los restantes
                    individual[i] = shuffled_containers[start_idx:]
                else:
                    individual[i] = shuffled_containers[start_idx:start_idx + containers_per_vehicle]
            
            population.append(individual)
        
        return population
    
    def _evaluate_fitness(self, individual: List[List[int]], distance_matrix: List[List[float]], 
                         vehicles: List[Vehicle]) -> float:
        """Evalúa fitness de un individuo (menor es mejor)"""
        total_distance = 0.0
        
        for vehicle_idx, route in enumerate(individual):
            if not route:
                continue
            
            # Distancia desde depot al primer contenedor
            total_distance += distance_matrix[0][route[0]]
            
            # Distancias entre contenedores consecutivos
            for i in range(len(route) - 1):
                total_distance += distance_matrix[route[i]][route[i + 1]]
            
            # Distancia desde último contenedor al depot
            total_distance += distance_matrix[route[-1]][0]
        
        return total_distance
    
    def _tournament_selection(self, population: List[List[List[int]]], fitness_scores: List[float], 
                            tournament_size: int = 3) -> List[List[int]]:
        """Selección por torneo"""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _crossover(self, parent1: List[List[int]], parent2: List[List[int]], 
                  containers: List[Container]) -> Tuple[List[List[int]], List[List[int]]]:
        """Cruzamiento entre dos padres"""
        # Cruzamiento simple: intercambiar rutas aleatorias
        child1 = [route.copy() for route in parent1]
        child2 = [route.copy() for route in parent2]
        
        if len(child1) > 1 and len(child2) > 1:
            # Intercambiar dos rutas aleatorias
            idx1, idx2 = random.sample(range(len(child1)), 2)
            child1[idx1], child2[idx1] = child2[idx1].copy(), child1[idx1].copy()
            child1[idx2], child2[idx2] = child2[idx2].copy(), child1[idx2].copy()
        
        return child1, child2
    
    def _mutate(self, individual: List[List[int]], containers: List[Container]) -> List[List[int]]:
        """Mutación de un individuo"""
        mutated = [route.copy() for route in individual]
        
        # Mutación: mover un contenedor aleatorio a otra ruta
        non_empty_routes = [i for i, route in enumerate(mutated) if route]
        
        if len(non_empty_routes) >= 2:
            # Seleccionar ruta origen y destino
            source_route_idx = random.choice(non_empty_routes)
            target_route_idx = random.choice(range(len(mutated)))
            
            if mutated[source_route_idx]:
                # Mover contenedor
                container_to_move = mutated[source_route_idx].pop(random.randint(0, len(mutated[source_route_idx]) - 1))
                mutated[target_route_idx].append(container_to_move)
        
        return mutated
    
    def _solution_to_response(self, solution: List[List[int]], containers: List[Container], 
                            vehicles: List[Vehicle], distance_matrix: List[List[float]]) -> Dict:
        """Convierte solución del AG a formato de respuesta"""
        routes = []
        total_distance = 0.0
        
        for vehicle_idx, route in enumerate(solution):
            if not route:
                continue
            
            vehicle = vehicles[vehicle_idx]
            route_distance = 0.0
            route_containers = []
            
            # Calcular distancia de la ruta
            route_distance += distance_matrix[0][route[0]]  # Depot a primer contenedor
            
            for i, container_idx in enumerate(route, 1):
                container = containers[container_idx - 1]
                
                # Distancia desde punto anterior
                if i == 1:
                    distance_from_previous = distance_matrix[0][container_idx]
                else:
                    distance_from_previous = distance_matrix[route[i-2]][container_idx]
                
                route_containers.append({
                    "container_id": container.id,
                    "latitude": container.lat,
                    "longitude": container.lon,
                    "fill_percentage": container.fill_percentage,
                    "distance_from_previous": round(distance_from_previous, 2),
                    "order": i
                })
                
                # Sumar distancia entre contenedores consecutivos
                if i < len(route):
                    route_distance += distance_matrix[container_idx][route[i]]
            
            # Distancia de regreso al depot
            route_distance += distance_matrix[route[-1]][0]
            
            # Obtener geometría de ruta
            route_geometry = self.get_route_geometry(containers, route)
            
            # Calcular estimaciones
            estimated_time = int(route_distance * 2.5 + len(route_containers) * 5)
            fuel_consumption = route_distance * 0.35
            co2_emissions = fuel_consumption * 2.6
            
            route_data = {
                "route_id": vehicle_idx + 1,
                "vehicle_id": vehicle.id,
                "containers": route_containers,
                "total_distance_km": round(route_distance, 2),
                "estimated_time_minutes": estimated_time,
                "fuel_consumption_liters": round(fuel_consumption, 2),
                "co2_emissions_kg": round(co2_emissions, 2),
                "depot_location": {"lat": self.depot_lat, "lon": self.depot_lon}
            }
            
            if route_geometry:
                route_data["route_coordinates"] = route_geometry
            
            routes.append(route_data)
            total_distance += route_distance
        
        return {
            "routes": routes,
            "total_distance": round(total_distance, 2),
            "containers_count": len(containers),
            "optimization_method": "genetic_algorithm_openrouteservice"
        }
    
    def _nearest_neighbor_with_matrix(self, containers, distance_matrix):
        n = len(containers)
        unvisited = list(range(n))  # 0-based indices
        route_order = []
        current_pos = 0  # empezamos en el depósito

        while unvisited:
            nearest_idx = min(
                unvisited,
                key=lambda x: distance_matrix[current_pos][x] / containers[x].priority
            )
            route_order.append(nearest_idx)
            current_pos = nearest_idx
            unvisited.remove(nearest_idx)

        return route_order

    
    def _calculate_route_distance_from_matrix(self, route_order: List[int], 
                                            distance_matrix: List[List[float]]) -> float:
        """Calcula distancia total usando matriz"""
        total_distance = 0.0
        current_pos = 0  # Depot
        
        for container_idx in route_order:
            total_distance += distance_matrix[current_pos][container_idx]
            current_pos = container_idx
        
        # Regreso al depot
        total_distance += distance_matrix[current_pos][0]
        
        return total_distance
    
    def _nearest_neighbor_haversine(self, containers: List[Container]) -> Tuple[List[int], float]:
        """Fallback usando distancias Haversine"""
        unvisited = containers.copy()
        route_order = []
        total_distance = 0.0
        current_lat, current_lon = self.depot_lat, self.depot_lon
        
        while unvisited:
            nearest_container = None
            nearest_idx = None
            min_distance = float('inf')
            
            for i, container in enumerate(unvisited):
                distance = self.calculate_distance(current_lat, current_lon, container.lat, container.lon)
                weighted_distance = distance / container.priority
                
                if weighted_distance < min_distance:
                    min_distance = weighted_distance
                    nearest_container = container
                    nearest_idx = i
            
            if nearest_container:
                actual_distance = self.calculate_distance(current_lat, current_lon, 
                                                        nearest_container.lat, nearest_container.lon)
                total_distance += actual_distance
                
                # Encontrar índice original
                original_idx = containers.index(nearest_container) + 1
                route_order.append(original_idx)
                
                current_lat, current_lon = nearest_container.lat, nearest_container.lon
                unvisited.remove(nearest_container)
        
        # Regreso al depot
        if route_order:
            last_container = containers[route_order[-1] - 1]
            return_distance = self.calculate_distance(last_container.lat, last_container.lon, 
                                                    self.depot_lat, self.depot_lon)
            total_distance += return_distance
        
        return route_order, total_distance


def compare_optimization_methods(self, containers: List[Container], n_runs=10):
    results = []
    for _ in range(n_runs):
        # Probar diferentes métodos
        nn_result = self.simple_route_optimization(containers)
        ga_result = self.genetic_algorithm_vrp(containers, [Vehicle("default", 5000, self.depot_lat, self.depot_lon)])
        
        results.append({
            "method": ["Nearest Neighbor", "Genetic Algorithm"],
            "distance": [nn_result["total_distance"], ga_result["total_distance"]],
            "time": [nn_result["routes"][0]["estimated_time_minutes"], 
                    ga_result["routes"][0]["estimated_time_minutes"]]
        })
    
    return pd.DataFrame(results).groupby("method").mean()


class HybridRouteOptimizer(RouteOptimizer):
    """Optimizador híbrido que combina LSTM + Optimización espacial"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.lstm_predictor = lstm_predictor
    
    def calculate_dynamic_priority(self, container: Container, 
                                 historical_data: pd.DataFrame) -> float:
        """Calcular prioridad dinámica usando LSTM"""
        if not self.lstm_predictor.is_loaded:
            # Fallback a prioridad basada en llenado actual
            return min(2.0, 1.0 + (container.fill_percentage / 100))
        
        try:
            # Obtener predicción LSTM
            prediction = self.lstm_predictor.predict_fill_trend(
                historical_data, container.id
            )
            
            if "error" in prediction:
                # Fallback si hay error en LSTM
                base_priority = 1.0 + (container.fill_percentage / 50)
                return min(3.0, base_priority)
            
            # Calcular prioridad basada en predicción LSTM
            predicted_fill = prediction["predicted_fill_24h"]
            fill_rate = prediction["hourly_fill_rate"]
            confidence = prediction["confidence"]
            
            # Fórmula de prioridad dinámica
            time_to_critical = (95 - container.fill_percentage) / fill_rate if fill_rate > 0 else 48
            urgency_factor = max(0.1, 24 / (time_to_critical + 1))
            
            dynamic_priority = (container.fill_percentage / 30) + urgency_factor
            return min(4.0, max(1.0, dynamic_priority * confidence))
            
        except Exception as e:
            print(f"Error calculando prioridad dinámica: {e}")
            return 1.5  # Prioridad por defecto
    
    def optimize_with_lstm(self, containers: List[Container], 
                          historical_data: pd.DataFrame,
                          vehicles: List[Vehicle],
                          use_genetic_algorithm: bool = True) -> Dict:
        """Optimización híbrida LSTM + Routing"""
        
        # Actualizar prioridades usando LSTM
        containers_with_dynamic_priority = []
        for container in containers:
            dynamic_priority = self.calculate_dynamic_priority(container, historical_data)
            updated_container = Container(
                id=container.id,
                lat=container.lat,
                lon=container.lon,
                fill_percentage=container.fill_percentage,
                priority=dynamic_priority,  # Prioridad dinámica
                capacity=container.capacity
            )
            containers_with_dynamic_priority.append(updated_container)
        
        # Ejecutar optimización normal con prioridades actualizadas
        if use_genetic_algorithm:
            return self.genetic_algorithm_vrp(
                containers_with_dynamic_priority, 
                vehicles
            )
        else:
            return self.optimize_route(containers_with_dynamic_priority)

def create_openrouteservice_optimizer() -> Optional[RouteOptimizer]:
    """Factory function para crear optimizador"""
    try:
        return RouteOptimizer()
    except Exception as e:
        print(f"Error al crear optimizador OpenRouteService: {e}")
        return None


# Ejemplo de uso y pruebas
if __name__ == "__main__":
    # Ejemplo de contenedores en Las Condes
    containers = [
        Container(1, -33.4100, -70.5200, 85.0, 1.0),
        Container(2, -33.4150, -70.5250, 92.0, 1.5),
        Container(3, -33.4080, -70.5180, 78.0, 1.0),
        Container(4, -33.4200, -70.5300, 88.0, 1.2),
        Container(5, -33.4050, -70.5150, 95.0, 1.8),
    ]
    
    # Crear vehículos
    vehicles = [
        Vehicle("TRUCK-001", 5000.0, -33.4119, -70.5241),
        Vehicle("TRUCK-002", 3000.0, -33.4119, -70.5241)
    ]
    
    # Crear optimizador
    optimizer = create_openrouteservice_optimizer()
    
    if optimizer:
        print("=== Prueba de Optimización Básica ===")
        result = optimizer.optimize_route(containers)
        print(json.dumps(result, indent=2))
        
        print("\n=== Prueba de Algoritmo Genético ===")
        ga_result = optimizer.genetic_algorithm_vrp(containers, vehicles, population_size=20, generations=30)
        print(json.dumps(ga_result, indent=2))
        
        print(f"\n=== Estado del Servicio ===")
        print(f"API Key configurada: {bool(optimizer.api_key)}")
        print(f"OpenRouteService disponible: {optimizer.is_openrouteservice_available()}")
    else:
        print("No se pudo crear el optimizador. Verifica la configuración.")