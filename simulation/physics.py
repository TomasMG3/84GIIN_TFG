# simulator/physics.py
"""
Módulo de cálculos físicos para la simulación de recolección de residuos.
Incluye consumo de combustible, emisiones, tiempos y distancias.
"""

import math
from typing import Tuple, Dict, Optional
from datetime import datetime
import numpy as np

class PhysicsCalculator:
    """Calculadora de física para la simulación"""
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calcula distancia entre dos puntos GPS usando la fórmula de Haversine.
        
        Args:
            lat1, lon1: Coordenadas del punto 1
            lat2, lon2: Coordenadas del punto 2
            
        Returns:
            Distancia en kilómetros
        """
        R = 6371  # Radio de la Tierra en km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    @staticmethod
    def calculate_fuel_consumption(
        distance_km: float,
        base_consumption: float,
        load_percentage: float,
        traffic_factor: float = 1.0,
        slope_factor: float = 1.0,
        weather_factor: float = 1.0
    ) -> float:
        """
        Calcula el consumo de combustible considerando múltiples factores.
        
        Args:
            distance_km: Distancia a recorrer
            base_consumption: Consumo base (L/km)
            load_percentage: Porcentaje de carga del camión (0-100)
            traffic_factor: Factor de tráfico (1.0 = normal, >1.0 = congestionado)
            slope_factor: Factor de pendiente (1.0 = plano, >1.0 = cuesta arriba)
            weather_factor: Factor climático (1.0 = normal, >1.0 = lluvia/viento)
            
        Returns:
            Consumo total en litros
        """
        # Factor de carga: más peso = más consumo (hasta 20% adicional)
        load_factor = 1.0 + (load_percentage / 100) * 0.20
        
        # Consumo ajustado
        adjusted_consumption = (base_consumption * 
                              load_factor * 
                              traffic_factor * 
                              slope_factor * 
                              weather_factor)
        
        return distance_km * adjusted_consumption
    
    @staticmethod
    def calculate_travel_time(
        distance_km: float,
        base_speed_kmh: float,
        traffic_factor: float = 1.0,
        weather_factor: float = 1.0,
        load_factor: float = 1.0
    ) -> float:
        """
        Calcula el tiempo de viaje considerando condiciones.
        
        Args:
            distance_km: Distancia a recorrer
            base_speed_kmh: Velocidad base
            traffic_factor: Factor de tráfico (>1 = más lento)
            weather_factor: Factor climático (>1 = más lento)
            load_factor: Factor de carga (>1 = más lento)
            
        Returns:
            Tiempo en minutos
        """
        # Velocidad efectiva
        effective_speed = base_speed_kmh / (traffic_factor * weather_factor * load_factor)
        
        # Evitar división por cero
        if effective_speed <= 0:
            effective_speed = 5  # Velocidad mínima de emergencia
        
        # Tiempo en horas * 60 para minutos
        return (distance_km / effective_speed) * 60
    
    @staticmethod
    def calculate_emissions(
        fuel_consumed_liters: float,
        co2_per_liter: float = 2.31
    ) -> float:
        """
        Calcula las emisiones de CO2 basadas en el combustible consumido.
        
        Args:
            fuel_consumed_liters: Combustible consumido en litros
            co2_per_liter: Factor de emisión de CO2 por litro (kg/L)
            
        Returns:
            Emisiones de CO2 en kg
        """
        return fuel_consumed_liters * co2_per_liter
    
    @staticmethod
    def calculate_collection_time(
        containers_count: int,
        base_time_per_container: float = 2.5,
        difficulty_factor: float = 1.0
    ) -> float:
        """
        Calcula el tiempo total de recolección para un punto.
        
        Args:
            containers_count: Número de contenedores a recolectar
            base_time_per_container: Tiempo base por contenedor (minutos)
            difficulty_factor: Factor de dificultad del acceso
            
        Returns:
            Tiempo total en minutos
        """
        return containers_count * base_time_per_container * difficulty_factor