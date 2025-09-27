import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple
import json

class LasCondesDataGenerator:
    def __init__(self):
        # Bounds de Las Condes
        self.bounds = {
            "lat_min": -33.4348,
            "lat_max": -33.3891,
            "lon_min": -70.5633,
            "lon_max": -70.4851
        }
        
        self.container_capacities = {
            'small': 240,   # Contenedores pequeños
            'medium': 340,  # Contenedores medianos  
            'large': 660    # Contenedores grandes
        }
        
        self.zone_container_distribution = {
            "CENTRO": {'small': 0.2, 'medium': 0.5, 'large': 0.3},
            "VITACURA_LIMITE": {'small': 0.1, 'medium': 0.4, 'large': 0.5},
            "KENNEDY": {'small': 0.3, 'medium': 0.5, 'large': 0.2},
            # ... otras zonas
        }
        
        # Zonas de Las Condes con diferentes densidades
        self.zones = {
            "CENTRO": {"lat": -33.4119, "lon": -70.5241, "density": 0.8},
            "VITACURA_LIMITE": {"lat": -33.3950, "lon": -70.5400, "density": 0.6},
            "KENNEDY": {"lat": -33.4200, "lon": -70.5100, "density": 0.9},
            "MANQUEHUE": {"lat": -33.4000, "lon": -70.5000, "density": 0.5},
            "TABANCURA": {"lat": -33.4300, "lon": -70.5300, "density": 0.7},
            "ESCUELA_MILITAR": {"lat": -33.4150, "lon": -70.5450, "density": 0.8}
        }
        
        # Patrones de llenado por tipo de área
        self.fill_patterns = {
            "residential": {"base_rate": 0.8, "weekend_multiplier": 1.2, "variance": 0.3},
            "commercial": {"base_rate": 1.5, "weekend_multiplier": 0.7, "variance": 0.4},
            "mixed": {"base_rate": 1.0, "weekend_multiplier": 1.0, "variance": 0.2}
        }
    
    def generate_container_locations(self, num_containers: int = 300) -> List[Dict]:
        """Genera ubicaciones realistas de contenedores en Las Condes"""
        containers = []
        
        for i in range(num_containers):
            # Seleccionar zona aleatoria basada en densidad
            zone_weights = [info["density"] for info in self.zones.values()]
            zone_name = np.random.choice(list(self.zones.keys()), p=np.array(zone_weights)/sum(zone_weights))
            zone_info = self.zones[zone_name]
            
            # Generar coordenadas cerca del centro de la zona
            lat_offset = np.random.normal(0, 0.005)  # ~500m std
            lon_offset = np.random.normal(0, 0.005)
            
            lat = np.clip(zone_info["lat"] + lat_offset, self.bounds["lat_min"], self.bounds["lat_max"])
            lon = np.clip(zone_info["lon"] + lon_offset, self.bounds["lon_min"], self.bounds["lon_max"])
            
            # Determinar tipo de área
            area_type = np.random.choice(
                ["residential", "commercial", "mixed"], 
                p=[0.5, 0.3, 0.2]
            )
            
            zone_dist = self.zone_container_distribution.get(zone_name, {'small': 0.33, 'medium': 0.33, 'large': 0.34})
            container_type = np.random.choice(
                list(zone_dist.keys()), 
                p=list(zone_dist.values())
            )
            capacity = self.container_capacities[container_type]
            
            current_level = np.random.uniform(0, capacity * 0.3)
            fill_percentage = (current_level / capacity) * 100
            
            containers.append({
                "container_id": f"LC-{i+1:04d}",
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "zone": zone_name,
                "area_type": area_type,
                "capacity": capacity,
                "container_type": container_type,
                "current_level": round(current_level, 2),
                "fill_percentage": round(fill_percentage, 2),
                "address": f"Calle {i+1}, Las Condes",
                "installation_date": datetime.now() - timedelta(days=np.random.randint(30, 365))
            })
        
        return containers
    
    def generate_historical_data(self, containers: List[Dict], days: int = 90) -> pd.DataFrame:
        """Genera datos históricos sintéticos pero realistas"""
        data = []
        
        for container in containers:
            area_pattern = self.fill_patterns[container["area_type"]]
            
            for day in range(days):
                date = datetime.now() - timedelta(days=days-day)
                
                # Generar múltiples lecturas por día (cada 4-6 horas)
                readings_per_day = np.random.randint(4, 7)
                daily_fill_increment = np.random.normal(
                    area_pattern["base_rate"] * 24 / readings_per_day, 
                    area_pattern["variance"]
                )
                
                # Ajustar por día de la semana
                if date.weekday() >= 5:  # Weekend
                    daily_fill_increment *= area_pattern["weekend_multiplier"]
                
                current_fill = 0.0
                last_empty_days = np.random.randint(1, 4)  # Vaciar cada 1-4 días
                
                if day % last_empty_days == 0:
                    current_fill = 0.0  # Contenedor vaciado
                
                for reading in range(readings_per_day):
                    timestamp = date + timedelta(hours=reading * (24/readings_per_day))
                    
                    # Incrementar nivel de llenado
                    fill_increment = max(0, np.random.normal(daily_fill_increment, 1))
                    current_fill = min(100, current_fill + fill_increment)
                    
                    # Temperatura simulada (Santiago patterns)
                    base_temp = 18 + 8 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                    temp_variation = np.random.normal(0, 3)
                    temperature = base_temp + temp_variation
                    
                    # Degradación de batería
                    days_since_install = (date - container["installation_date"]).days
                    battery_level = max(20, 100 - (days_since_install * 0.02) + np.random.normal(0, 2))
                    
                    data.append({
                        "container_id": container["container_id"],
                        "timestamp": timestamp,
                        "fill_percentage": round(current_fill, 2),
                        "fill_level": round((current_fill / 1000) * container["capacity"], 2),     # Revisar Porcentaje de llenado
                        "temperature": round(temperature, 1),
                        "battery_level": round(battery_level, 1),
                        "zone": container["zone"],
                        "area_type": container["area_type"],
                        "capacity": container["capacity"],
                        "latitude": container["latitude"],
                        "longitude": container["longitude"]
                    })
        
        return pd.DataFrame(data)
    
    def generate_traffic_patterns(self) -> Dict:
        """Genera patrones de tráfico para Las Condes"""
        return {
            "rush_hours": [
                {"start": 7, "end": 9, "multiplier": 1.8},
                {"start": 18, "end": 20, "multiplier": 1.6}
            ],
            "weekend_factor": 0.7,
            "base_speed_kmh": 25,
            "road_types": {
                "main_avenue": {"speed": 35, "congestion": 1.4},
                "secondary": {"speed": 25, "congestion": 1.1},
                "residential": {"speed": 20, "congestion": 1.0}
            }
        }
    
    def generate_collection_events(self, containers: List[Dict], days: int = 30) -> List[Dict]:
        """Genera eventos históricos de recolección"""
        events = []
        
        for container in containers:
            # Simular recolecciones cada 2-4 días
            collection_frequency = np.random.randint(2, 5)
            
            for day in range(0, days, collection_frequency):
                collection_date = datetime.now() - timedelta(days=days-day)
                
                # Volumen recolectado (70-95% de la capacidad)
                fill_at_collection = np.random.uniform(70, 95)
                volume_collected = (fill_at_collection / 100) * container["capacity"]
                
                # Tiempo de recolección (3-8 minutos por contenedor)
                collection_time = np.random.randint(3, 9)
                
                # Camión asignado
                truck_id = f"TRUCK-{np.random.randint(1, 6):02d}"
                
                events.append({
                    "container_id": container["container_id"],
                    "collection_date": collection_date,
                    "volume_collected": round(volume_collected, 2),
                    "collection_time_minutes": collection_time,
                    "truck_id": truck_id,
                    "pre_collection_fill": round(fill_at_collection, 2),
                    "zone": container["zone"]
                })
        
        return events
    
    def create_sample_dataset(self, num_containers: int = 200, history_days: int = 60) -> Dict:
        """Crea un dataset completo de muestra"""
        print(f"Generating {num_containers} containers...")
        containers = self.generate_container_locations(num_containers)
        
        print(f"Generating {history_days} days of historical data...")
        historical_data = self.generate_historical_data(containers, history_days)
        
        print("Generating collection events...")
        collection_events = self.generate_collection_events(containers, history_days)
        
        print("Generating traffic patterns...")
        traffic_patterns = self.generate_traffic_patterns()
        
        return {
            "containers": containers,
            "historical_data": historical_data,
            "collection_events": collection_events,
            "traffic_patterns": traffic_patterns,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_containers": num_containers,
                "history_days": history_days,
                "total_readings": len(historical_data),
                "total_collections": len(collection_events)
            }
        }
    
    def save_dataset(self, dataset: Dict, base_path: str = "data/"):
        """Guarda el dataset en archivos CSV y JSON"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Guardar contenedores
        pd.DataFrame(dataset["containers"]).to_csv(
            f"{base_path}/containers.csv", index=False
        )
        
        # Guardar datos históricos
        dataset["historical_data"].to_csv(
            f"{base_path}/historical_data.csv", index=False
        )
        
        # Guardar eventos de recolección
        pd.DataFrame(dataset["collection_events"]).to_csv(
            f"{base_path}/collection_events.csv", index=False
        )
        
        # Guardar patrones de tráfico
        with open(f"{base_path}/traffic_patterns.json", "w") as f:
            json.dump(dataset["traffic_patterns"], f, indent=2)
        
        # Guardar metadata
        with open(f"{base_path}/dataset_metadata.json", "w") as f:
            json.dump(dataset["metadata"], f, indent=2, default=str)
        
        print(f"Dataset saved to {base_path}")
        return True

# Función de utilidad para generar datos rápidamente
def create_las_condes_dataset(containers=200, days=60):
    """Función conveniente para crear dataset de Las Condes"""
    generator = LasCondesDataGenerator()
    dataset = generator.create_sample_dataset(containers, days)
    generator.save_dataset(dataset)
    return dataset

if __name__ == "__main__":
    # Generar dataset de ejemplo
    print("Generando dataset de ejemplo para Las Condes...")
    dataset = create_las_condes_dataset(300, 90)
    print("Dataset generado exitosamente!")