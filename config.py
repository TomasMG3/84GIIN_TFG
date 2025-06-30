import os
from typing import Optional

class Settings:
    DATABASE_URL: str = "sqlite:///./waste_management.db"
    SECRET_KEY: str = "tu-clave-secreta-para-tesis"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Configuración Las Condes
    LAS_CONDES_BOUNDS = {
        "lat_min": -33.4348,
        "lat_max": -33.3891,
        "lon_min": -70.5633,
        "lon_max": -70.4851
    }
    
    # Parámetros IoT simulados
    SENSOR_UPDATE_INTERVAL: int = 300  # 5 minutos
    MAX_CONTAINERS: int = 500
    BATTERY_LIFE_DAYS: int = 1825  # 5 años

settings = Settings()