# simulator/config.py
"""
Configuración y parámetros para el sistema de simulación de recolección de residuos.
"""

# ============== PARÁMETROS DEL CAMIÓN ==============
TRUCK_CONFIG = {
    # Combustible
    'fuel_capacity_l': 270.0,              # Capacidad máxima del tanque (litros)
    'fuel_consumption_urban_l_per_km': 0.50,  # Consumo en ciudad con paradas frecuentes
    'fuel_consumption_highway_l_per_km': 0.35,  # Consumo en autopista (si aplica)
    'fuel_warning_threshold_l': 50.0,      # Umbral de advertencia de combustible bajo
    
    # Capacidad de carga
    'capacity_m3': 19.0,                   # Capacidad del compartimento (m³)
    'compaction_ratio': 3.0,               # Ratio de compactación 3:1
    'max_load_kg': 12000.0,                # Peso máximo legal (kg)
    
    # Tiempos operacionales (minutos)
    'preparation_time_min': 15,            # Tiempo de preparación inicial
    'disposal_time_min': 35,               # Tiempo de descarga en vertedero
    'break_time_min': 30,                  # Tiempo de descanso obligatorio
    'buffer_time_min': 15,                 # Buffer al final de jornada
    
    # Velocidades promedio (km/h)
    'speed_residential_kmh': 20,           # Velocidad en zonas residenciales
    'speed_urban_kmh': 30,                 # Velocidad en vías urbanas
    'speed_highway_kmh': 60,               # Velocidad en autopista (al vertedero)
}

# ============== PARÁMETROS DE RESIDUOS ==============
WASTE_CONFIG = {
    # Densidad de residuos (kg/m³)
    'density_uncompacted_kg_m3': 175,      # Densidad sin compactar (promedio)
    'density_compacted_kg_m3': 525,        # Densidad después de compactar (175 * 3)
    
    # Peso por contenedor lleno (kg) - valores promedio
    'weight_per_container': {
        240: 42,    # Contenedor de 240L lleno
        340: 60,    # Contenedor de 340L lleno
        660: 115,   # Contenedor de 660L lleno
        1100: 190,  # Contenedor de 1100L lleno (si existe)
    },
    
    # Tiempo de servicio por contenedor (minutos)
    'service_time_per_container': {
        240: 1.5,   # Más rápido, contenedor pequeño
        340: 2.0,
        660: 2.5,
        1100: 3.5,
    },
    
    # Factor de tiempo adicional por contenedores múltiples
    'multiple_container_time_factor': 0.8,  # Reduce tiempo por contenedor adicional
}

# ============== UMBRALES DE RECOLECCIÓN ==============
COLLECTION_THRESHOLDS = {
    'min_fill_single': 70,          # Umbral mínimo para contenedor único (%)
    'min_fill_multiple': 70,        # Si alguno supera esto, se recogen todos
    'critical_fill': 90,            # Nivel crítico - prioridad alta (%)
    'overflow_fill': 100,           # Desbordado (%)
    'max_overflow': 150,            # Máximo de desborde para simulación (%)
    
    # Penalizaciones
    'unnecessary_trip_penalty': 10,  # Penalización por viaje innecesario (puntos)
    'overflow_penalty': 20,          # Penalización por contenedor desbordado (puntos)
}

# ============== PARÁMETROS DE EMISIONES ==============
EMISSIONS_CONFIG = {
    'co2_kg_per_liter_diesel': 2.64,    # Factor de emisión CO₂ para diésel
    'nox_g_per_km': 0.5,                # Emisiones NOx (g/km)
    'pm_g_per_km': 0.02,                # Material particulado (g/km)
}

# ============== CONFIGURACIÓN DE JORNADA ==============
WORKDAY_CONFIG = {
    'start_time': '07:00',              # Hora de inicio
    'end_time': '15:00',                # Hora de fin
    'total_hours': 8,                   # Horas totales
    'effective_hours': 7,               # Horas efectivas (descontando descansos)
    
    # Días de la semana (0=Lunes, 6=Domingo)
    'working_days': [0, 1, 2, 3, 4, 5], # Lunes a Sábado
    
    # Factores de tráfico por hora
    'traffic_factors': {
        7: 1.3,   # 07:00 - Mayor tráfico
        8: 1.4,   # 08:00 - Hora punta
        9: 1.3,   # 09:00 - Aún congestionado
        10: 1.1,  # 10:00 - Mejora
        11: 1.0,  # 11:00 - Normal
        12: 1.1,  # 12:00 - Almuerzo
        13: 1.2,  # 13:00 - Retorno almuerzo
        14: 1.1,  # 14:00 - Normal-alto
        15: 1.0,  # 15:00 - Fin de jornada
    }
}

# ============== PUNTOS DE INFRAESTRUCTURA ==============
INFRASTRUCTURE = {
    'depot': {
        'name': 'Depot Central Las Condes',
        'latitude': -33.4119,
        'longitude': -70.5241,
        'type': 'depot',
    },
    'disposal_sites': [
        {
            'name': 'Relleno Sanitario Santiago Poniente',
            'latitude': -33.3667,
            'longitude': -70.7500,
            'distance_km': 25,  # Distancia aproximada desde Las Condes
            'type': 'landfill',
        },
        {
            'name': 'Estación de Transferencia',
            'latitude': -33.4000,
            'longitude': -70.6500,
            'distance_km': 12,
            'type': 'transfer_station',
        }
    ],
    'fuel_stations': [
        {
            'name': 'Estación de Servicio Municipal',
            'latitude': -33.4150,
            'longitude': -70.5300,
            'type': 'fuel',
        }
    ]
}

# ============== EVENTOS Y PROBABILIDADES ==============
EVENT_PROBABILITIES = {
    'container_blocked': 0.05,          # 5% prob de contenedor bloqueado
    'traffic_jam': 0.10,                # 10% prob de atasco significativo
    'minor_breakdown': 0.02,            # 2% prob de avería menor
    'emergency_collection': 0.03,       # 3% prob de recolección de emergencia
    'weather_delay': 0.08,              # 8% prob de retraso por clima
}

# ============== FACTORES DE AJUSTE ==============
ADJUSTMENT_FACTORS = {
    # Factor de pendiente para Las Condes (zona con pendientes)
    'slope_consumption_factor': 1.15,   # 15% más consumo en zonas con pendiente
    
    # Factor de clima
    'rain_service_time_factor': 1.20,   # 20% más tiempo de servicio con lluvia
    'rain_speed_factor': 0.85,          # 15% menor velocidad con lluvia
    
    # Factor de día de la semana
    'saturday_traffic_factor': 0.8,     # Menos tráfico los sábados
    'monday_waste_factor': 1.2,         # Más basura los lunes (acumulado del domingo)
}

# ============== KPIs Y MÉTRICAS ==============
KPI_TARGETS = {
    'min_route_efficiency': 0.75,       # Eficiencia mínima de ruta (75%)
    'max_fuel_per_ton': 25,            # Máximo L/tonelada recogida
    'max_time_per_container': 5,        # Máximo minutos promedio por contenedor
    'min_capacity_utilization': 0.70,   # Utilización mínima de capacidad (70%)
    'max_co2_per_ton': 66,             # Máximo kg CO₂/tonelada (25L * 2.64)
}

# ============== CONFIGURACIÓN DE SIMULACIÓN ==============
SIMULATION_CONFIG = {
    'time_step_minutes': 1,              # Resolución temporal de la simulación
    'random_seed': 42,                  # Semilla para reproducibilidad
    'enable_random_events': True,       # Habilitar eventos aleatorios
    'enable_traffic_simulation': True,  # Habilitar simulación de tráfico
    'enable_weather_effects': True,     # Habilitar efectos del clima
    'log_level': 'INFO',                # Nivel de logging
    'save_results': True,               # Guardar resultados de simulación
    'results_path': './simulation_results/',
}

# ============== COSTS (para análisis económico) ==============
COST_CONFIG = {
    'fuel_cost_per_liter': 1050,        # Costo del diésel (CLP/L)
    'driver_hourly_cost': 8500,         # Costo por hora del conductor (CLP)
    'truck_maintenance_per_km': 150,    # Mantenimiento por km (CLP)
    'penalty_overflow_clp': 50000,      # Multa por contenedor desbordado (CLP)
    'penalty_missed_collection': 25000, # Multa por no recolectar (CLP)
    'disposal_cost_per_ton': 15000,     # Costo de disposición (CLP/ton)
}