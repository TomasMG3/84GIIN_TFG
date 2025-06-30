from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional, List

class ContainerBase(BaseModel):
    container_id: str
    latitude: float
    longitude: float
    address: Optional[str] = None
    zone: Optional[str] = None
    capacity: float = Field(default=1000.0, gt=0)

class ContainerCreate(ContainerBase):
    temperature: float
    battery_level: float = Field(default=100.0, ge=0, le=100)

class ContainerUpdate(BaseModel):
    fill_percentage: Optional[float] = None
    current_level: Optional[float] = None
    temperature: Optional[float] = None
    battery_level: Optional[float] = Field(default=None, ge=0, le=100)

class Container(ContainerBase):
    id: int
    current_level: float = Field(default=0.0)
    fill_percentage: float = Field(default=0.0)
    temperature: Optional[float] = None
    battery_level: float = Field(default=100.0)
    is_active: bool = Field(default=True)
    last_emptied: Optional[datetime] = None
    last_update: datetime
    model_config = ConfigDict(from_attributes=True)

class SensorReadingCreate(BaseModel):
    container_id: int
    fill_level: float = Field(ge=0)
    fill_percentage: float = Field(ge=0, le=100)
    temperature: float
    battery_voltage: float = Field(gt=0)
    signal_strength: int

class SensorReading(SensorReadingCreate):
    id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True

class RouteCreate(BaseModel):
    route_name: str
    container_ids: List[int]
    optimization_algorithm: str = "genetic_algorithm"

class Route(BaseModel):
    id: int
    route_name: str
    created_at: datetime
    total_distance_km: Optional[float] = None
    estimated_time_minutes: Optional[int] = None
    fuel_consumption_liters: Optional[float] = None
    co2_emissions_kg: Optional[float] = None
    containers_count: int
    optimization_algorithm: str
    is_optimized: bool = Field(default=False)
    
    class Config:
        from_attributes = True

class VehicleCreate(BaseModel):
    truck_id: str
    capacity_kg: float = Field(gt=0)
    fuel_efficiency_kmpl: float = Field(gt=0)
    contractor: str

class Vehicle(VehicleCreate):
    id: int
    is_active: bool = Field(default=True)
    current_lat: Optional[float] = None
    current_lon: Optional[float] = None
    last_update: datetime
    
    class Config:
        from_attributes = True

# Esquemas adicionales para respuestas más específicas
class ContainerSummary(BaseModel):
    id: int
    container_id: str
    fill_percentage: float
    zone: Optional[str]
    is_active: bool
    last_update: datetime

class ContainerAlert(BaseModel):
    container_id: int
    container_name: str
    alert_type: str
    message: str
    severity: str
    timestamp: datetime