from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship, Mapped, mapped_column
from typing import Optional, List
from datetime import datetime
from app.database import Base
from sqlalchemy.dialects.postgresql import JSON


class Container(Base):
    __tablename__ = "containers"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    container_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    latitude: Mapped[float] = mapped_column(nullable=False)
    longitude: Mapped[float] = mapped_column(nullable=False)
    address: Mapped[str] = mapped_column(nullable=False)
    zone: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    capacity: Mapped[float] = mapped_column(nullable=False)
    current_level: Mapped[float] = mapped_column(nullable=False, default=0.0)
    fill_percentage: Mapped[float] = mapped_column(nullable=False, default=0.0)
    temperature: Mapped[float] = mapped_column(nullable=False)
    battery_level: Mapped[float] = mapped_column(nullable=False, default=100.0)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    last_emptied: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    last_update: Mapped[datetime] = mapped_column(nullable=False, default=datetime.utcnow)

    sensor_readings: Mapped[list["SensorReading"]] = relationship(back_populates="container")
    collection_events: Mapped[list["CollectionEvent"]] = relationship(back_populates="container")

class SensorReading(Base):
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True, index=True)
    container_id = Column(Integer, ForeignKey("containers.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    fill_level = Column(Float)  # cm desde el fondo
    fill_percentage = Column(Float)
    temperature = Column(Float)
    battery_voltage = Column(Float)
    signal_strength = Column(Integer)  # RSSI
    
    container = relationship("Container", back_populates="sensor_readings")

class CollectionEvent(Base):
    __tablename__ = "collection_events"
    
    id = Column(Integer, primary_key=True, index=True)
    container_id = Column(Integer, ForeignKey("containers.id"))
    route_id = Column(Integer, ForeignKey("routes.id"))
    collected_at = Column(DateTime, default=datetime.utcnow)
    volume_collected = Column(Float)
    collection_time_minutes = Column(Integer)
    truck_id = Column(String)
    
    container = relationship("Container", back_populates="collection_events")
    route = relationship("Route", back_populates="collection_events")

class Route(Base):
    __tablename__ = "routes"
    
    id = Column(Integer, primary_key=True, index=True)
    route_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_distance_km = Column(Float)
    estimated_time_minutes = Column(Integer)
    fuel_consumption_liters = Column(Float)
    co2_emissions_kg = Column(Float)
    containers_count = Column(Integer)
    optimization_algorithm = Column(String)  # GA, SA, etc.
    is_optimized = Column(Boolean, default=False)
    
    route_coordinates = Column(JSON)
    
    collection_events = relationship("CollectionEvent", back_populates="route")

class Vehicle(Base):
    __tablename__ = "vehicles"
    
    id = Column(Integer, primary_key=True, index=True)
    truck_id = Column(String, unique=True)
    capacity_kg = Column(Float)
    fuel_efficiency_kmpl = Column(Float)
    contractor = Column(String)  # VEOLIA, GENCO
    is_active = Column(Boolean, default=True)
    current_lat = Column(Float)
    current_lon = Column(Float)
    last_update = Column(DateTime, default=datetime.utcnow)