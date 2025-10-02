# simulator/visualizer.py
"""
Módulo de visualizaciones específicas para la simulación de recolección de residuos.
Genera gráficos y representaciones visuales de los resultados de la simulación.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from .models import Truck, CollectionPoint, Container, SimulationState, Route
from .metrics import OperationalMetrics, EnvironmentalMetrics, EconomicMetrics, QualityMetrics
from .events import SimulationEvent


class SimulationVisualizer:
    """Generador de visualizaciones para la simulación"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'info': '#17becf',
            'secondary': '#7f7f7f'
        }
    
    def create_truck_status_timeline(
        self, 
        truck_history: List[Dict[str, Any]],
        events: List[SimulationEvent] = None
    ) -> go.Figure:
        """
        Crea un timeline del estado del camión durante la simulación.
        
        Args:
            truck_history: Lista de estados del camión con timestamps
            events: Eventos ocurridos durante la simulación
            
        Returns:
            Figura de Plotly con el timeline
        """
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                'Nivel de Combustible (%)', 
                'Carga del Camión (%)', 
                'Contenedores Recolectados',
                'Eventos'
            ],
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        if not truck_history:
            return fig
        
        # Convertir a DataFrame para facilitar manipulación
        df = pd.DataFrame(truck_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Gráfico 1: Nivel de combustible
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['fuel_percentage'],
                mode='lines',
                name='Combustible',
                line=dict(color=self.color_palette['primary'], width=2),
                fill='tonexty' if len(df) > 0 else None
            ),
            row=1, col=1
        )
        
        # Línea de advertencia de combustible bajo
        fig.add_hline(y=20, line_dash="dash", line_color="red", row=1, col=1,
                      annotation_text="Nivel Crítico")
        
        # Gráfico 2: Carga del camión (volumen y peso)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cargo_percentage'],
                mode='lines',
                name='Carga (Volumen)',
                line=dict(color=self.color_palette['success'], width=2)
            ),
            row=2, col=1
        )
        
        if 'weight_percentage' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['weight_percentage'],
                    mode='lines',
                    name='Carga (Peso)',
                    line=dict(color=self.color_palette['warning'], width=2, dash='dot')
                ),
                row=2, col=1
            )
        
        # Gráfico 3: Contenedores acumulados
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['containers_collected'],
                mode='lines+markers',
                name='Contenedores',
                line=dict(color=self.color_palette['info'], width=2),
                marker=dict(size=4)
            ),
            row=3, col=1
        )
        
        # Gráfico 4: Eventos
        if events:
            event_times = []
            event_descriptions = []
            event_colors = []
            
            for event in events:
                if hasattr(event, 'timestamp'):
                    event_times.append(event.timestamp)
                    event_descriptions.append(event.description)
                    
                    color_map = {
                        'LOW': self.color_palette['info'],
                        'MEDIUM': self.color_palette['warning'],
                        'HIGH': self.color_palette['danger'],
                        'CRITICAL': 'darkred'
                    }
                    event_colors.append(color_map.get(event.severity.value, 'gray'))
            
            if event_times:
                fig.add_trace(
                    go.Scatter(
                        x=event_times,
                        y=[1] * len(event_times),
                        mode='markers+text',
                        name='Eventos',
                        marker=dict(
                            size=10,
                            color=event_colors,
                            symbol='diamond'
                        ),
                        text=event_descriptions,
                        textposition="top center",
                        showlegend=False
                    ),
                    row=4, col=1
                )
        
        # Configuración de layout
        fig.update_layout(
            height=800,
            title_text="Timeline de Estado del Camión",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Configurar ejes Y
        fig.update_yaxes(title_text="Porcentaje (%)", row=1, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Porcentaje (%)", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Cantidad", row=3, col=1)
        fig.update_yaxes(title_text="", row=4, col=1, showticklabels=False)
        
        return fig
    
    def create_route_performance_dashboard(
        self, 
        operational_metrics: OperationalMetrics,
        environmental_metrics: EnvironmentalMetrics,
        economic_metrics: EconomicMetrics,
        quality_metrics: QualityMetrics
    ) -> go.Figure:
        """
        Crea un dashboard de rendimiento de la ruta.
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Eficiencia Operacional',
                'Impacto Ambiental',
                'Costos por Categoría',
                'Scores de Calidad',
                'Métricas por Contenedor',
                'Comparación vs Objetivos'
            ],
            specs=[[{"type": "indicator"}, {"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.12
        )
        
        # 1. Indicadores de eficiencia operacional
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=operational_metrics.average_capacity_utilization,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Utilización de Capacidad (%)"},
                delta={'reference': 75},  # Target de 75%
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['primary']},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Distribución de impacto ambiental
        emissions_labels = ['CO₂', 'NOx', 'PM']
        emissions_values = [
            environmental_metrics.co2_emissions_kg,
            environmental_metrics.nox_emissions_g / 1000,  # Convertir a kg
            environmental_metrics.pm_emissions_g / 1000    # Convertir a kg
        ]
        
        fig.add_trace(
            go.Pie(
                labels=emissions_labels,
                values=emissions_values,
                name="Emisiones",
                marker_colors=[self.color_palette['danger'], 
                              self.color_palette['warning'], 
                              self.color_palette['secondary']]
            ),
            row=1, col=2
        )
        
        # 3. Costos por categoría
        cost_categories = ['Combustible', 'Laboral', 'Mantenimiento', 'Disposición', 'Penalizaciones']
        cost_values = [
            economic_metrics.fuel_cost_clp,
            economic_metrics.labor_cost_clp,
            economic_metrics.maintenance_cost_clp,
            economic_metrics.disposal_cost_clp,
            economic_metrics.penalty_costs_clp
        ]
        
        fig.add_trace(
            go.Bar(
                x=cost_categories,
                y=cost_values,
                name="Costos",
                marker_color=self.color_palette['warning']
            ),
            row=1, col=3
        )
        
        # 4. Scores de calidad
        quality_scores = {
            'Servicio': quality_metrics.service_level_score,
            'Ruta': quality_metrics.route_efficiency_score,
            'Tiempo': quality_metrics.time_efficiency_score,
            'Combustible': quality_metrics.fuel_efficiency_score
        }
        
        fig.add_trace(
            go.Bar(
                x=list(quality_scores.keys()),
                y=list(quality_scores.values()),
                name="Scores de Calidad",
                marker_color=self.color_palette['success']
            ),
            row=2, col=1
        )
        
        # 5. Métricas por contenedor
        container_metrics = {
            'Fuel (L)': environmental_metrics.fuel_per_container_l,
            'CO₂ (kg)': environmental_metrics.co2_per_container_kg,
            'Costo (CLP)': economic_metrics.cost_per_container_clp / 1000  # En miles
        }
        
        fig.add_trace(
            go.Scatter(
                x=list(container_metrics.keys()),
                y=list(container_metrics.values()),
                mode='markers+lines',
                name="Por Contenedor",
                marker=dict(size=10, color=self.color_palette['info'])
            ),
            row=2, col=2
        )
        
        # 6. Comparación vs objetivos (placeholder - requiere targets específicos)
        target_comparison = {
            'Combustible': 85,  # % de cumplimiento
            'Emisiones': 92,
            'Costos': 78,
            'Tiempo': 88
        }
        
        fig.add_trace(
            go.Bar(
                x=list(target_comparison.keys()),
                y=list(target_comparison.values()),
                name="% Cumplimiento",
                marker_color=self.color_palette['primary']
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            height=800,
            title_text="Dashboard de Rendimiento de la Simulación",
            showlegend=False
        )
        
        return fig
    
    def create_collection_points_map(
        self, 
        completed_points: List[CollectionPoint],
        truck_path: List[Tuple[float, float]] = None,
        depot_location: Tuple[float, float] = (-33.4119, -70.5241)
    ) -> go.Figure:
        """
        Crea un mapa de los puntos de recolección visitados.
        """
        fig = go.Figure()
        
        # Agregar puntos de recolección
        for i, point in enumerate(completed_points):
            # Determinar color basado en el estado de los contenedores
            max_fill = max(container.current_fill_percentage for container in point.containers) if point.containers else 0
            
            if max_fill > 100:
                color = 'red'
                symbol = 'square'
            elif max_fill >= 90:
                color = 'orange'
                symbol = 'circle'
            elif max_fill >= 70:
                color = 'yellow'
                symbol = 'circle'
            else:
                color = 'green'
                symbol = 'circle'
            
            # Información del hover
            hover_text = f"Punto {i+1}: {point.address}<br>"
            hover_text += f"Contenedores: {len(point.containers)}<br>"
            hover_text += f"Llenado máximo: {max_fill:.1f}%<br>"
            hover_text += f"Peso total: {point.total_weight_to_collect:.1f} kg"
            
            fig.add_trace(
                go.Scattermapbox(
                    lat=[point.latitude],
                    lon=[point.longitude],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=color,
                        symbol=symbol
                    ),
                    text=[str(i+1)],
                    textposition="middle center",
                    textfont=dict(color="white", size=8),
                    hovertext=hover_text,
                    hoverinfo='text',
                    name=f'Punto {i+1}'
                )
            )
        
        # Agregar ruta del camión si está disponible
        if truck_path and len(truck_path) > 1:
            lats, lons = zip(*truck_path)
            fig.add_trace(
                go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='lines',
                    line=dict(width=3, color='blue'),
                    name='Ruta del Camión',
                    showlegend=True
                )
            )
        
        # Agregar depósito
        fig.add_trace(
            go.Scattermapbox(
                lat=[depot_location[0]],
                lon=[depot_location[1]],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='black',
                    symbol='star'
                ),
                text=['DEPOT'],
                textposition="bottom center",
                name='Depósito'
            )
        )
        
        # Configurar mapa
        if completed_points:
            center_lat = np.mean([point.latitude for point in completed_points])
            center_lon = np.mean([point.longitude for point in completed_points])
        else:
            center_lat, center_lon = depot_location
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=12
            ),
            height=600,
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            title="Mapa de Recolección - Puntos Visitados"
        )
        
        return fig
    
    def create_fuel_consumption_analysis(
        self, 
        truck_history: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Analiza el consumo de combustible durante la simulación.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Consumo de Combustible vs Tiempo',
                'Eficiencia por Segmento',
                'Consumo vs Carga',
                'Proyección de Combustible'
            ]
        )
        
        if not truck_history:
            return fig
        
        df = pd.DataFrame(truck_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 1. Consumo vs tiempo
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['fuel_consumed_cumulative'] if 'fuel_consumed_cumulative' in df.columns else df['fuel_percentage'],
                mode='lines',
                name='Combustible Acumulado',
                line=dict(color=self.color_palette['primary'])
            ),
            row=1, col=1
        )
        
        # 2. Eficiencia por segmento (si hay datos de segmentos)
        if 'segment_efficiency' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['segment_efficiency'],
                    name='Eficiencia L/km',
                    marker_color=self.color_palette['success']
                ),
                row=1, col=2
            )
        
        # 3. Consumo vs carga
        if 'cargo_percentage' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['cargo_percentage'],
                    y=df['fuel_consumed_segment'] if 'fuel_consumed_segment' in df.columns else df['fuel_percentage'],
                    mode='markers',
                    name='Consumo vs Carga',
                    marker=dict(
                        size=8,
                        color=self.color_palette['warning'],
                        opacity=0.7
                    )
                ),
                row=2, col=1
            )
        
        # 4. Proyección de combustible restante
        if len(df) > 1:
            # Calcular tendencia
            consumption_rate = (df['fuel_percentage'].iloc[0] - df['fuel_percentage'].iloc[-1]) / len(df)
            remaining_hours = df['fuel_percentage'].iloc[-1] / consumption_rate if consumption_rate > 0 else 0
            
            projection_x = list(range(len(df), len(df) + int(remaining_hours)))
            projection_y = [df['fuel_percentage'].iloc[-1] - (i * consumption_rate) for i in projection_x]
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['fuel_percentage'],
                    mode='lines',
                    name='Real',
                    line=dict(color=self.color_palette['primary'])
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=projection_x,
                    y=projection_y,
                    mode='lines',
                    name='Proyección',
                    line=dict(color=self.color_palette['danger'], dash='dash')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=600,
            title_text="Análisis de Consumo de Combustible"
        )
        
        return fig
    
    def create_events_impact_chart(
        self, 
        events: List[SimulationEvent],
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float]
    ) -> go.Figure:
        """
        Muestra el impacto de los eventos en las métricas de la simulación.
        """
        fig = go.Figure()
        
        # Crear datos para el gráfico
        event_types = [event.event_type.value for event in events]
        event_counts = {}
        for event_type in event_types:
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Gráfico de barras con tipos de eventos
        fig.add_trace(
            go.Bar(
                x=list(event_counts.keys()),
                y=list(event_counts.values()),
                name="Frecuencia de Eventos",
                marker_color=self.color_palette['warning']
            )
        )
        
        fig.update_layout(
            title="Distribución de Eventos Durante la Simulación",
            xaxis_title="Tipo de Evento",
            yaxis_title="Frecuencia",
            height=400
        )
        
        return fig
    
    def create_comparison_chart(
        self, 
        scenarios: List[Dict[str, Any]],
        metric_name: str = "total_cost_clp"
    ) -> go.Figure:
        """
        Compara múltiples escenarios de simulación.
        
        Args:
            scenarios: Lista de diccionarios con resultados de diferentes simulaciones
            metric_name: Nombre de la métrica a comparar
        """
        fig = go.Figure()
        
        scenario_names = [scenario['name'] for scenario in scenarios]
        metric_values = [scenario['metrics'].get(metric_name, 0) for scenario in scenarios]
        
        fig.add_trace(
            go.Bar(
                x=scenario_names,
                y=metric_values,
                name=metric_name.replace('_', ' ').title(),
                marker_color=self.color_palette['primary']
            )
        )
        
        fig.update_layout(
            title=f"Comparación de Escenarios - {metric_name.replace('_', ' ').title()}",
            xaxis_title="Escenario",
            yaxis_title=metric_name.replace('_', ' ').title(),
            height=400
        )
        
        return fig