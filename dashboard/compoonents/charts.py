import plotly.express as px
import pandas as pd
from typing import List, Dict
from plotly.graph_objects import Figure

def create_fill_distribution_chart(containers: List[Dict]) -> Figure:
    """
    Crea un histograma de distribución de llenado de contenedores
    
    Args:
        containers: Lista de contenedores con sus datos
    
    Returns:
        Gráfico de histograma de Plotly
    """
    df = pd.DataFrame(containers)
    fig = px.histogram(
        df, 
        x='fill_percentage',
        nbins=20,
        title='Distribución de Niveles de Llenado',
        labels={'fill_percentage': 'Porcentaje de Llenado (%)'},
        color_discrete_sequence=['#2ca02c']
    )
    
    fig.update_layout(
        bargap=0.1,
        xaxis_title="Porcentaje de Llenado",
        yaxis_title="Número de Contenedores"
    )
    
    return fig

def create_zone_analysis_chart(containers: List[Dict]) -> Figure:
    """
    Crea gráfico de análisis por zonas
    
    Args:
        containers: Lista de contenedores con sus datos
    
    Returns:
        Gráfico de barras de Plotly
    """
    df = pd.DataFrame(containers)
    
    # Calcular métricas por zona
    zone_stats = df.groupby('zone').agg(
        avg_fill=('fill_percentage', 'mean'),
        count=('container_id', 'count'),
        high_fill=('fill_percentage', lambda x: (x > 75).sum())
    ).reset_index()
    
    fig = px.bar(
        zone_stats,
        x='zone',
        y=['avg_fill', 'high_fill'],
        title='Análisis por Zona',
        labels={'value': 'Valor', 'variable': 'Métrica'},
        barmode='group',
        color_discrete_sequence=['#1f77b4', '#ff7f0e']
    )
    
    fig.update_layout(
        xaxis_title="Zona",
        yaxis_title="Valor",
        legend_title="Métrica"
    )
    
    return fig

def create_timeseries_chart(historical_data: pd.DataFrame, container_id: str) -> Figure:
    """
    Crea gráfico temporal de llenado para un contenedor específico
    
    Args:
        historical_data: DataFrame con datos históricos
        container_id: ID del contenedor a analizar
    
    Returns:
        Gráfico de línea de Plotly
    """
    container_data = historical_data[historical_data['container_id'] == container_id]
    
    fig = px.line(
        container_data,
        x='timestamp',
        y='fill_percentage',
        title=f'Histórico de Llenado - Contenedor {container_id}',
        labels={'fill_percentage': 'Porcentaje de Llenado (%)'},
        color_discrete_sequence=['#9467bd']
    )
    
    fig.update_layout(
        xaxis_title="Fecha y Hora",
        yaxis_title="Porcentaje de Llenado",
        hovermode="x unified"
    )
    
    # Agregar línea de alerta
    fig.add_hline(
        y=75,
        line_dash="dot",
        line_color="red",
        annotation_text="Umbral de Alerta (75%)"
    )
    
    return fig