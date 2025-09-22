# config/__init__.py
"""Configuration package for WhatsApp Chat Analyzer"""

from .settings import settings, COLOR_SCHEMES, MESSAGE_TYPES

__all__ = ['settings', 'COLOR_SCHEMES', 'MESSAGE_TYPES']

# src/__init__.py
"""Main source package for WhatsApp Chat Analyzer"""

__version__ = "2.0.0"
__author__ = "Khushi"

# src/components/__init__.py
"""Components package containing reusable modules"""

from src.components.dashboard import DashboardComponents
from src.components.sentiment_analyzer import SentimentAnalysisEngine
from src.components.visualization import AdvancedVisualizations
from src.components.export_manager import ExportManager

__all__ = [
    'DashboardComponents',
    'SentimentAnalysisEngine', 
    'AdvancedVisualizations',
    'ExportManager'
]