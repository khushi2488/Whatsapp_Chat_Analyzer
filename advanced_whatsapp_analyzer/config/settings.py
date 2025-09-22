"""
Configuration settings for Advanced WhatsApp Chat Analyzer
This module handles all configuration management and environment variables.
"""

import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass,field

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = os.getenv("DATABASE_URL", "sqlite:///chat_analyzer.db")
    echo: bool = os.getenv("DEBUG", "False").lower() == "true"
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    encryption_key: str = os.getenv("ENCRYPTION_KEY", "dev-encryption-key")
    jwt_secret: str = os.getenv("JWT_SECRET", "dev-jwt-secret")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expiration_hours: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    bcrypt_rounds: int = int(os.getenv("BCRYPT_ROUNDS", "12"))

@dataclass
class FileConfig:
    """File upload and processing configuration"""
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    allowed_file_types: List[str] = field(default_factory=lambda: os.getenv("ALLOWED_FILE_TYPES", "txt,csv,xlsx").split(","))
    upload_folder: Path = BASE_DIR / "uploads"
    temp_folder: Path = BASE_DIR / "temp"
    assets_folder: Path = BASE_DIR / "assets"

@dataclass
class AnalyticsConfig:
    """Analytics and ML model configuration"""
    default_language: str = os.getenv("DEFAULT_LANGUAGE", "en")
    sentiment_model: str = os.getenv("SENTIMENT_MODEL", "vader")
    topic_model_num_topics: int = int(os.getenv("TOPIC_MODEL_NUM_TOPICS", "10"))
    min_words_for_analysis: int = int(os.getenv("MIN_WORDS_FOR_ANALYSIS", "50"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "1000"))
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    processing_timeout: int = int(os.getenv("PROCESSING_TIMEOUT", "300"))

@dataclass
class CacheConfig:
    """Cache configuration"""
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    enable_cache: bool = os.getenv("ENABLE_CACHE", "True").lower() == "true"

@dataclass
class APIConfig:
    """External API configuration"""
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    rate_limit_per_minute: int = int(os.getenv("API_RATE_LIMIT", "60"))

@dataclass
class UIConfig:
    """UI configuration for Streamlit"""
    theme: str = os.getenv("THEME", "light")
    sidebar_state: str = os.getenv("SIDEBAR_STATE", "expanded")
    layout: str = os.getenv("LAYOUT", "wide")
    page_title: str = os.getenv("APP_NAME", "Advanced WhatsApp Chat Analyzer")
    page_icon: str = "📊"

@dataclass
class FeatureFlags:
    """Feature toggle configuration"""
    enable_sentiment_analysis: bool = os.getenv("ENABLE_SENTIMENT_ANALYSIS", "True").lower() == "true"
    enable_topic_modeling: bool = os.getenv("ENABLE_TOPIC_MODELING", "True").lower() == "true"
    enable_predictive_analytics: bool = os.getenv("ENABLE_PREDICTIVE_ANALYTICS", "True").lower() == "true"
    enable_export_features: bool = os.getenv("ENABLE_EXPORT_FEATURES", "True").lower() == "true"
    enable_real_time_processing: bool = os.getenv("ENABLE_REAL_TIME_PROCESSING", "False").lower() == "true"

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Path = BASE_DIR / os.getenv("LOG_FILE", "logs/app.log")
    max_size_mb: int = int(os.getenv("MAX_LOG_SIZE_MB", "10"))
    backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"

class Settings:
    """Main settings class that combines all configuration"""
    
    def __init__(self):
        # App info
        self.app_name: str = os.getenv("APP_NAME", "Advanced WhatsApp Chat Analyzer")
        self.app_version: str = os.getenv("APP_VERSION", "2.0.0")
        self.debug: bool = os.getenv("DEBUG", "False").lower() == "true"
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        
        # Configuration objects
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        self.files = FileConfig()
        self.analytics = AnalyticsConfig()
        self.cache = CacheConfig()
        self.api = APIConfig()
        self.ui = UIConfig()
        self.features = FeatureFlags()
        self.logging = LoggingConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.files.upload_folder,
            self.files.temp_folder,
            BASE_DIR / "logs",
            BASE_DIR / "models" / "trained",
            BASE_DIR / "data" / "processed",
            BASE_DIR / "exports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_database_url(self) -> str:
        """Get properly formatted database URL"""
        return self.database.url
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    def get_allowed_file_extensions(self) -> List[str]:
        """Get list of allowed file extensions with dots"""
        return [f".{ext.strip()}" for ext in self.files.allowed_file_types]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary (for debugging)"""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "debug": self.debug,
            "environment": self.environment,
            "database_url": self.database.url.split("@")[-1] if "@" in self.database.url else self.database.url,  # Hide credentials
            "features": {
                "sentiment_analysis": self.features.enable_sentiment_analysis,
                "topic_modeling": self.features.enable_topic_modeling,
                "predictive_analytics": self.features.enable_predictive_analytics,
                "export_features": self.features.enable_export_features,
                "real_time_processing": self.features.enable_real_time_processing,
            }
        }

# Global settings instance
settings = Settings()

# Language support configuration
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech"
}

# Model configurations
MODEL_CONFIGS = {
    "sentiment": {
        "vader": {
            "name": "VADER",
            "description": "Rule-based sentiment analysis, fast and efficient",
            "languages": ["en"],
            "accuracy": 0.85
        },
        "textblob": {
            "name": "TextBlob",
            "description": "Pattern-based sentiment analysis",
            "languages": ["en"],
            "accuracy": 0.80
        },
        "transformers": {
            "name": "DistilBERT",
            "description": "Transformer-based model, high accuracy",
            "languages": list(SUPPORTED_LANGUAGES.keys()),
            "accuracy": 0.92
        }
    },
    "topic": {
        "lda": {
            "name": "Latent Dirichlet Allocation",
            "description": "Traditional topic modeling",
            "min_documents": 20,
            "max_topics": 20
        },
        "bertopic": {
            "name": "BERTopic",
            "description": "BERT-based topic modeling",
            "min_documents": 10,
            "max_topics": 50
        }
    }
}

# Export format configurations
EXPORT_FORMATS = {
    "pdf": {
        "name": "PDF Report",
        "extension": ".pdf",
        "mime_type": "application/pdf"
    },
    "excel": {
        "name": "Excel Spreadsheet", 
        "extension": ".xlsx",
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    },
    "csv": {
        "name": "CSV Data",
        "extension": ".csv", 
        "mime_type": "text/csv"
    },
    "json": {
        "name": "JSON Data",
        "extension": ".json",
        "mime_type": "application/json"
    }
}

# Color schemes for visualizations
COLOR_SCHEMES = {
    "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "sentiment": {
        "positive": "#22c55e",
        "negative": "#ef4444", 
        "neutral": "#6b7280"
    },
    "corporate": ["#2563eb", "#7c3aed", "#db2777", "#dc2626", "#ea580c", "#ca8a04"],
    "pastel": ["#fecaca", "#fed7af", "#fef3c7", "#d9f99d", "#a7f3d0", "#bfdbfe", "#ddd6fe", "#f3e8ff"],
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "file_size_warning_mb": 50,
    "processing_time_warning_seconds": 60,
    "memory_usage_warning_mb": 500,
    "max_messages_for_realtime": 10000,
    "batch_processing_threshold": 1000
}
MESSAGE_TYPES = {
    "text": "Text",
    "image": "Image",
    "video": "Video",
    "audio": "Audio",
    "document": "Document",
    "link": "Link",
    "sticker": "Sticker",
}

if __name__ == "__main__":
    # Test configuration
    print("📋 Configuration Summary:")
    print("=" * 50)
    import json
    print(json.dumps(settings.to_dict(), indent=2))
    print("=" * 50)
    print(f"✅ Configuration loaded successfully!")
    print(f"🗂️  Base directory: {BASE_DIR}")
    print(f"🔧 Environment: {settings.environment}")
    print(f"🐛 Debug mode: {settings.debug}")
    print(f"💾 Database: {settings.database.url.split('/')[-1]}")
    print(f"📁 Upload folder: {settings.files.upload_folder}")
    print(f"🌐 Supported languages: {len(SUPPORTED_LANGUAGES)}")
    print(f"🎯 Features enabled: {sum(1 for f in [settings.features.enable_sentiment_analysis, settings.features.enable_topic_modeling, settings.features.enable_predictive_analytics, settings.features.enable_export_features] if f)}/4")