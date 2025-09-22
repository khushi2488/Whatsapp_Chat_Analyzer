"""
Application constants for Advanced WhatsApp Chat Analyzer
Contains all constant values, patterns, and configuration data used across the application.
"""

from datetime import timedelta
from typing import Dict, List, Tuple, Any

# Application Information
APP_NAME = "Advanced WhatsApp Chat Analyzer"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Enterprise-grade WhatsApp chat analysis with AI-powered insights"
APP_AUTHOR = "AI Analytics Team"

# File Processing Constants
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_FILE_TYPES = ['.txt', '.csv']
SUPPORTED_ENCODINGS = ['utf-8', 'utf-16', 'latin1', 'cp1252']

# WhatsApp Message Patterns
WHATSAPP_PATTERNS = {
    'format1': r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s*(?:AM|PM))\s-\s([^:]+?):\s*(.*)',
    'format2': r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM))\]\s([^:]+?):\s*(.*)',
    'format3': r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2})\s-\s([^:]+?):\s*(.*)',
    'format4': r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s-\s([^:]+?):\s*(.*)',
    'system': r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s*(?:AM|PM)?)\s-\s(.*?)(?:\s-\s(.*))?'
}

# Date Format Patterns
DATE_FORMATS = [
    '%m/%d/%y, %I:%M %p',      # 1/1/23, 12:00 AM
    '%m/%d/%Y, %I:%M %p',      # 1/1/2023, 12:00 AM
    '%d/%m/%y, %H:%M',         # 1/1/23, 12:00
    '%d/%m/%Y, %H:%M',         # 1/1/2023, 12:00
    '%Y-%m-%d %H:%M:%S',       # 2023-01-01 12:00:00
    '%m/%d/%y, %I:%M:%S %p',   # 1/1/23, 12:00:00 AM
    '%d.%m.%y, %H:%M',         # 1.1.23, 12:00
    '%d.%m.%Y, %H:%M',         # 1.1.2023, 12:00
]

# System Message Indicators
SYSTEM_MESSAGE_INDICATORS = [
    'Messages and calls are end-to-end encrypted',
    'created group',
    'added',
    'left',
    'removed',
    'changed the group description',
    'changed the group name',
    'changed this group\'s icon',
    'security code changed',
    'You deleted this message',
    'This message was deleted',
    'joined using this group\'s invite link',
    'changed the group settings',
    'changed to',
    'was removed',
    'Messages to this chat and calls are now secured',
    'changed their phone number',
    'Your security code with'
]

# Media Message Indicators
MEDIA_INDICATORS = [
    '<Media omitted>',
    'image omitted',
    'video omitted',
    'audio omitted',
    'document omitted',
    'GIF omitted',
    'sticker omitted',
    'Contact card omitted',
    'Live location',
    '📷 Photo',
    '🎥 Video',
    '🎵 Audio',
    '📄 Document',
    '📍 Location',
    '🎤 Voice message',
    '📞 Missed voice call',
    '📞 Missed video call',
    'voice message',
    'video message',
    'photo',
    'document',
    'location',
    'contact'
]

# Language Support
SUPPORTED_LANGUAGES = {
    'auto': 'Auto-detect',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'da': 'Danish',
    'fi': 'Finnish',
    'pl': 'Polish',
    'cs': 'Czech',
    'he': 'Hebrew',
    'th': 'Thai',
    'vi': 'Vietnamese'
}

# Sentiment Analysis Models
SENTIMENT_MODELS = {
    'vader': {
        'name': 'VADER',
        'description': 'Rule-based sentiment analysis, fast and efficient',
        'languages': ['en'],
        'accuracy': 0.85,
        'speed': 'fast'
    },
    'textblob': {
        'name': 'TextBlob',
        'description': 'Pattern-based sentiment analysis',
        'languages': ['en'],
        'accuracy': 0.80,
        'speed': 'fast'
    },
    'transformers': {
        'name': 'DistilBERT',
        'description': 'Transformer-based model, high accuracy',
        'languages': list(SUPPORTED_LANGUAGES.keys()),
        'accuracy': 0.92,
        'speed': 'medium'
    },
    'multi': {
        'name': 'Multi-Model Ensemble',
        'description': 'Combines multiple models for best accuracy',
        'languages': ['en'],
        'accuracy': 0.95,
        'speed': 'slow'
    }
}

# Color Schemes for Visualizations
COLOR_SCHEMES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'sentiment': {
        'positive': '#22c55e',
        'negative': '#ef4444',
        'neutral': '#6b7280'
    },
    'corporate': ['#2563eb', '#7c3aed', '#db2777', '#dc2626', '#ea580c', '#ca8a04'],
    'pastel': ['#fecaca', '#fed7af', '#fef3c7', '#d9f99d', '#a7f3d0', '#bfdbfe', '#ddd6fe', '#f3e8ff'],
    'time_periods': {
        'Morning': '#fbbf24',
        'Afternoon': '#f59e0b',
        'Evening': '#dc2626',
        'Night': '#6366f1'
    },
    'user_activity': ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#84cc16', '#f97316'],
    'emotion': {
        'joy': '#22c55e',
        'sadness': '#3b82f6',
        'anger': '#ef4444',
        'fear': '#6b7280',
        'surprise': '#f59e0b',
        'disgust': '#84cc16',
        'anticipation': '#8b5cf6',
        'trust': '#06b6d4'
    }
}

# Response Time Categories
RESPONSE_TIME_CATEGORIES = {
    'immediate': {'min': 0, 'max': 1, 'label': 'Immediate (<1min)', 'color': '#22c55e'},
    'very_fast': {'min': 1, 'max': 5, 'label': 'Very Fast (1-5min)', 'color': '#84cc16'},
    'fast': {'min': 5, 'max': 30, 'label': 'Fast (5-30min)', 'color': '#f59e0b'},
    'moderate': {'min': 30, 'max': 60, 'label': 'Moderate (30-60min)', 'color': '#f97316'},
    'slow': {'min': 60, 'max': 240, 'label': 'Slow (1-4hr)', 'color': '#ef4444'},
    'very_slow': {'min': 240, 'max': float('inf'), 'label': 'Very Slow (>4hr)', 'color': '#991b1b'}
}

# Time Periods
TIME_PERIODS = {
    'Morning': {'start': 6, 'end': 12, 'icon': '🌅'},
    'Afternoon': {'start': 12, 'end': 17, 'icon': '☀️'},
    'Evening': {'start': 17, 'end': 21, 'icon': '🌆'},
    'Night': {'start': 21, 'end': 6, 'icon': '🌙'}
}

# Days of Week
DAYS_OF_WEEK = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}

# Export Formats
EXPORT_FORMATS = {
    'pdf': {
        'name': 'PDF Report',
        'extension': '.pdf',
        'mime_type': 'application/pdf',
        'description': 'Comprehensive analysis report in PDF format'
    },
    'excel': {
        'name': 'Excel Spreadsheet',
        'extension': '.xlsx',
        'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'description': 'Data and analysis in Excel format with multiple sheets'
    },
    'csv': {
        'name': 'CSV Data',
        'extension': '.csv',
        'mime_type': 'text/csv',
        'description': 'Raw data in comma-separated values format'
    },
    'json': {
        'name': 'JSON Data',
        'extension': '.json',
        'mime_type': 'application/json',
        'description': 'Structured data and analysis in JSON format'
    }
}

# Analysis Thresholds
ANALYSIS_THRESHOLDS = {
    'min_messages_for_analysis': 10,
    'min_users_for_group_analysis': 2,
    'min_days_for_trend_analysis': 7,
    'max_users_for_detailed_analysis': 50,
    'large_dataset_warning': 10000,
    'very_large_dataset_warning': 50000,
    'sentiment_confidence_threshold': 0.6,
    'response_time_outlier_threshold': 1440,  # 24 hours in minutes
    'activity_burst_threshold': 10,  # messages in 5 minutes
    'long_message_threshold': 500,  # characters
    'short_message_threshold': 10   # characters
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'file_size_warning_mb': 50,
    'processing_time_warning_seconds': 60,
    'memory_usage_warning_mb': 500,
    'max_messages_for_realtime': 10000,
    'batch_processing_threshold': 1000,
    'visualization_point_limit': 5000,
    'export_size_warning_mb': 100
}

# Regex Patterns
REGEX_PATTERNS = {
    'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'[\+]?[(]?[\d\s\-\(\)]{10,}',
    'mention': r'@(\w+)',
    'hashtag': r'#(\w+)',
    'emoji': r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]',
    'question': r'[?？]\s*$',
    'exclamation': r'[!！]\s*$',
    'caps_lock': r'^[A-Z\s\d\W]+$'
}

# Stop Words for Text Analysis
STOP_WORDS = {
    'en': {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our',
        'their', 'a', 'an', 'as', 'if', 'each', 'how', 'which', 'who', 'when', 'where',
        'why', 'what', 'there', 'here', 'now', 'then', 'than', 'so', 'very', 'just',
        'get', 'got', 'go', 'going', 'gone', 'know', 'like', 'think', 'see', 'look',
        'come', 'came', 'want', 'said', 'say', 'well', 'good', 'right', 'yeah', 'yes',
        'no', 'ok', 'okay', 'oh', 'ah', 'um', 'uh', 'hm', 'hmm'
    },
    'es': {
        'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le',
        'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una',
        'sobre', 'todo', 'también', 'tras', 'otro', 'algún', 'tanto', 'esa', 'estos',
        'mucho', 'antes', 'hasta', 'sin', 'entre', 'cuando', 'donde', 'como', 'porque',
        'pero', 'si', 'ya', 'muy', 'más', 'me', 'mi', 'tu', 'él', 'ella', 'nosotros',
        'vosotros', 'ellos', 'ellas', 'ser', 'estar', 'tener', 'hacer', 'poder', 'decir',
        'ir', 'ver', 'dar', 'saber', 'querer', 'llegar', 'pasar', 'deber', 'poner',
        'parecer', 'quedar', 'creer', 'hablar', 'llevar', 'dejar', 'seguir', 'encontrar',
        'llamar', 'venir', 'pensar', 'salir', 'volver', 'tomar', 'conocer', 'vivir',
        'sentir', 'tratar', 'mirar', 'contar', 'empezar', 'esperar', 'buscar', 'existir'
    }
}

# Error Messages
ERROR_MESSAGES = {
    'file_too_large': 'File size exceeds the maximum limit of {max_size} MB',
    'invalid_file_type': 'Invalid file type. Supported formats: {formats}',
    'empty_file': 'The uploaded file is empty',
    'invalid_format': 'File does not appear to be a valid WhatsApp chat export',
    'processing_failed': 'Failed to process the chat file. Please check the format',
    'insufficient_data': 'Not enough data for meaningful analysis',
    'analysis_failed': 'Analysis failed due to an unexpected error',
    'export_failed': 'Failed to generate export file',
    'permission_denied': 'You do not have permission to perform this action',
    'rate_limit_exceeded': 'Rate limit exceeded. Please try again later',
    'invalid_parameters': 'Invalid analysis parameters provided',
    'memory_limit': 'Dataset too large for available memory',
    'timeout': 'Operation timed out. Please try with a smaller dataset'
}

# Success Messages
SUCCESS_MESSAGES = {
    'file_uploaded': 'File uploaded successfully',
    'processing_complete': 'Chat analysis completed successfully',
    'export_generated': 'Export file generated successfully',
    'settings_saved': 'Settings saved successfully',
    'data_cleared': 'Data cleared successfully',
    'analysis_complete': 'Analysis completed with {count} insights generated'
}

# Default Settings
DEFAULT_SETTINGS = {
    'theme': 'light',
    'language': 'en',
    'sentiment_model': 'vader',
    'max_file_size_mb': 100,
    'enable_caching': True,
    'auto_save_results': True,
    'privacy_mode': False,
    'detailed_logging': False,
    'export_format': 'pdf',
    'visualization_style': 'modern',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'timezone': 'UTC',
    'batch_size': 1000,
    'max_users_display': 20,
    'chart_height': 400,
    'animation_enabled': True
}

# API Endpoints (for future API integration)
API_ENDPOINTS = {
    'upload': '/api/upload',
    'analyze': '/api/analyze',
    'export': '/api/export',
    'status': '/api/status',
    'health': '/api/health',
    'insights': '/api/insights',
    'users': '/api/users',
    'sentiment': '/api/sentiment',
    'trends': '/api/trends',
    'reports': '/api/reports'
}

# Cache Settings
CACHE_SETTINGS = {
    'default_ttl': 3600,  # 1 hour
    'max_size': 100,      # Maximum number of cached items
    'sentiment_ttl': 7200, # 2 hours for sentiment cache
    'analysis_ttl': 1800,  # 30 minutes for analysis cache
    'export_ttl': 600,     # 10 minutes for export cache
    'cleanup_interval': 300 # 5 minutes cleanup interval
}

# Security Settings
SECURITY_SETTINGS = {
    'max_login_attempts': 5,
    'session_timeout': 3600,
    'password_min_length': 8,
    'require_special_chars': True,
    'encryption_key_length': 32,
    'hash_salt_length': 16,
    'secure_cookie': True,
    'csrf_protection': True,
    'rate_limit_requests': 100,
    'rate_limit_window': 3600
}

# Chart Configuration
CHART_CONFIG = {
    'default_width': 800,
    'responsive': True,
    'show_legend': True,
    'show_toolbar': True,
    'export_enabled': True,
    'animation_duration': 750,
    'hover_mode': 'closest',
    'color_palette': 'viridis',
    'font_family': 'Arial, sans-serif',
    'font_size': 12,
    'title_font_size': 16,
    'axis_font_size': 10,
    'margin': {'l': 60, 'r': 60, 't': 60, 'b': 60}
}

# Notification Settings
NOTIFICATION_SETTINGS = {
    'show_processing_notifications': True,
    'show_success_notifications': True,
    'show_warning_notifications': True,
    'show_error_notifications': True,
    'notification_duration': 5000,  # milliseconds
    'auto_dismiss': True,
    'show_progress_bars': True,
    'sound_enabled': False
}

# Feature Flags
FEATURE_FLAGS = {
    'sentiment_analysis': True,
    'topic_modeling': True,
    'predictive_analytics': True,
    'export_features': True,
    'real_time_processing': False,
    'advanced_visualizations': True,
    'user_management': False,
    'api_access': False,
    'cloud_storage': False,
    'machine_learning': True,
    'data_encryption': True,
    'audit_logging': True,
    'rate_limiting': True,
    'caching': True,
    'multi_language': True,
    'dark_mode': True,
    'mobile_responsive': True,
    'accessibility': True,
    'gdpr_compliance': True
}

# Database Configuration
DATABASE_CONFIG = {
    'connection_timeout': 30,
    'query_timeout': 60,
    'max_connections': 100,
    'connection_pool_size': 10,
    'retry_attempts': 3,
    'backup_interval': 86400,  # 24 hours
    'cleanup_interval': 3600,  # 1 hour
    'index_optimization': True,
    'foreign_key_checks': True,
    'transaction_isolation': 'READ_COMMITTED'
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# UI Component Sizes
UI_SIZES = {
    'button_height': 40,
    'input_height': 40,
    'card_padding': 20,
    'card_margin': 15,
    'sidebar_width': 300,
    'header_height': 80,
    'footer_height': 60,
    'modal_width': 600,
    'modal_height': 400,
    'table_row_height': 45,
    'chart_margin': 20
}

# Validation Rules
VALIDATION_RULES = {
    'username': {
        'min_length': 2,
        'max_length': 50,
        'pattern': r'^[a-zA-Z0-9_\-\s\.]+',
        'required': True
    },
    'email': {
        'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'required': False
    },
    'message': {
        'min_length': 1,
        'max_length': 10000,
        'required': True
    },
    'file_name': {
        'min_length': 1,
        'max_length': 255,
        'pattern': r'^[^<>:"/\\|?*\x00-\x1f]+',
        'required': True
    }
}

# System Limits
SYSTEM_LIMITS = {
    'max_concurrent_users': 100,
    'max_file_uploads_per_hour': 10,
    'max_analysis_requests_per_hour': 20,
    'max_export_requests_per_hour': 5,
    'max_memory_usage_mb': 2048,
    'max_processing_time_seconds': 300,
    'max_database_connections': 50,
    'max_cache_size_mb': 512,
    'max_log_file_size_mb': 100,
    'max_temp_files': 1000
}

# Content Security Policies
CONTENT_SECURITY_POLICY = {
    'default-src': "'self'",
    'script-src': "'self' 'unsafe-inline' https://cdnjs.cloudflare.com",
    'style-src': "'self' 'unsafe-inline' https://fonts.googleapis.com",
    'font-src': "'self' https://fonts.gstatic.com",
    'img-src': "'self' data: https:",
    'connect-src': "'self'",
    'frame-ancestors': "'none'",
    'base-uri': "'self'",
    'form-action': "'self'"
}

# Help and Documentation
HELP_CONTENT = {
    'getting_started': {
        'title': 'Getting Started',
        'content': 'Learn how to upload and analyze your WhatsApp chat files',
        'url': '/help/getting-started'
    },
    'file_formats': {
        'title': 'Supported File Formats',
        'content': 'Understanding which chat export formats are supported',
        'url': '/help/file-formats'
    },
    'analysis_features': {
        'title': 'Analysis Features',
        'content': 'Comprehensive guide to all analysis capabilities',
        'url': '/help/analysis-features'
    },
    'troubleshooting': {
        'title': 'Troubleshooting',
        'content': 'Common issues and their solutions',
        'url': '/help/troubleshooting'
    },
    'privacy_security': {
        'title': 'Privacy & Security',
        'content': 'How your data is protected and processed',
        'url': '/help/privacy-security'
    },
    'api_documentation': {
        'title': 'API Documentation',
        'content': 'Developer guide for API integration',
        'url': '/help/api-docs'
    }
}

# Keyboard Shortcuts
KEYBOARD_SHORTCUTS = {
    'Ctrl+U': 'Upload file',
    'Ctrl+P': 'Process chat',
    'Ctrl+E': 'Export results',
    'Ctrl+R': 'Refresh analysis',
    'Ctrl+S': 'Save current view',
    'Ctrl+H': 'Show help',
    'Ctrl+K': 'Open command palette',
    'Esc': 'Close modals/dialogs',
    'F11': 'Toggle fullscreen',
    'Ctrl+Z': 'Undo last action'
}