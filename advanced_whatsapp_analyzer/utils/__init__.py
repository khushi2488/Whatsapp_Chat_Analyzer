"""
Utils package for Advanced WhatsApp Chat Analyzer
Provides utility functions, validation, security, and constants.
"""

from .helpers import (
    format_number,
    format_duration,
    calculate_growth_rate,
    get_time_greeting,
    calculate_text_statistics,
    extract_urls,
    extract_mentions,
    extract_emojis,
    calculate_readability_score,
    detect_language_simple,
    create_file_hash,
    safe_divide,
    normalize_username,
    calculate_activity_score,
    get_peak_activity_times,
    clean_message_text,
    calculate_response_patterns,
    generate_summary_stats,
    validate_dataframe
)

from .validators import (
    validate_file_type,
    validate_file_size,
    validate_whatsapp_format,
    validate_date_format,
    validate_user_name,
    validate_message_content,
    validate_dataframe_structure,
    validate_analysis_parameters,
    validate_export_request
)

from .security import (
    sanitize_user_input,
    validate_file_content,
    generate_session_token,
    hash_sensitive_data,
    encrypt_sensitive_data,
    decrypt_sensitive_data,
    anonymize_user_data,
    validate_user_permissions,
    rate_limit_check,
    sanitize_filename,
    validate_data_privacy,
    create_audit_log,
    secure_delete
)

from .constants import (
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    MAX_FILE_SIZE_MB,
    SUPPORTED_FILE_TYPES,
    WHATSAPP_PATTERNS,
    DATE_FORMATS,
    SYSTEM_MESSAGE_INDICATORS,
    MEDIA_INDICATORS,
    SUPPORTED_LANGUAGES,
    SENTIMENT_MODELS,
    COLOR_SCHEMES,
    RESPONSE_TIME_CATEGORIES,
    TIME_PERIODS,
    DAYS_OF_WEEK,
    EXPORT_FORMATS,
    ANALYSIS_THRESHOLDS,
    PERFORMANCE_THRESHOLDS,
    REGEX_PATTERNS,
    STOP_WORDS,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    DEFAULT_SETTINGS,
    FEATURE_FLAGS,
    CHART_CONFIG,
    VALIDATION_RULES,
    SYSTEM_LIMITS,
    HELP_CONTENT,
    KEYBOARD_SHORTCUTS
)

__version__ = "2.0.0"
__author__ = "AI Analytics Team"
__description__ = "Utility functions for Advanced WhatsApp Chat Analyzer"

# Package metadata
__all__ = [
    # Helper functions
    'format_number',
    'format_duration', 
    'calculate_growth_rate',
    'get_time_greeting',
    'calculate_text_statistics',
    'extract_urls',
    'extract_mentions',
    'extract_emojis',
    'calculate_readability_score',
    'detect_language_simple',
    'create_file_hash',
    'safe_divide',
    'normalize_username',
    'calculate_activity_score',
    'get_peak_activity_times',
    'clean_message_text',
    'calculate_response_patterns',
    'generate_summary_stats',
    'validate_dataframe',
    
    # Validation functions
    'validate_file_type',
    'validate_file_size',
    'validate_whatsapp_format',
    'validate_date_format',
    'validate_user_name',
    'validate_message_content',
    'validate_dataframe_structure',
    'validate_analysis_parameters',
    'validate_export_request',
    
    # Security functions
    'sanitize_user_input',
    'validate_file_content',
    'generate_session_token',
    'hash_sensitive_data',
    'encrypt_sensitive_data',
    'decrypt_sensitive_data',
    'anonymize_user_data',
    'validate_user_permissions',
    'rate_limit_check',
    'sanitize_filename',
    'validate_data_privacy',
    'create_audit_log',
    'secure_delete',
    
    # Constants
    'APP_NAME',
    'APP_VERSION',
    'APP_DESCRIPTION',
    'MAX_FILE_SIZE_MB',
    'SUPPORTED_FILE_TYPES',
    'WHATSAPP_PATTERNS',
    'DATE_FORMATS',
    'SYSTEM_MESSAGE_INDICATORS',
    'MEDIA_INDICATORS',
    'SUPPORTED_LANGUAGES',
    'SENTIMENT_MODELS',
    'COLOR_SCHEMES',
    'RESPONSE_TIME_CATEGORIES',
    'TIME_PERIODS',
    'DAYS_OF_WEEK',
    'EXPORT_FORMATS',
    'ANALYSIS_THRESHOLDS',
    'PERFORMANCE_THRESHOLDS',
    'REGEX_PATTERNS',
    'STOP_WORDS',
    'ERROR_MESSAGES',
    'SUCCESS_MESSAGES',
    'DEFAULT_SETTINGS',
    'FEATURE_FLAGS',
    'CHART_CONFIG',
    'VALIDATION_RULES',
    'SYSTEM_LIMITS',
    'HELP_CONTENT',
    'KEYBOARD_SHORTCUTS'
]