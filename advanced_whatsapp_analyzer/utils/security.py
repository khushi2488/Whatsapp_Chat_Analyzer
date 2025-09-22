"""
Security utilities for Advanced WhatsApp Chat Analyzer
Provides functions for input sanitization, data encryption, and security validation.
"""

import re
import html
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Union
import logging
from cryptography.fernet import Fernet
import base64
import os

logger = logging.getLogger(__name__)

# Security patterns for input sanitization
DANGEROUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',
    r'javascript:',
    r'on\w+\s*=',
    r'<iframe[^>]*>.*?</iframe>',
    r'<object[^>]*>.*?</object>',
    r'<embed[^>]*>.*?</embed>',
    r'<form[^>]*>.*?</form>',
    r'<input[^>]*>',
    r'<link[^>]*>',
    r'<meta[^>]*>',
]

def sanitize_user_input(text: str, max_length: int = 100000) -> str:
    """Sanitize user input to prevent XSS and injection attacks"""
    try:
        if not text or text is None:
            return ""
        
        # Convert to string and limit length
        text = str(text)[:max_length]
        
        # HTML escape
        text = html.escape(text, quote=True)
        
        # Remove potentially dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize whitespace but preserve line breaks
        text = re.sub(r'[ \t\r\f\v]+', ' ', text)
        
        return text.strip()
    
    except Exception as e:
        logger.error(f"Error sanitizing input: {e}")
        return ""

def validate_file_content(content: bytes, max_size: int = 50 * 1024 * 1024) -> Dict[str, Any]:
    """Validate uploaded file content for security"""
    result = {
        'safe': True,
        'issues': [],
        'warnings': []
    }
    
    try:
        if not content:
            result['safe'] = False
            result['issues'].append("Empty file content")
            return result
        
        # Check file size
        if len(content) > max_size:
            result['safe'] = False
            result['issues'].append(f"File too large: {len(content)} bytes (max: {max_size})")
            return result
        
        # Check for binary content patterns that might be suspicious
        try:
            # Try to decode as text
            text_content = content.decode('utf-8', errors='ignore')
            
            # Look for suspicious patterns
            suspicious_patterns = [
                b'\x00\x00',  # Null bytes
                b'MZ',        # Windows executable header
                b'\x7fELF',   # Linux executable header
                b'\x89PNG',   # PNG header (shouldn't be in chat)
                b'\xff\xd8',  # JPEG header
                b'PK\x03\x04' # ZIP header
            ]
            
            for pattern in suspicious_patterns:
                if pattern in content:
                    result['warnings'].append(f"Found suspicious binary pattern: {pattern}")
            
            # Check for executable patterns in text
            executable_patterns = [
                r'<%.*?%>',  # Server-side scripting
                r'exec\s*\(',
                r'eval\s*\(',
                r'system\s*\(',
                r'shell_exec\s*\('
            ]
            
            for pattern in executable_patterns:
                if re.search(pattern, text_content, re.IGNORECASE):
                    result['warnings'].append(f"Found potentially dangerous pattern: {pattern}")
        
        except UnicodeDecodeError:
            result['warnings'].append("File contains non-UTF-8 content")
        
        return result
    
    except Exception as e:
        logger.error(f"Error validating file content: {e}")
        result['safe'] = False
        result['issues'].append(f"Validation error: {str(e)}")
        return result

def generate_session_token() -> str:
    """Generate a secure session token"""
    try:
        return secrets.token_urlsafe(32)
    except Exception as e:
        logger.error(f"Error generating session token: {e}")
        return hashlib.sha256(os.urandom(32)).hexdigest()

def hash_sensitive_data(data: str, salt: Optional[str] = None) -> Dict[str, str]:
    """Hash sensitive data with salt"""
    try:
        if not salt:
            salt = secrets.token_hex(16)
        
        # Combine data with salt
        salted_data = f"{data}{salt}".encode('utf-8')
        
        # Create hash
        hash_object = hashlib.sha256(salted_data)
        hashed = hash_object.hexdigest()
        
        
    
    except Exception as e:
        logger.error(f"Error hashing data: {e}")
        return {
            'hash': '',
            'salt': ''
        }

def encrypt_sensitive_data(data: str, key: Optional[str] = None) -> Dict[str, str]:
    """Encrypt sensitive data using Fernet symmetric encryption"""
    try:
        if not key:
            # Generate a new key
            key = Fernet.generate_key()
        else:
            # Ensure key is bytes
            if isinstance(key, str):
                key = key.encode('utf-8')
        
        fernet = Fernet(key)
        
        # Encrypt the data
        encrypted_data = fernet.encrypt(data.encode('utf-8'))
        
        return {
            'encrypted': base64.b64encode(encrypted_data).decode('utf-8'),
            'key': base64.b64encode(key).decode('utf-8')
        }
    
    except Exception as e:
        logger.error(f"Error encrypting data: {e}")
        return {
            'encrypted': '',
            'key': ''
        }

def decrypt_sensitive_data(encrypted_data: str, key: str) -> str:
    """Decrypt sensitive data"""
    try:
        # Decode from base64
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        key_bytes = base64.b64decode(key.encode('utf-8'))
        
        fernet = Fernet(key_bytes)
        
        # Decrypt the data
        decrypted_data = fernet.decrypt(encrypted_bytes)
        
        return decrypted_data.decode('utf-8')
    
    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        return ""

def anonymize_user_data(data: Dict[str, Any], 
                       fields_to_anonymize: List[str] = None) -> Dict[str, Any]:
    """Anonymize sensitive user data"""
    try:
        if fields_to_anonymize is None:
            fields_to_anonymize = ['user', 'phone', 'email', 'name']
        
        anonymized_data = data.copy()
        
        # Generate consistent anonymized values
        user_mapping = {}
        
        for field in fields_to_anonymize:
            if field in anonymized_data:
                original_value = str(anonymized_data[field])
                
                # Use hash to create consistent anonymization
                if original_value not in user_mapping:
                    hash_value = hashlib.sha256(original_value.encode()).hexdigest()[:8]
                    user_mapping[original_value] = f"User_{hash_value}"
                
                anonymized_data[field] = user_mapping[original_value]
        
        return anonymized_data
    
    except Exception as e:
        logger.error(f"Error anonymizing data: {e}")
        return data

def validate_user_permissions(user_role: str, required_permission: str) -> bool:
    """Validate user permissions for specific actions"""
    try:
        # Define permission hierarchy
        permissions = {
            'admin': [
                'read_all', 'write_all', 'delete_all', 'export_all',
                'manage_users', 'view_analytics', 'system_config'
            ],
            'analyst': [
                'read_all', 'view_analytics', 'export_reports',
                'create_dashboards'
            ],
            'user': [
                'read_own', 'export_own', 'view_basic_analytics'
            ],
            'guest': [
                'read_sample', 'view_demo'
            ]
        }
        
        user_permissions = permissions.get(user_role.lower(), [])
        return required_permission in user_permissions
    
    except Exception as e:
        logger.error(f"Error validating permissions: {e}")
        return False

def rate_limit_check(user_id: str, action: str, 
                    limits: Dict[str, int] = None) -> Dict[str, Any]:
    """Check if user has exceeded rate limits"""
    result = {
        'allowed': True,
        'remaining': 0,
        'reset_time': None,
        'message': ''
    }
    
    try:
        if limits is None:
            limits = {
                'file_upload': 10,      # per hour
                'analysis_request': 20,  # per hour
                'export_request': 5      # per hour
            }
        
        # In a real implementation, this would check against a database
        # For now, we'll implement a simple in-memory check
        
        limit = limits.get(action, 100)
        result['remaining'] = limit  # Simplified - always allow
        result['message'] = f"Rate limit: {limit} {action}s per hour"
        
        return result
    
    except Exception as e:
        logger.error(f"Error checking rate limit: {e}")
        result['allowed'] = False
        result['message'] = "Rate limit check failed"
        return result

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and other issues"""
    try:
        if not filename:
            return "unnamed_file"
        
        # Remove directory separators and dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Prevent reserved names on Windows
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        name_without_ext = filename.rsplit('.', 1)[0].upper()
        if name_without_ext in reserved_names:
            filename = f"file_{filename}"
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_length = 255 - len(ext) - 1 if ext else 255
            filename = name[:max_name_length] + ('.' + ext if ext else '')
        
        # Ensure we have a valid filename
        if not filename or filename in ['.', '..']:
            filename = "unnamed_file"
        
        return filename
    
    except Exception as e:
        logger.error(f"Error sanitizing filename: {e}")
        return "unnamed_file"

def validate_data_privacy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data for privacy compliance (GDPR, etc.)"""
    result = {
        'compliant': True,
        'issues': [],
        'recommendations': []
    }
    
    try:
        # Check for potentially sensitive data
        sensitive_patterns = {
            'phone_number': r'\+?[\d\s\-\(\)]{10,}',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'credit_card': r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b'
        }
        
        for key, value in data.items():
            if isinstance(value, str):
                for pattern_name, pattern in sensitive_patterns.items():
                    if re.search(pattern, value):
                        result['issues'].append(f"Potential {pattern_name} found in field: {key}")
                        result['recommendations'].append(f"Consider anonymizing {pattern_name} in {key}")
        
        # Check for large datasets that might need consent
        if isinstance(data, dict) and 'messages' in data:
            message_count = len(data.get('messages', []))
            if message_count > 10000:
                result['recommendations'].append(
                    "Large dataset detected - ensure proper consent for processing"
                )
        
        return result
    
    except Exception as e:
        logger.error(f"Error validating data privacy: {e}")
        result['compliant'] = False
        result['issues'].append(f"Privacy validation error: {str(e)}")
        return result

def create_audit_log(action: str, user_id: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create audit log entry"""
    try:
        import time
        
        log_entry = {
            'timestamp': time.time(),
            'action': sanitize_user_input(action, 100),
            'user_id': sanitize_user_input(user_id, 50),
            'details': details or {},
            'session_id': generate_session_token()[:16]  # Truncated for logging
        }
        
        # In a real implementation, this would be stored in a secure audit database
        logger.info(f"Audit log: {log_entry}")
        
        return log_entry
    
    except Exception as e:
        logger.error(f"Error creating audit log: {e}")
        return {}

def secure_delete(data: Any) -> bool:
    """Securely delete sensitive data from memory"""
    try:
        if isinstance(data, str):
            # Overwrite string data (limited effectiveness in Python due to immutability)
            data = '0' * len(data)
        elif isinstance(data, (list, dict)):
            if isinstance(data, list):
                data.clear()
            else:
                data.clear()
        elif hasattr(data, 'clear'):
            data.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return True
    
    except Exception as e:
        logger.error(f"Error securely deleting data: {e}")
        return False