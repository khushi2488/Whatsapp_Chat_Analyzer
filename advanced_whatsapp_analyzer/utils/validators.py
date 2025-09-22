"""
Data Validation Module
Contains validation functions for data integrity and security.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import mimetypes
from urllib.parse import urlparse
import json

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class DataValidator:
    """Comprehensive data validation class"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
        # File validation settings
        self.max_file_size_mb = 100
        self.allowed_file_types = ['.txt', '.csv', '.xlsx']
        
        # Data validation settings
        self.min_messages = 5
        self.max_messages = 1000000
        self.min_users = 1
        self.max_users = 1000
        
        # Date validation settings
        self.min_date = datetime(2009, 1, 1)  # WhatsApp launch date
        self.max_date = datetime.now() + timedelta(days=1)
    
    def validate_file(self, file_content: Union[str, bytes], filename: str) -> Dict[str, Any]:
        """
        Validate uploaded file
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {},
            'security_issues': []
        }
        
        try:
            # File extension validation
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.allowed_file_types:
                validation_result['errors'].append(f"Unsupported file type: {file_ext}")
                validation_result['is_valid'] = False
            
            # File size validation
            if isinstance(file_content, bytes):
                file_size = len(file_content)
            else:
                file_size = len(file_content.encode('utf-8'))
            
            file_size_mb = file_size / (1024 * 1024)
            validation_result['file_info']['size_mb'] = round(file_size_mb, 2)
            
            if file_size_mb > self.max_file_size_mb:
                validation_result['errors'].append(f"File too large: {file_size_mb:.1f}MB (max: {self.max_file_size_mb}MB)")
                validation_result['is_valid'] = False
            
            # Content validation
            if isinstance(file_content, bytes):
                try:
                    content_str = file_content.decode('utf-8')
                except UnicodeDecodeError:
                    validation_result['errors'].append("File encoding error - please ensure UTF-8 encoding")
                    validation_result['is_valid'] = False
                    return validation_result
            else:
                content_str = file_content
            
            # Security checks
            security_issues = self._check_security_issues(content_str)
            validation_result['security_issues'] = security_issues
            
            # Content format validation for WhatsApp files
            if file_ext == '.txt':
                format_validation = self._validate_whatsapp_format(content_str)
                validation_result.update(format_validation)
            
            return validation_result
            
        except Exception as e:
            validation_result['errors'].append(f"File validation failed: {str(e)}")
            validation_result['is_valid'] = False
            return validation_result
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate processed DataFrame
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality': {},
            'suggestions': []
        }
        
        try:
            # Basic structure validation
            if df.empty:
                validation_result['errors'].append("DataFrame is empty")
                validation_result['is_valid'] = False
                return validation_result
            
            # Required columns validation
            required_columns = ['date', 'user', 'message']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
                validation_result['is_valid'] = False
            
            # Data size validation
            num_messages = len(df)
            if num_messages < self.min_messages:
                validation_result['warnings'].append(f"Very few messages ({num_messages}) - analysis may be limited")
            elif num_messages > self.max_messages:
                validation_result['warnings'].append(f"Large dataset ({num_messages}) - processing may be slow")
            
            # User validation
            if 'user' in df.columns:
                unique_users = df[df['user'] != 'system']['user'].nunique()
                if unique_users < self.min_users:
                    validation_result['warnings'].append("No active users found")
                elif unique_users > self.max_users:
                    validation_result['warnings'].append(f"Unusually high number of users ({unique_users})")
            
            # Date validation
            if 'date' in df.columns:
                date_issues = self._validate_dates(df['date'])
                if date_issues['errors']:
                    validation_result['errors'].extend(date_issues['errors'])
                    validation_result['is_valid'] = False
                if date_issues['warnings']:
                    validation_result['warnings'].extend(date_issues['warnings'])
            
            # Data quality assessment
            validation_result['data_quality'] = self._assess_data_quality(df)
            
            # Generate suggestions
            validation_result['suggestions'] = self._generate_suggestions(df, validation_result)
            
            return validation_result
            
        except Exception as e:
            validation_result['errors'].append(f"DataFrame validation failed: {str(e)}")
            validation_result['is_valid'] = False
            return validation_result
    
    def validate_user_input(self, user_input: str, input_type: str = 'general') -> Dict[str, Any]:
        """
        Validate user input for security and correctness
        """
        validation_result = {
            'is_valid': True,
            'cleaned_input': user_input,
            'errors': [],
            'warnings': []
        }
        
        try:
            if not user_input or not user_input.strip():
                validation_result['errors'].append("Input cannot be empty")
                validation_result['is_valid'] = False
                return validation_result
            
            # Security validation
            security_check = self._validate_input_security(user_input)
            if not security_check['is_safe']:
                validation_result['errors'].extend(security_check['issues'])
                validation_result['is_valid'] = False
            
            # Type-specific validation
            if input_type == 'username':
                username_validation = self._validate_username(user_input)
                validation_result.update(username_validation)
            elif input_type == 'email':
                email_validation = self._validate_email(user_input)
                validation_result.update(email_validation)
            elif input_type == 'date':
                date_validation = self._validate_date_string(user_input)
                validation_result.update(date_validation)
            
            return validation_result
            
        except Exception as e:
            validation_result['errors'].append(f"Input validation failed: {str(e)}")
            validation_result['is_valid'] = False
            return validation_result
    
    def _validate_whatsapp_format(self, content: str) -> Dict[str, Any]:
        """
        Validate WhatsApp chat format
        """
        format_result = {
            'format_valid': False,
            'format_detected': None,
            'sample_messages': 0,
            'format_confidence': 0.0
        }
        
        try:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            if len(lines) < 3:
                return format_result
            
            # WhatsApp format patterns
            patterns = {
                'format_1': r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s*(?:AM|PM)\s-\s(.+?):\s*(.*)',
                'format_2': r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)\]\s(.+?):\s*(.*)',
                'format_3': r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s(.+?):\s*(.*)',
                'format_4': r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\s-\s(.+?):\s*(.*)'
            }
            
            best_match_count = 0
            best_format = None
            
            # Check each format
            for format_name, pattern in patterns.items():
                match_count = 0
                for line in lines[:50]:  # Check first 50 lines
                    if re.match(pattern, line):
                        match_count += 1
                
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_format = format_name
            
            format_result['sample_messages'] = best_match_count
            format_result['format_confidence'] = min(best_match_count / min(len(lines), 50), 1.0)
            
            if best_match_count >= 3 and format_result['format_confidence'] > 0.1:
                format_result['format_valid'] = True
                format_result['format_detected'] = best_format
            
            return format_result
            
        except Exception as e:
            logger.error(f"Error validating WhatsApp format: {e}")
            return format_result
    
    def _check_security_issues(self, content: str) -> List[str]:
        """
        Check for potential security issues in file content
        """
        security_issues = []
        
        try:
            # Check for suspicious patterns
            suspicious_patterns = [
                (r'<script.*?>.*?</script>', 'JavaScript code detected'),
                (r'javascript:', 'JavaScript URL detected'),
                (r'data:.*?base64', 'Base64 encoded data detected'),
                (r'eval\s*\(', 'Eval function detected'),
                (r'exec\s*\(', 'Exec function detected'),
                (r'import\s+os|import\s+sys|import\s+subprocess', 'System imports detected')
            ]
            
            content_lower = content.lower()
            
            for pattern, message in suspicious_patterns:
                if re.search(pattern, content_lower, re.IGNORECASE | re.DOTALL):
                    security_issues.append(message)
            
            # Check for excessive special characters
            special_char_ratio = len(re.findall(r'[<>{}[\]\\|`~!@#$%^&*()_+=]', content)) / max(len(content), 1)
            if special_char_ratio > 0.1:
                security_issues.append(f"High special character ratio: {special_char_ratio:.1%}")
            
            return security_issues
            
        except Exception as e:
            logger.error(f"Error checking security issues: {e}")
            return ['Security check failed']
    
    def _validate_dates(self, date_series: pd.Series) -> Dict[str, List[str]]:
        """
        Validate date column
        """
        result = {'errors': [], 'warnings': []}
        
        try:
            # Check for null dates
            null_count = date_series.isnull().sum()
            if null_count > 0:
                result['warnings'].append(f"{null_count} messages have missing dates")
            
            # Check date range
            valid_dates = date_series.dropna()
            if len(valid_dates) == 0:
                result['errors'].append("No valid dates found")
                return result
            
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            
            # Check if dates are reasonable
            if min_date < self.min_date:
                result['warnings'].append(f"Dates before WhatsApp launch detected: {min_date.date()}")
            
            if max_date > self.max_date:
                result['warnings'].append(f"Future dates detected: {max_date.date()}")
            
            # Check for date consistency
            date_span = (max_date - min_date).days
            if date_span > 365 * 10:  # More than 10 years
                result['warnings'].append(f"Very long date range: {date_span} days")
            
            # Check for date gaps
            daily_counts = valid_dates.dt.date.value_counts().sort_index()
            if len(daily_counts) > 1:
                date_range = pd.date_range(daily_counts.index.min(), daily_counts.index.max())
                missing_days = len(date_range) - len(daily_counts)
                
                if missing_days > len(daily_counts) * 0.5:  # More than 50% missing days
                    result['warnings'].append(f"Many days with no messages: {missing_days} missing days")
            
            return result
            
        except Exception as e:
            result['errors'].append(f"Date validation failed: {str(e)}")
            return result
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess overall data quality
        """
        quality = {
            'score': 100.0,
            'completeness': {},
            'consistency': {},
            'validity': {},
            'issues': []
        }
        
        try:
            # Completeness assessment
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            completeness_score = ((total_cells - missing_cells) / total_cells) * 100
            
            quality['completeness'] = {
                'score': round(completeness_score, 2),
                'missing_cells': int(missing_cells),
                'total_cells': int(total_cells)
            }
            
            # Consistency assessment
            consistency_issues = 0
            
            # Check user name consistency
            if 'user' in df.columns:
                user_variations = self._find_user_variations(df['user'])
                if user_variations:
                    consistency_issues += len(user_variations)
                    quality['issues'].extend([f"Similar usernames found: {pair}" for pair in user_variations])
            
            # Check message type consistency
            if 'message_type' in df.columns:
                valid_types = {'text', 'media', 'system', 'deleted'}
                invalid_types = set(df['message_type'].unique()) - valid_types
                if invalid_types:
                    consistency_issues += len(invalid_types)
                    quality['issues'].append(f"Invalid message types: {invalid_types}")
            
            consistency_score = max(0, 100 - (consistency_issues * 5))
            quality['consistency'] = {
                'score': consistency_score,
                'issues_found': consistency_issues
            }
            
            # Validity assessment
            validity_issues = 0
            
            # Check for duplicate messages
            if len(df) > 0:
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    validity_issues += 1
                    quality['issues'].append(f"Found {duplicates} duplicate messages")
            
            # Check message length validity
            if 'message_length' in df.columns:
                invalid_lengths = len(df[df['message_length'] < 0])
                if invalid_lengths > 0:
                    validity_issues += 1
                    quality['issues'].append(f"Found {invalid_lengths} messages with negative length")
            
            validity_score = max(0, 100 - (validity_issues * 10))
            quality['validity'] = {
                'score': validity_score,
                'issues_found': validity_issues
            }
            
            # Calculate overall score
            quality['score'] = (completeness_score + consistency_score + validity_score) / 3
            
            return quality
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            quality['score'] = 0
            quality['issues'].append(f"Quality assessment failed: {str(e)}")
            return quality
    
    def _find_user_variations(self, user_series: pd.Series) -> List[Tuple[str, str]]:
        """
        Find similar usernames that might be variations of the same user
        """
        variations = []
        
        try:
            unique_users = user_series[user_series != 'system'].unique()
            
            for i, user1 in enumerate(unique_users):
                for user2 in unique_users[i+1:]:
                    # Simple similarity check
                    similarity = self._calculate_string_similarity(user1, user2)
                    if similarity > 0.8:  # 80% similarity threshold
                        variations.append((user1, user2))
            
            return variations
            
        except Exception as e:
            logger.error(f"Error finding user variations: {e}")
            return []
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using Levenshtein distance
        """
        try:
            str1, str2 = str1.lower(), str2.lower()
            
            if str1 == str2:
                return 1.0
            
            len1, len2 = len(str1), len(str2)
            if len1 == 0 or len2 == 0:
                return 0.0
            
            # Simple character overlap calculation
            set1, set2 = set(str1), set(str2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _generate_suggestions(self, df: pd.DataFrame, validation_result: Dict[str, Any]) -> List[str]:
        """
        Generate suggestions for data improvement
        """
        suggestions = []
        
        try:
            # Size-based suggestions
            if len(df) < 50:
                suggestions.append("Consider analyzing a larger chat for more meaningful insights")
            elif len(df) > 50000:
                suggestions.append("Large dataset detected - consider filtering by date range for faster processing")
            
            # Missing data suggestions
            if validation_result['data_quality']['completeness']['score'] < 80:
                suggestions.append("High amount of missing data - check data export process")
            
            # User suggestions
            if 'user' in df.columns:
                unique_users = df[df['user'] != 'system']['user'].nunique()
                if unique_users == 1:
                    suggestions.append("Individual chat detected - some group chat features won't be available")
                elif unique_users > 20:
                    suggestions.append("Large group chat - consider filtering by active users for clearer insights")
            
            # Date range suggestions
            if 'date' in df.columns:
                date_span = (df['date'].max() - df['date'].min()).days
                if date_span < 7:
                    suggestions.append("Short time period - trends analysis may be limited")
                elif date_span > 365:
                    suggestions.append("Long time period - consider date filtering for specific period analysis")
            
            # Performance suggestions
            if len(df) > 10000:
                suggestions.append("Enable caching for better performance with large datasets")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return ["Unable to generate suggestions due to analysis error"]
    
    def _validate_input_security(self, user_input: str) -> Dict[str, Any]:
        """
        Validate input for security issues
        """
        security_result = {
            'is_safe': True,
            'issues': []
        }
        
        try:
            # Check for SQL injection patterns
            sql_patterns = [
                r"('|(\\\')|(%27)|(\%27))",
                r"(\"|(\\\")|(\\%22))",
                r"(or|and)\s+\w*\s*=",
                r"union\s+select",
                r"drop\s+table",
                r"delete\s+from",
                r"insert\s+into"
            ]
            
            input_lower = user_input.lower()
            for pattern in sql_patterns:
                if re.search(pattern, input_lower):
                    security_result['is_safe'] = False
                    security_result['issues'].append("Potential SQL injection detected")
                    break
            
            # Check for XSS patterns
            xss_patterns = [
                r"<script.*?>",
                r"javascript:",
                r"on\w+\s*=",
                r"eval\s*\(",
                r"document\."
            ]
            
            for pattern in xss_patterns:
                if re.search(pattern, input_lower):
                    security_result['is_safe'] = False
                    security_result['issues'].append("Potential XSS detected")
                    break
            
            # Check input length
            if len(user_input) > 10000:
                security_result['is_safe'] = False
                security_result['issues'].append("Input too long")
            
            return security_result
            
        except Exception as e:
            security_result['is_safe'] = False
            security_result['issues'].append(f"Security validation failed: {str(e)}")
            return security_result
    
    def _validate_username(self, username: str) -> Dict[str, Any]:
        """
        Validate username format
        """
        result = {
            'is_valid': True,
            'cleaned_input': username.strip(),
            'errors': [],
            'warnings': []
        }
        
        try:
            username = username.strip()
            
            if len(username) < 1:
                result['errors'].append("Username cannot be empty")
                result['is_valid'] = False
            elif len(username) > 100:
                result['errors'].append("Username too long (max 100 characters)")
                result['is_valid'] = False
            
            # Check for invalid characters
            if re.search(r'[<>"\'\\/]', username):
                result['warnings'].append("Username contains special characters that may cause issues")
            
            result['cleaned_input'] = username
            return result
            
        except Exception as e:
            result['errors'].append(f"Username validation failed: {str(e)}")
            result['is_valid'] = False
            return result
    
    def _validate_email(self, email: str) -> Dict[str, Any]:
        """
        Validate email format
        """
        result = {
            'is_valid': True,
            'cleaned_input': email.strip().lower(),
            'errors': [],
            'warnings': []
        }
        
        try:
            email = email.strip().lower()
            
            # Basic email pattern
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            if not re.match(email_pattern, email):
                result['errors'].append("Invalid email format")
                result['is_valid'] = False
            
            result['cleaned_input'] = email
            return result
            
        except Exception as e:
            result['errors'].append(f"Email validation failed: {str(e)}")
            result['is_valid'] = False
            return result
    
    def _validate_date_string(self, date_str: str) -> Dict[str, Any]:
        """
        Validate date string format
        """
        result = {
            'is_valid': True,
            'cleaned_input': date_str.strip(),
            'errors': [],
            'warnings': []
        }
        
        try:
            date_str = date_str.strip()
            
            # Common date formats
            date_formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y %H:%M'
            ]
            
            parsed_date = None
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if parsed_date is None:
                result['errors'].append("Invalid date format")
                result['is_valid'] = False
            else:
                # Check if date is reasonable
                if parsed_date < self.min_date:
                    result['warnings'].append("Date is before WhatsApp launch")
                elif parsed_date > self.max_date:
                    result['warnings'].append("Future date detected")
            
            return result
            
        except Exception as e:
            result['errors'].append(f"Date validation failed: {str(e)}")
            result['is_valid'] = False
            return result

# Utility validation functions
def is_valid_whatsapp_export(content: str) -> bool:
    """
    Quick check if content looks like a WhatsApp export
    """
    try:
        validator = DataValidator()
        format_result = validator._validate_whatsapp_format(content)
        return format_result['format_valid']
    except Exception:
        return False

def sanitize_user_input(user_input: str, max_length: int = 1000) -> str:
    """
    Sanitize user input for safe processing
    """
    try:
        if not user_input:
            return ""
        
        # Remove potentially harmful characters
        sanitized = re.sub(r'[<>&"\'\x00-\x1f\x7f-\x9f]', '', str(user_input))
        
        # Trim to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
        
    except Exception:
        return ""

def validate_file_extension(filename: str, allowed_extensions: List[str] = None) -> bool:
    """
    Validate file extension
    """
    try:
        if allowed_extensions is None:
            allowed_extensions = ['.txt', '.csv', '.xlsx']
        
        file_ext = Path(filename).suffix.lower()
        return file_ext in allowed_extensions
        
    except Exception:
        return False

def check_data_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick data consistency check
    """
    try:
        validator = DataValidator()
        return validator._assess_data_quality(df)
    except Exception as e:
        return {'error': f"Consistency check failed: {str(e)}"}

def validate_environment() -> Dict[str, Any]:
    """
    Validate the application environment
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'system_info': {}
    }
    
    try:
        import sys
        import platform
        
        # Python version check
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            validation_result['errors'].append(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            validation_result['is_valid'] = False
        
        # Required packages check
        required_packages = [
            'pandas', 'numpy', 'streamlit', 'plotly', 'scikit-learn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            validation_result['errors'].append(f"Missing required packages: {missing_packages}")
            validation_result['is_valid'] = False
        
        # System info
        validation_result['system_info'] = {
            'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            'platform': platform.platform(),
            'architecture': platform.architecture()[0]
        }
        
        return validation_result
        
    except Exception as e:
        validation_result['errors'].append(f"Environment validation failed: {str(e)}")
        validation_result['is_valid'] = False
        return validation_result