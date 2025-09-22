"""
Helper functions for Advanced WhatsApp Chat Analyzer
Provides utility functions for data processing, formatting, and calculations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import re
import hashlib
import logging
from pathlib import Path
import emoji

logger = logging.getLogger(__name__)

def format_number(num: Union[int, float], decimal_places: int = 1) -> str:
    """Format numbers for display with appropriate units"""
    try:
        if pd.isna(num) or num is None:
            return "N/A"
        
        num = float(num)
        
        if abs(num) >= 1_000_000:
            return f"{num/1_000_000:.{decimal_places}f}M"
        elif abs(num) >= 1_000:
            return f"{num/1_000:.{decimal_places}f}K"
        else:
            return f"{num:.{decimal_places}f}"
    except (ValueError, TypeError):
        return "N/A"

def format_duration(days: int) -> str:
    """Format duration in days to human-readable format"""
    try:
        if days <= 0:
            return "0 days"
        elif days == 1:
            return "1 day"
        elif days < 7:
            return f"{days} days"
        elif days < 30:
            weeks = days // 7
            remaining_days = days % 7
            if remaining_days == 0:
                return f"{weeks} week{'s' if weeks > 1 else ''}"
            else:
                return f"{weeks} week{'s' if weeks > 1 else ''}, {remaining_days} day{'s' if remaining_days > 1 else ''}"
        elif days < 365:
            months = days // 30
            remaining_days = days % 30
            if remaining_days < 7:
                return f"{months} month{'s' if months > 1 else ''}"
            else:
                return f"{months} month{'s' if months > 1 else ''}, {remaining_days} days"
        else:
            years = days // 365
            remaining_days = days % 365
            if remaining_days < 30:
                return f"{years} year{'s' if years > 1 else ''}"
            else:
                months = remaining_days // 30
                return f"{years} year{'s' if years > 1 else ''}, {months} month{'s' if months > 1 else ''}"
    except (ValueError, TypeError):
        return "N/A"

def calculate_growth_rate(current: float, previous: float) -> Tuple[str, float]:
    """Calculate growth rate and return direction and percentage"""
    try:
        if previous == 0:
            if current > 0:
                return "↗", float('inf')
            else:
                return "→", 0.0
        
        change = ((current - previous) / previous) * 100
        
        if change > 0:
            return "↗", change
        elif change < 0:
            return "↘", abs(change)
        else:
            return "→", 0.0
    except (ValueError, TypeError, ZeroDivisionError):
        return "→", 0.0

def get_time_greeting() -> str:
    """Get appropriate greeting based on current time"""
    try:
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            return "Good morning"
        elif 12 <= current_hour < 17:
            return "Good afternoon"
        elif 17 <= current_hour < 21:
            return "Good evening"
        else:
            return "Good night"
    except Exception:
        return "Hello"

def calculate_text_statistics(text: str) -> Dict[str, Any]:
    """Calculate comprehensive text statistics"""
    try:
        if not text or pd.isna(text):
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'reading_time_minutes': 0,
                'complexity_score': 0
            }
        
        text = str(text).strip()
        
        # Basic counts
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Reading time (average 200 words per minute)
        reading_time_minutes = word_count / 200
        
        # Simple complexity score based on avg word length and sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        complexity_score = (avg_word_length * 0.4 + avg_sentence_length * 0.6) / 10
        complexity_score = min(complexity_score, 10)  # Cap at 10
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'reading_time_minutes': round(reading_time_minutes, 2),
            'complexity_score': round(complexity_score, 2)
        }
    
    except Exception as e:
        logger.error(f"Error calculating text statistics: {e}")
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'reading_time_minutes': 0,
            'complexity_score': 0
        }

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text"""
    try:
        if not text or pd.isna(text):
            return []
        
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        urls = url_pattern.findall(str(text))
        return urls
    
    except Exception as e:
        logger.error(f"Error extracting URLs: {e}")
        return []

def extract_mentions(text: str) -> List[str]:
    """Extract @mentions from text"""
    try:
        if not text or pd.isna(text):
            return []
        
        mention_pattern = re.compile(r'@(\w+)')
        mentions = mention_pattern.findall(str(text))
        return mentions
    
    except Exception as e:
        logger.error(f"Error extracting mentions: {e}")
        return []

def extract_emojis(text: str) -> List[str]:
    """Extract emojis from text"""
    try:
        if not text or pd.isna(text):
            return []
        
        emojis = [char for char in str(text) if char in emoji.EMOJI_DATA]
        return emojis
    
    except Exception as e:
        logger.error(f"Error extracting emojis: {e}")
        return []

def calculate_readability_score(text: str) -> float:
    """Calculate Flesch Reading Ease score"""
    try:
        if not text or pd.isna(text):
            return 0.0
        
        text = str(text).strip()
        
        # Count sentences, words, and syllables
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Approximate syllable counting
        syllables = 0
        for word in text.split():
            word = word.lower().strip('.,!?;:"')
            if word:
                syllable_count = max(1, len(re.findall(r'[aeiouAEIOU]', word)))
                if word.endswith('e'):
                    syllable_count -= 1
                syllables += max(1, syllable_count)
        
        # Flesch Reading Ease formula
        if sentences > 0 and words > 0:
            score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
            return max(0, min(100, score))  # Clamp between 0-100
        
        return 0.0
    
    except Exception as e:
        logger.error(f"Error calculating readability score: {e}")
        return 0.0

def detect_language_simple(text: str) -> str:
    """Simple language detection based on common words"""
    try:
        if not text or pd.isna(text):
            return 'unknown'
        
        text = str(text).lower()
        
        # Simple language indicators
        language_indicators = {
            'en': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with'],
            'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no'],
            'fr': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'],
            'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
            'pt': ['o', 'de', 'a', 'e', 'do', 'da', 'em', 'um', 'para', 'é'],
            'it': ['il', 'di', 'che', 'e', 'la', 'a', 'in', 'un', 'è', 'per']
        }
        
        scores = {}
        words = text.split()[:100]  # Check first 100 words
        
        for lang, indicators in language_indicators.items():
            score = sum(1 for word in words if word in indicators)
            scores[lang] = score
        
        if scores:
            detected_lang = max(scores, key=scores.get)
            if scores[detected_lang] > 0:
                return detected_lang
        
        return 'en'  # Default to English
    
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return 'unknown'

def create_file_hash(content: bytes) -> str:
    """Create MD5 hash of file content"""
    try:
        return hashlib.md5(content).hexdigest()
    except Exception as e:
        logger.error(f"Error creating file hash: {e}")
        return ""

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default

def normalize_username(username: str) -> str:
    """Normalize username for consistent analysis"""
    try:
        if not username or pd.isna(username):
            return 'Unknown'
        
        username = str(username).strip()
        
        # Remove phone numbers
        username = re.sub(r'\+?\d{10,15}', '', username)
        
        # Remove special characters but keep spaces and basic punctuation
        username = re.sub(r'[^\w\s\-_.]', '', username)
        
        # Normalize whitespace
        username = ' '.join(username.split())
        
        # Handle empty results
        if not username or username.lower() in ['', 'unknown', 'null', 'none']:
            return 'Unknown'
        
        return username
    
    except Exception as e:
        logger.error(f"Error normalizing username: {e}")
        return 'Unknown'

def calculate_activity_score(df: pd.DataFrame, user: str = None) -> Dict[str, float]:
    """Calculate activity score for user or overall chat"""
    try:
        if user:
            user_df = df[df['user'] == user]
        else:
            user_df = df[df['user'] != 'system']
        
        if user_df.empty:
            return {'activity_score': 0.0, 'consistency_score': 0.0, 'engagement_score': 0.0}
        
        # Activity metrics
        total_messages = len(user_df)
        days_active = user_df['date'].dt.date.nunique()
        avg_daily_messages = total_messages / days_active if days_active > 0 else 0
        
        # Consistency (based on daily activity variance)
        daily_counts = user_df.groupby(user_df['date'].dt.date).size()
        consistency_score = 1 - (daily_counts.std() / daily_counts.mean()) if daily_counts.mean() > 0 else 0
        consistency_score = max(0, min(1, consistency_score))
        
        # Engagement (based on message length, emojis, etc.)
        engagement_factors = []
        
        if 'message_length' in user_df.columns:
            avg_length = user_df['message_length'].mean()
            engagement_factors.append(min(avg_length / 100, 1))  # Normalize to 0-1
        
        if 'emoji_count' in user_df.columns:
            emoji_rate = len(user_df[user_df['emoji_count'] > 0]) / len(user_df)
            engagement_factors.append(emoji_rate)
        
        if 'url_count' in user_df.columns:
            url_rate = len(user_df[user_df['url_count'] > 0]) / len(user_df)
            engagement_factors.append(url_rate * 2)  # Weight URL sharing higher
        
        engagement_score = np.mean(engagement_factors) if engagement_factors else 0
        engagement_score = max(0, min(1, engagement_score))
        
        # Overall activity score (weighted combination)
        activity_score = (
            min(avg_daily_messages / 50, 1) * 0.4 +  # Daily activity (normalized to 50 msgs/day max)
            consistency_score * 0.3 +
            engagement_score * 0.3
        )
        
        return {
            'activity_score': round(activity_score, 3),
            'consistency_score': round(consistency_score, 3),
            'engagement_score': round(engagement_score, 3),
            'avg_daily_messages': round(avg_daily_messages, 1)
        }
    
    except Exception as e:
        logger.error(f"Error calculating activity score: {e}")
        return {'activity_score': 0.0, 'consistency_score': 0.0, 'engagement_score': 0.0}

def get_peak_activity_times(df: pd.DataFrame) -> Dict[str, Any]:
    """Get peak activity times and patterns"""
    try:
        if df.empty:
            return {}
        
        # Hourly analysis
        hourly_activity = df.groupby('hour').size()
        peak_hour = hourly_activity.idxmax()
        
        # Daily analysis
        if 'day_name' in df.columns:
            daily_activity = df.groupby('day_name').size()
            peak_day = daily_activity.idxmax()
        else:
            peak_day = 'Unknown'
        
        # Time period analysis
        if 'time_period' in df.columns:
            period_activity = df.groupby('time_period').size()
            peak_period = period_activity.idxmax()
        else:
            peak_period = 'Unknown'
        
        # Weekend vs weekday
        weekend_activity = 0
        weekday_activity = 0
        if 'is_weekend' in df.columns:
            weekend_activity = len(df[df['is_weekend'] == True])
            weekday_activity = len(df[df['is_weekend'] == False])
        
        return {
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'peak_period': peak_period,
            'weekend_percentage': weekend_activity / len(df) * 100 if len(df) > 0 else 0,
            'most_active_hours': hourly_activity.nlargest(3).index.tolist(),
            'least_active_hours': hourly_activity.nsmallest(3).index.tolist()
        }
    
    except Exception as e:
        logger.error(f"Error getting peak activity times: {e}")
        return {}

def clean_message_text(text: str) -> str:
    """Clean and normalize message text"""
    try:
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove or replace problematic characters
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        return text
    
    except Exception as e:
        logger.error(f"Error cleaning message text: {e}")
        return ""

def calculate_response_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate response time patterns and statistics"""
    try:
        if 'response_time_minutes' not in df.columns:
            return {}
        
        # Filter valid response times
        response_data = df[
            (df['response_time_minutes'].notna()) & 
            (df['response_time_minutes'] > 0) & 
            (df['response_time_minutes'] < 1440)  # Less than 24 hours
        ]
        
        if response_data.empty:
            return {}
        
        response_times = response_data['response_time_minutes']
        
        patterns = {
            'avg_response_time': response_times.mean(),
            'median_response_time': response_times.median(),
            'min_response_time': response_times.min(),
            'max_response_time': response_times.max(),
            'std_response_time': response_times.std(),
            'response_categories': {
                'immediate': len(response_data[response_data['response_time_minutes'] < 1]),
                'very_fast': len(response_data[
                    (response_data['response_time_minutes'] >= 1) & 
                    (response_data['response_time_minutes'] < 5)
                ]),
                'fast': len(response_data[
                    (response_data['response_time_minutes'] >= 5) & 
                    (response_data['response_time_minutes'] < 30)
                ]),
                'moderate': len(response_data[
                    (response_data['response_time_minutes'] >= 30) & 
                    (response_data['response_time_minutes'] < 240)
                ]),
                'slow': len(response_data[response_data['response_time_minutes'] >= 240])
            }
        }
        
        # Calculate percentiles
        patterns['percentiles'] = {
            'p25': response_times.quantile(0.25),
            'p50': response_times.quantile(0.50),
            'p75': response_times.quantile(0.75),
            'p90': response_times.quantile(0.90)
        }
        
        return patterns
    
    except Exception as e:
        logger.error(f"Error calculating response patterns: {e}")
        return {}

def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive summary statistics"""
    try:
        if df.empty:
            return {}
        
        stats = {
            'total_messages': len(df),
            'unique_users': df[df['user'] != 'system']['user'].nunique(),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'duration_days': (df['date'].max() - df['date'].min()).days + 1
            },
            'message_types': df['message_type'].value_counts().to_dict(),
            'daily_average': len(df) / ((df['date'].max() - df['date'].min()).days + 1)
        }
        
        # Content statistics
        text_df = df[df['message_type'] == 'text']
        if not text_df.empty:
            if 'message_length' in text_df.columns:
                stats['content_stats'] = {
                    'avg_message_length': text_df['message_length'].mean(),
                    'total_characters': text_df['message_length'].sum(),
                    'longest_message': text_df['message_length'].max()
                }
            
            if 'word_count' in text_df.columns:
                stats['word_stats'] = {
                    'total_words': text_df['word_count'].sum(),
                    'avg_words_per_message': text_df['word_count'].mean(),
                    'vocabulary_estimate': len(set(' '.join(text_df['message'].astype(str)).split()))
                }
        
        # Activity patterns
        stats['activity_patterns'] = get_peak_activity_times(df)
        
        # Response patterns
        stats['response_patterns'] = calculate_response_patterns(df)
        
        return stats
    
    except Exception as e:
        logger.error(f"Error generating summary stats: {e}")
        return {}

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate DataFrame structure and content"""
    try:
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Required columns
        required_columns = ['date', 'user', 'message', 'message_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation['errors'].append(f"Missing required columns: {missing_columns}")
            validation['valid'] = False
        
        # Data quality checks
        if df.empty:
            validation['errors'].append("DataFrame is empty")
            validation['valid'] = False
        else:
            # Check for null dates
            null_dates = df['date'].isnull().sum()
            if null_dates > 0:
                validation['warnings'].append(f"{null_dates} messages have null dates")
            
            # Check date range
            if 'date' in df.columns and not df['date'].isnull().all():
                date_range = df['date'].max() - df['date'].min()
                if date_range.days < 1:
                    validation['warnings'].append("Chat spans less than 1 day")
                elif date_range.days > 3650:  # 10 years
                    validation['warnings'].append("Chat spans more than 10 years - check date parsing")
            
            # Check user distribution
            if 'user' in df.columns:
                user_counts = df['user'].value_counts()
                if len(user_counts) == 1:
                    validation['warnings'].append("Only one user found in chat")
                elif user_counts.iloc[0] / len(df) > 0.95:
                    validation['warnings'].append("One user dominates 95%+ of messages")
        
        # Info about the dataset
        validation['info'] = {
            'total_messages': len(df),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        return validation
    
    except Exception as e:
        logger.error(f"Error validating DataFrame: {e}")
        return {
            'valid': False,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': [],
            'info': {}
        }