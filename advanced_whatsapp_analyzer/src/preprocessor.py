"""
Enhanced WhatsApp Chat Preprocessor with Advanced Features
This module handles parsing, cleaning, and preprocessing of WhatsApp chat data
with support for multiple formats, languages, and advanced analytics.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from langdetect import detect, DetectorFactory
import emoji
from urllib.parse import urlparse
import warnings

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ChatMetadata:
    """Metadata about the chat file"""
    total_messages: int
    date_range: Tuple[datetime, datetime]
    participants: List[str]
    language: str
    file_size: int
    chat_type: str  # 'group' or 'individual'
    platform: str = 'whatsapp'

@dataclass 
class ProcessingStats:
    """Statistics from preprocessing"""
    messages_processed: int
    messages_filtered: int
    media_messages: int
    deleted_messages: int
    system_messages: int
    processing_time: float
    errors: List[str]

class AdvancedPreprocessor:
    """Enhanced preprocessor for WhatsApp chat data"""
    
    def __init__(self):
        self.stats = ProcessingStats(0, 0, 0, 0, 0, 0.0, [])
        self.metadata = None
        
        # Comprehensive regex patterns for different WhatsApp formats
        self.patterns = {
            # Format: 1/1/23, 12:00 AM - User: Message
            'format1': r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s*(?:AM|PM))\s-\s([^:]+?):\s*(.*)',
            
            # Format: [1/1/23, 12:00:00 AM] User: Message
            'format2': r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM))\]\s([^:]+?):\s*(.*)',
            
            # Format: 1/1/23, 12:00 - User: Message (24-hour)
            'format3': r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2})\s-\s([^:]+?):\s*(.*)',
            
            # Format: 2023-01-01 12:00:00 - User: Message (ISO format)
            'format4': r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s-\s([^:]+?):\s*(.*)',
            
            # System messages
            'system': r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s*(?:AM|PM)?)\s-\s(.*?)(?:\s-\s(.*))?'
        }
        
        # Common system message indicators
        self.system_indicators = [
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
            'This message was deleted'
        ]
        
        # Media indicators
        self.media_indicators = [
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
            '📄 Document'
        ]
        
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

    def detect_chat_format(self, text: str) -> str:
        """Detect the format of the WhatsApp chat file"""
        sample_lines = text.split('\n')[:50]  # Check first 50 lines
        
        format_scores = {}
        for format_name, pattern in self.patterns.items():
            if format_name == 'system':
                continue
            score = sum(1 for line in sample_lines if re.match(pattern, line))
            format_scores[format_name] = score
        
        if not format_scores or max(format_scores.values()) == 0:
            raise ValueError("Unable to detect WhatsApp chat format. Please check the file format.")
        
        best_format = max(format_scores, key=format_scores.get)
        logger.info(f"Detected chat format: {best_format}")
        return best_format

    def extract_metadata(self, df: pd.DataFrame, file_size: int = 0) -> ChatMetadata:
        """Extract metadata from the processed dataframe"""
        try:
            # Basic stats
            total_messages = len(df)
            date_range = (df['date'].min(), df['date'].max())
            participants = df[df['user'] != 'system']['user'].unique().tolist()
            
            # Detect language from a sample of messages
            sample_messages = df[df['message_type'] == 'text']['message'].dropna().head(100)
            language = self._detect_language(sample_messages.tolist())
            
            # Determine chat type
            chat_type = 'group' if len(participants) > 2 else 'individual'
            
            metadata = ChatMetadata(
                total_messages=total_messages,
                date_range=date_range,
                participants=participants,
                language=language,
                file_size=file_size,
                chat_type=chat_type
            )
            
            logger.info(f"Chat metadata: {len(participants)} participants, {total_messages} messages, language: {language}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return None

    def _detect_language(self, messages: List[str]) -> str:
        """Detect the primary language of the chat"""
        try:
            # Combine first 1000 characters from messages
            text_sample = ' '.join(messages)[:1000]
            if len(text_sample.strip()) < 50:
                return 'en'  # Default to English
            
            detected = detect(text_sample)
            return detected
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en'  # Default to English

    def parse_messages(self, text: str) -> pd.DataFrame:
        """Parse WhatsApp chat text into structured DataFrame"""
        start_time = datetime.now()
        
        try:
            # Detect format
            chat_format = self.detect_chat_format(text)
            pattern = self.patterns[chat_format]
            
            # Split text into lines
            lines = text.strip().split('\n')
            
            messages_data = []
            current_message = None
            
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Try to match message pattern
                match = re.match(pattern, line)
                
                if match:
                    # Save previous message if exists
                    if current_message:
                        messages_data.append(current_message)
                    
                    # Start new message
                    if chat_format in ['format1', 'format3']:
                        timestamp_str, user, message = match.groups()
                    elif chat_format == 'format2':
                        timestamp_str, user, message = match.groups()
                    elif chat_format == 'format4':
                        timestamp_str, user, message = match.groups()
                    
                    current_message = {
                        'timestamp_str': timestamp_str,
                        'user': user.strip(),
                        'message': message.strip() if message else '',
                        'line_number': line_num + 1
                    }
                    
                else:
                    # Check if it's a system message
                    system_match = re.match(self.patterns['system'], line)
                    if system_match and any(indicator in line.lower() for indicator in self.system_indicators):
                        if current_message:
                            messages_data.append(current_message)
                        
                        current_message = {
                            'timestamp_str': system_match.group(1),
                            'user': 'system',
                            'message': line,
                            'line_number': line_num + 1
                        }
                    
                    elif current_message:
                        # Continuation of previous message
                        current_message['message'] += '\n' + line
            
            # Don't forget the last message
            if current_message:
                messages_data.append(current_message)
            
            # Convert to DataFrame
            df = pd.DataFrame(messages_data)
            
            if df.empty:
                raise ValueError("No messages found in the chat file")
            
            # Parse timestamps
            df = self._parse_timestamps(df, chat_format)
            
            # Clean and categorize messages
            df = self._clean_and_categorize(df)
            
            # Extract additional features
            df = self._extract_features(df)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats = ProcessingStats(
                messages_processed=len(df),
                messages_filtered=len(df[df['message_type'] != 'text']),
                media_messages=len(df[df['message_type'] == 'media']),
                deleted_messages=len(df[df['message_type'] == 'deleted']),
                system_messages=len(df[df['user'] == 'system']),
                processing_time=processing_time,
                errors=[]
            )
            
            logger.info(f"Successfully processed {len(df)} messages in {processing_time:.2f} seconds")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing messages: {e}")
            self.stats.errors.append(str(e))
            raise

    def _parse_timestamps(self, df: pd.DataFrame, chat_format: str) -> pd.DataFrame:
        """Parse timestamp strings to datetime objects"""
        try:
            if chat_format == 'format1':
                # Format: 1/1/23, 12:00 AM
                df['date'] = pd.to_datetime(df['timestamp_str'], format='%m/%d/%y, %I:%M %p', errors='coerce')
            elif chat_format == 'format2':
                # Format: 1/1/23, 12:00:00 AM
                df['date'] = pd.to_datetime(df['timestamp_str'], format='%m/%d/%y, %I:%M:%S %p', errors='coerce')
            elif chat_format == 'format3':
                # Format: 1/1/23, 12:00 (24-hour)
                df['date'] = pd.to_datetime(df['timestamp_str'], format='%m/%d/%y, %H:%M', errors='coerce')
            elif chat_format == 'format4':
                # Format: 2023-01-01 12:00:00
                df['date'] = pd.to_datetime(df['timestamp_str'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            
            # Handle failed parsing
            failed_parsing = df['date'].isna()
            if failed_parsing.any():
                logger.warning(f"Failed to parse {failed_parsing.sum()} timestamps")
                # Try alternative formats
                for idx in df[failed_parsing].index:
                    timestamp_str = df.loc[idx, 'timestamp_str']
                    try:
                        # Try different common formats
                        for fmt in ['%d/%m/%y, %H:%M', '%m/%d/%Y, %I:%M %p', '%d/%m/%Y, %H:%M']:
                            try:
                                df.loc[idx, 'date'] = pd.to_datetime(timestamp_str, format=fmt)
                                break
                            except ValueError:
                                continue
                    except Exception:
                        logger.warning(f"Could not parse timestamp: {timestamp_str}")
            
            # Remove messages with invalid dates
            df = df.dropna(subset=['date'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing timestamps: {e}")
            raise

    def _clean_and_categorize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean messages and categorize them by type"""
        try:
            # Initialize message type
            df['message_type'] = 'text'
            df['is_media'] = False
            df['is_deleted'] = False
            df['is_system'] = False
            
            # Categorize messages
            for idx, row in df.iterrows():
                message = str(row['message']).strip()
                user = str(row['user']).strip()
                
                # System messages
                if user == 'system' or any(indicator in message.lower() for indicator in self.system_indicators):
                    df.loc[idx, 'message_type'] = 'system'
                    df.loc[idx, 'is_system'] = True
                    df.loc[idx, 'user'] = 'system'
                
                # Media messages
                elif any(media in message for media in self.media_indicators):
                    df.loc[idx, 'message_type'] = 'media'
                    df.loc[idx, 'is_media'] = True
                
                # Deleted messages
                elif 'deleted this message' in message.lower() or 'message was deleted' in message.lower():
                    df.loc[idx, 'message_type'] = 'deleted'
                    df.loc[idx, 'is_deleted'] = True
                
                # Regular text messages
                else:
                    df.loc[idx, 'message_type'] = 'text'
            
            # Clean user names
            df['user'] = df['user'].apply(self._clean_username)
            
            logger.info(f"Message categorization: {df['message_type'].value_counts().to_dict()}")
            return df
            
        except Exception as e:
            logger.error(f"Error in cleaning and categorization: {e}")
            return df

    def _clean_username(self, username: str) -> str:
        """Clean and standardize usernames"""
        if pd.isna(username):
            return 'Unknown'
        
        username = str(username).strip()
        
        # Remove phone number patterns
        username = re.sub(r'\+?\d{10,15}', '', username)
        
        # Remove extra whitespace
        username = ' '.join(username.split())
        
        # Handle empty usernames
        if not username or username.lower() in ['', 'unknown', 'null']:
            return 'Unknown'
        
        return username

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional features from messages"""
        try:
            # Time-based features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['month_name'] = df['date'].dt.month_name()
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.day_of_week
            df['day_name'] = df['date'].dt.day_name()
            df['hour'] = df['date'].dt.hour
            df['minute'] = df['date'].dt.minute
            df['date_only'] = df['date'].dt.date
            
            # Create time periods
            df['time_period'] = df['hour'].apply(self._get_time_period)
            df['hour_period'] = df['hour'].apply(lambda x: f"{x}-{x+1}")
            
            # Weekend/weekday
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
            
            # Message features (only for text messages)
            text_mask = df['message_type'] == 'text'
            
            df['message_length'] = 0
            df['word_count'] = 0
            df['emoji_count'] = 0
            df['url_count'] = 0
            df['has_emoji'] = False
            df['has_url'] = False
            df['has_mention'] = False
            
            # Extract features for text messages
            for idx in df[text_mask].index:
                message = str(df.loc[idx, 'message'])
                
                # Basic counts
                df.loc[idx, 'message_length'] = len(message)
                df.loc[idx, 'word_count'] = len(message.split()) if message.strip() else 0
                
                # Emoji analysis
                emoji_count = len([c for c in message if c in emoji.EMOJI_DATA])
                df.loc[idx, 'emoji_count'] = emoji_count
                df.loc[idx, 'has_emoji'] = emoji_count > 0
                
                # URL analysis
                urls = self.url_pattern.findall(message)
                df.loc[idx, 'url_count'] = len(urls)
                df.loc[idx, 'has_url'] = len(urls) > 0
                
                # Mention analysis (@username)
                df.loc[idx, 'has_mention'] = '@' in message and len(re.findall(r'@\w+', message)) > 0
            
            # Response time analysis
            df = self._calculate_response_times(df)
            
            logger.info("Successfully extracted message features")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return df

    def _get_time_period(self, hour: int) -> str:
        """Convert hour to time period"""
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    def _calculate_response_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate response times between messages"""
        try:
            df = df.sort_values('date').reset_index(drop=True)
            df['response_time_minutes'] = np.nan
            df['response_time_category'] = 'unknown'
            
            for i in range(1, len(df)):
                current_user = df.loc[i, 'user']
                prev_user = df.loc[i-1, 'user']
                
                # Only calculate if different users and not system messages
                if (current_user != prev_user and 
                    current_user != 'system' and 
                    prev_user != 'system'):
                    
                    time_diff = df.loc[i, 'date'] - df.loc[i-1, 'date']
                    minutes = time_diff.total_seconds() / 60
                    
                    df.loc[i, 'response_time_minutes'] = minutes
                    
                    # Categorize response time
                    if minutes < 1:
                        category = 'immediate'
                    elif minutes < 5:
                        category = 'very_fast'
                    elif minutes < 30:
                        category = 'fast'
                    elif minutes < 60:
                        category = 'moderate'
                    elif minutes < 240:
                        category = 'slow'
                    else:
                        category = 'very_slow'
                    
                    df.loc[i, 'response_time_category'] = category
            
            logger.info("Response time analysis completed")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating response times: {e}")
            return df

    def preprocess(self, data: str, file_size: int = 0) -> pd.DataFrame:
        """Main preprocessing function - entry point"""
        try:
            logger.info("Starting advanced preprocessing...")
            
            # Parse messages
            df = self.parse_messages(data)
            
            # Extract metadata
            self.metadata = self.extract_metadata(df, file_size)
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Add sequential message ID
            df['message_id'] = range(1, len(df) + 1)
            
            logger.info(f"✅ Preprocessing completed: {len(df)} messages processed")
            
            return df
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing results"""
        return {
            'stats': {
                'messages_processed': self.stats.messages_processed,
                'messages_filtered': self.stats.messages_filtered,
                'media_messages': self.stats.media_messages,
                'deleted_messages': self.stats.deleted_messages,
                'system_messages': self.stats.system_messages,
                'processing_time': f"{self.stats.processing_time:.2f}s",
                'errors': self.stats.errors
            },
            'metadata': {
                'total_messages': self.metadata.total_messages if self.metadata else 0,
                'participants': len(self.metadata.participants) if self.metadata else 0,
                'date_range': f"{self.metadata.date_range[0].date()} to {self.metadata.date_range[1].date()}" if self.metadata else "N/A",
                'language': self.metadata.language if self.metadata else "unknown",
                'chat_type': self.metadata.chat_type if self.metadata else "unknown",
                'duration_days': (self.metadata.date_range[1] - self.metadata.date_range[0]).days if self.metadata else 0
            }
        }

# Legacy function for backward compatibility
def preprocess(data: str) -> pd.DataFrame:
    """Legacy preprocessing function for backward compatibility"""
    preprocessor = AdvancedPreprocessor()
    return preprocessor.preprocess(data)