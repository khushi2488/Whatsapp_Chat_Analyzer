"""
Core Analysis Engine for Advanced WhatsApp Chat Analyzer
Provides comprehensive analysis capabilities including statistical analysis,
pattern detection, and advanced insights generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import warnings
from dataclasses import dataclass
import re

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    overview: Dict[str, Any]
    user_analysis: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    content_analysis: Dict[str, Any]
    interaction_analysis: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]

class AdvancedAnalyzer:
    """Core analysis engine with advanced analytics capabilities"""
    
    def __init__(self):
        self.cache = {}
        self.analysis_results = None
        
    def analyze_chat(self, df: pd.DataFrame) -> AnalysisResult:
        """Perform comprehensive chat analysis"""
        try:
            logger.info(f"Starting comprehensive analysis of {len(df)} messages...")
            
            if df.empty:
                return self._empty_result()
            
            # Perform different types of analysis
            overview = self._analyze_overview(df)
            user_analysis = self._analyze_users(df)
            temporal_analysis = self._analyze_temporal_patterns(df)
            content_analysis = self._analyze_content(df)
            interaction_analysis = self._analyze_interactions(df)
            
            # Generate insights and recommendations
            insights = self._generate_insights(df, overview, user_analysis, temporal_analysis)
            recommendations = self._generate_recommendations(df, insights)
            
            result = AnalysisResult(
                overview=overview,
                user_analysis=user_analysis,
                temporal_analysis=temporal_analysis,
                content_analysis=content_analysis,
                interaction_analysis=interaction_analysis,
                insights=insights,
                recommendations=recommendations
            )
            
            self.analysis_results = result
            logger.info("Comprehensive analysis completed")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self._empty_result()
    
    def _analyze_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate overview statistics"""
        try:
            overview = {}
            
            # Basic counts
            overview['total_messages'] = len(df)
            overview['unique_users'] = df[df['user'] != 'system']['user'].nunique()
            overview['date_range'] = {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'duration_days': (df['date'].max() - df['date'].min()).days + 1
            }
            
            # Message type breakdown
            overview['message_types'] = df['message_type'].value_counts().to_dict()
            
            # Activity metrics
            overview['activity_metrics'] = {
                'avg_daily_messages': overview['total_messages'] / overview['date_range']['duration_days'],
                'messages_per_user': overview['total_messages'] / overview['unique_users'] if overview['unique_users'] > 0 else 0,
                'peak_activity_hour': df.groupby('hour').size().idxmax(),
                'peak_activity_day': df.groupby('day_name').size().idxmax()
            }
            
            # Content metrics
            if 'message_length' in df.columns:
                text_df = df[df['message_type'] == 'text']
                overview['content_metrics'] = {
                    'avg_message_length': text_df['message_length'].mean(),
                    'total_characters': text_df['message_length'].sum(),
                    'total_words': text_df['word_count'].sum() if 'word_count' in text_df.columns else 0,
                    'avg_words_per_message': text_df['word_count'].mean() if 'word_count' in text_df.columns else 0
                }
            
            # Engagement metrics
            if 'emoji_count' in df.columns:
                overview['engagement_metrics'] = {
                    'total_emojis': df['emoji_count'].sum(),
                    'messages_with_emojis': len(df[df['emoji_count'] > 0]),
                    'emoji_usage_rate': len(df[df['emoji_count'] > 0]) / len(df) * 100
                }
            
            if 'url_count' in df.columns:
                overview['sharing_metrics'] = {
                    'total_urls': df['url_count'].sum(),
                    'messages_with_urls': len(df[df['url_count'] > 0]),
                    'url_sharing_rate': len(df[df['url_count'] > 0]) / len(df) * 100
                }
            
            return overview
            
        except Exception as e:
            logger.error(f"Overview analysis failed: {e}")
            return {}
    
    def _analyze_users(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user behavior and patterns"""
        try:
            user_analysis = {}
            
            # Filter non-system users
            user_df = df[df['user'] != 'system']
            
            if user_df.empty:
                return {}
            
            # User activity statistics
            user_stats = {}
            total_messages = len(user_df)
            
            for user in user_df['user'].unique():
                user_messages = user_df[user_df['user'] == user]
                
                stats = {
                    'message_count': len(user_messages),
                    'percentage': len(user_messages) / total_messages * 100,
                    'avg_message_length': user_messages['message_length'].mean() if 'message_length' in user_messages.columns else 0,
                    'total_words': user_messages['word_count'].sum() if 'word_count' in user_messages.columns else 0,
                    'most_active_hour': user_messages.groupby('hour').size().idxmax(),
                    'most_active_day': user_messages.groupby('day_name').size().idxmax(),
                    'weekend_activity': len(user_messages[user_messages['is_weekend'] == True]) if 'is_weekend' in user_messages.columns else 0,
                    'first_message': user_messages['date'].min(),
                    'last_message': user_messages['date'].max()
                }
                
                # Engagement metrics per user
                if 'emoji_count' in user_messages.columns:
                    stats['emoji_usage'] = {
                        'total_emojis': user_messages['emoji_count'].sum(),
                        'messages_with_emojis': len(user_messages[user_messages['emoji_count'] > 0]),
                        'emoji_rate': len(user_messages[user_messages['emoji_count'] > 0]) / len(user_messages) * 100
                    }
                
                if 'url_count' in user_messages.columns:
                    stats['url_sharing'] = {
                        'total_urls': user_messages['url_count'].sum(),
                        'messages_with_urls': len(user_messages[user_messages['url_count'] > 0]),
                        'sharing_rate': len(user_messages[user_messages['url_count'] > 0]) / len(user_messages) * 100
                    }
                
                # Response time analysis (if available)
                if 'response_time_minutes' in user_messages.columns:
                    valid_responses = user_messages[user_messages['response_time_minutes'].notna()]
                    if not valid_responses.empty:
                        stats['response_patterns'] = {
                            'avg_response_time': valid_responses['response_time_minutes'].mean(),
                            'median_response_time': valid_responses['response_time_minutes'].median(),
                            'fast_responses': len(valid_responses[valid_responses['response_time_minutes'] < 5]),
                            'fast_response_rate': len(valid_responses[valid_responses['response_time_minutes'] < 5]) / len(valid_responses) * 100
                        }
                
                # Sentiment analysis per user (if available)
                if 'sentiment_label' in user_messages.columns:
                    sentiment_data = user_messages[user_messages['sentiment_label'].notna()]
                    if not sentiment_data.empty:
                        sentiment_counts = sentiment_data['sentiment_label'].value_counts()
                        stats['sentiment_profile'] = {
                            'positive_pct': sentiment_counts.get('positive', 0) / len(sentiment_data) * 100,
                            'negative_pct': sentiment_counts.get('negative', 0) / len(sentiment_data) * 100,
                            'neutral_pct': sentiment_counts.get('neutral', 0) / len(sentiment_data) * 100,
                            'avg_sentiment_score': sentiment_data['compound_score'].mean() if 'compound_score' in sentiment_data.columns else 0
                        }
                
                user_stats[user] = stats
            
            # Rank users by activity
            sorted_users = sorted(user_stats.items(), key=lambda x: x[1]['message_count'], reverse=True)
            user_analysis['user_rankings'] = [{'user': user, **stats} for user, stats in sorted_users]
            
            # User interaction patterns
            user_analysis['interaction_matrix'] = self._calculate_interaction_matrix(user_df)
            
            # User activity patterns
            user_analysis['activity_patterns'] = self._analyze_user_activity_patterns(user_df)
            
            return user_analysis
            
        except Exception as e:
            logger.error(f"User analysis failed: {e}")
            return {}
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in the chat"""
        try:
            temporal = {}
            
            # Daily patterns
            daily_activity = df.groupby(df['date'].dt.date).size()
            temporal['daily_patterns'] = {
                'daily_counts': daily_activity.to_dict(),
                'avg_daily_messages': daily_activity.mean(),
                'peak_day': daily_activity.idxmax(),
                'quiet_day': daily_activity.idxmin(),
                'activity_variance': daily_activity.var()
            }
            
            # Hourly patterns
            hourly_activity = df.groupby('hour').size()
            temporal['hourly_patterns'] = {
                'hourly_distribution': hourly_activity.to_dict(),
                'peak_hours': hourly_activity.nlargest(3).index.tolist(),
                'quiet_hours': hourly_activity.nsmallest(3).index.tolist(),
                'business_hours_activity': hourly_activity[9:17].sum(),  # 9 AM to 5 PM
                'after_hours_activity': hourly_activity[17:24].sum() + hourly_activity[0:9].sum()
            }
            
            # Weekly patterns
            if 'day_name' in df.columns:
                weekly_activity = df.groupby('day_name').size()
                weekend_activity = df[df['is_weekend'] == True] if 'is_weekend' in df.columns else pd.DataFrame()
                
                temporal['weekly_patterns'] = {
                    'weekly_distribution': weekly_activity.to_dict(),
                    'most_active_day': weekly_activity.idxmax(),
                    'least_active_day': weekly_activity.idxmin(),
                    'weekend_vs_weekday': {
                        'weekend_messages': len(weekend_activity),
                        'weekday_messages': len(df) - len(weekend_activity),
                        'weekend_percentage': len(weekend_activity) / len(df) * 100 if len(df) > 0 else 0
                    }
                }
            
            # Monthly patterns (if data spans multiple months)
            monthly_activity = df.groupby(df['date'].dt.to_period('M')).size()
            if len(monthly_activity) > 1:
                temporal['monthly_patterns'] = {
                    'monthly_distribution': {str(k): v for k, v in monthly_activity.to_dict().items()},
                    'growth_trend': self._calculate_trend(monthly_activity),
                    'peak_month': str(monthly_activity.idxmax()),
                    'quiet_month': str(monthly_activity.idxmin())
                }
            
            # Activity bursts and quiet periods
            temporal['activity_anomalies'] = self._detect_activity_anomalies(df)
            
            # Time-based content patterns
            temporal['content_timing'] = self._analyze_content_timing(df)
            
            return temporal
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            return {}
    
    def _analyze_content(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze content patterns and characteristics"""
        try:
            content = {}
            
            # Filter text messages
            text_df = df[df['message_type'] == 'text'].copy()
            
            if text_df.empty:
                return content
            
            # Message length analysis
            if 'message_length' in text_df.columns:
                content['length_analysis'] = {
                    'avg_length': text_df['message_length'].mean(),
                    'median_length': text_df['message_length'].median(),
                    'length_distribution': {
                        'short': len(text_df[text_df['message_length'] < 50]),
                        'medium': len(text_df[(text_df['message_length'] >= 50) & (text_df['message_length'] < 200)]),
                        'long': len(text_df[text_df['message_length'] >= 200])
                    },
                    'longest_message': {
                        'length': text_df['message_length'].max(),
                        'user': text_df.loc[text_df['message_length'].idxmax(), 'user'],
                        'date': text_df.loc[text_df['message_length'].idxmax(), 'date']
                    }
                }
            
            # Word usage analysis
            if 'word_count' in text_df.columns:
                content['word_analysis'] = {
                    'total_words': text_df['word_count'].sum(),
                    'avg_words_per_message': text_df['word_count'].mean(),
                    'vocabulary_richness': self._calculate_vocabulary_richness(text_df),
                    'most_verbose_user': text_df.groupby('user')['word_count'].sum().idxmax()
                }
            
            # Emoji usage patterns
            if 'emoji_count' in df.columns:
                content['emoji_patterns'] = self._analyze_emoji_patterns(df)
            
            # URL and link sharing
            if 'url_count' in df.columns:
                content['link_sharing'] = {
                    'total_links': df['url_count'].sum(),
                    'users_sharing_links': len(df[df['url_count'] > 0]['user'].unique()),
                    'most_shared_by': df[df['url_count'] > 0].groupby('user')['url_count'].sum().idxmax() if df['url_count'].sum() > 0 else None
                }
            
            # Question and interaction patterns
            if 'is_question' in df.columns:
                content['interaction_patterns'] = {
                    'total_questions': len(df[df['is_question'] == True]),
                    'question_rate': len(df[df['is_question'] == True]) / len(df) * 100,
                    'most_inquisitive_user': df[df['is_question'] == True]['user'].value_counts().index[0] if len(df[df['is_question'] == True]) > 0 else None
                }
            
            # Communication style analysis
            content['communication_style'] = self._analyze_communication_style(text_df)
            
            return content
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {}
    
    def _analyze_interactions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user interactions and conversation dynamics"""
        try:
            interactions = {}
            
            # Response time analysis
            if 'response_time_minutes' in df.columns:
                response_data = df[df['response_time_minutes'].notna()]
                
                if not response_data.empty:
                    interactions['response_times'] = {
                        'overall_avg': response_data['response_time_minutes'].mean(),
                        'overall_median': response_data['response_time_minutes'].median(),
                        'fast_response_rate': len(response_data[response_data['response_time_minutes'] < 5]) / len(response_data) * 100,
                        'response_time_distribution': {
                            'immediate': len(response_data[response_data['response_time_minutes'] < 1]),
                            'fast': len(response_data[(response_data['response_time_minutes'] >= 1) & (response_data['response_time_minutes'] < 30)]),
                            'moderate': len(response_data[(response_data['response_time_minutes'] >= 30) & (response_data['response_time_minutes'] < 240)]),
                            'slow': len(response_data[response_data['response_time_minutes'] >= 240])
                        }
                    }
            
            # Conversation initiation patterns
            interactions['conversation_starters'] = self._analyze_conversation_starters(df)
            
            # User interaction frequency
            interactions['user_interactions'] = self._calculate_user_interactions(df)
            
            # Conversation sustainability
            interactions['conversation_flow'] = self._analyze_conversation_flow(df)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Interaction analysis failed: {e}")
            return {}
    
    def _generate_insights(self, df: pd.DataFrame, overview: Dict, user_analysis: Dict, temporal: Dict) -> List[str]:
        """Generate actionable insights from analysis"""
        insights = []
        
        try:
            # Activity insights
            if overview.get('activity_metrics'):
                avg_daily = overview['activity_metrics']['avg_daily_messages']
                if avg_daily > 100:
                    insights.append(f"Very active chat with {avg_daily:.1f} messages per day on average")
                elif avg_daily < 10:
                    insights.append(f"Low activity chat with only {avg_daily:.1f} messages per day on average")
                
                peak_hour = overview['activity_metrics']['peak_activity_hour']
                insights.append(f"Most active during {peak_hour}:00-{peak_hour+1}:00")
            
            # User behavior insights
            if user_analysis.get('user_rankings'):
                top_user = user_analysis['user_rankings'][0]
                if top_user['percentage'] > 50:
                    insights.append(f"Conversation dominated by {top_user['user']} ({top_user['percentage']:.1f}% of messages)")
                elif len(user_analysis['user_rankings']) > 3 and top_user['percentage'] < 30:
                    insights.append("Well-balanced participation among users")
            
            # Temporal insights
            if temporal.get('weekly_patterns'):
                weekend_pct = temporal['weekly_patterns']['weekend_vs_weekday']['weekend_percentage']
                if weekend_pct > 30:
                    insights.append(f"High weekend activity ({weekend_pct:.1f}% of messages)")
                else:
                    insights.append(f"Primarily weekday communication ({100-weekend_pct:.1f}% weekday messages)")
            
            # Response time insights
            if 'response_times' in df.columns:
                response_data = df[df['response_time_minutes'].notna()]
                if not response_data.empty:
                    avg_response = response_data['response_time_minutes'].mean()
                    if avg_response < 10:
                        insights.append(f"Very responsive group with {avg_response:.1f} minute average response time")
                    elif avg_response > 60:
                        insights.append(f"Slower response patterns with {avg_response:.1f} minute average")
            
            # Content insights
            if 'message_length' in df.columns:
                text_df = df[df['message_type'] == 'text']
                avg_length = text_df['message_length'].mean()
                if avg_length > 100:
                    insights.append(f"Detailed communication style with {avg_length:.0f} character average")
                elif avg_length < 30:
                    insights.append(f"Concise communication style with {avg_length:.0f} character average")
            
            # Engagement insights
            if 'emoji_count' in df.columns and df['emoji_count'].sum() > 0:
                emoji_rate = len(df[df['emoji_count'] > 0]) / len(df) * 100
                if emoji_rate > 30:
                    insights.append(f"High emoji usage ({emoji_rate:.1f}% of messages contain emojis)")
            
            # Sentiment insights (if available)
            if 'sentiment_label' in df.columns:
                sentiment_data = df[df['sentiment_label'].notna()]
                if not sentiment_data.empty:
                    positive_pct = len(sentiment_data[sentiment_data['sentiment_label'] == 'positive']) / len(sentiment_data) * 100
                    if positive_pct > 60:
                        insights.append(f"Very positive communication tone ({positive_pct:.1f}% positive sentiment)")
                    elif positive_pct < 40:
                        insights.append(f"Mixed communication tone with {positive_pct:.1f}% positive sentiment")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("Unable to generate detailed insights due to processing limitations")
        
        return insights
    
    def _generate_recommendations(self, df: pd.DataFrame, insights: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Activity-based recommendations
            daily_activity = df.groupby(df['date'].dt.date).size()
            activity_variance = daily_activity.var()
            
            if activity_variance > daily_activity.mean() * 2:
                recommendations.append("Consider establishing more consistent communication patterns")
            
            # User participation recommendations
            if 'user' in df.columns:
                user_counts = df[df['user'] != 'system']['user'].value_counts()
                if len(user_counts) > 1:
                    participation_balance = user_counts.std() / user_counts.mean()
                    if participation_balance > 1:
                        recommendations.append("Encourage more balanced participation from all group members")
            
            # Response time recommendations
            if 'response_time_minutes' in df.columns:
                response_data = df[df['response_time_minutes'].notna()]
                if not response_data.empty:
                    avg_response = response_data['response_time_minutes'].mean()
                    if avg_response > 120:  # 2 hours
                        recommendations.append("Consider improving response times for better engagement")
            
            # Content recommendations
            if 'message_length' in df.columns:
                text_df = df[df['message_type'] == 'text']
                if not text_df.empty:
                    short_messages = len(text_df[text_df['message_length'] < 10])
                    if short_messages / len(text_df) > 0.5:
                        recommendations.append("Consider more detailed communication to improve clarity")
            
            # Engagement recommendations
            if 'emoji_count' in df.columns:
                emoji_usage = len(df[df['emoji_count'] > 0]) / len(df) * 100
                if emoji_usage < 10:
                    recommendations.append("Consider using more emojis and reactions for better engagement")
            
            # Time-based recommendations
            if 'hour' in df.columns:
                business_hours_msgs = len(df[(df['hour'] >= 9) & (df['hour'] <= 17)])
                after_hours_msgs = len(df) - business_hours_msgs
                
                if after_hours_msgs > business_hours_msgs * 1.5:
                    recommendations.append("Consider setting communication boundaries for work-life balance")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Review communication patterns for optimization opportunities")
        
        return recommendations
    
    # Helper methods for specific analysis tasks
    
    def _calculate_interaction_matrix(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Calculate user interaction matrix"""
        try:
            users = df['user'].unique()
            matrix = {user: {other_user: 0 for other_user in users} for user in users}
            
            # Count interactions (responses between users)
            df_sorted = df.sort_values('date')
            
            for i in range(1, len(df_sorted)):
                current_user = df_sorted.iloc[i]['user']
                previous_user = df_sorted.iloc[i-1]['user']
                
                if current_user != previous_user and current_user != 'system' and previous_user != 'system':
                    matrix[previous_user][current_user] += 1
            
            return matrix
            
        except Exception as e:
            logger.error(f"Error calculating interaction matrix: {e}")
            return {}
    
    def _analyze_user_activity_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual user activity patterns"""
        try:
            patterns = {}
            
            for user in df['user'].unique():
                user_df = df[df['user'] == user]
                
                patterns[user] = {
                    'most_active_hour': user_df.groupby('hour').size().idxmax(),
                    'most_active_day': user_df.groupby('day_name').size().idxmax(),
                    'activity_consistency': self._calculate_consistency_score(user_df),
                    'burst_patterns': self._detect_burst_patterns(user_df)
                }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing user activity patterns: {e}")
            return {}
    
    def _detect_activity_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect unusual activity patterns"""
        try:
            daily_activity = df.groupby(df['date'].dt.date).size()
            
            # Statistical thresholds
            mean_activity = daily_activity.mean()
            std_activity = daily_activity.std()
            
            # Detect high activity days
            high_threshold = mean_activity + (2 * std_activity)
            high_activity_days = daily_activity[daily_activity > high_threshold]
            
            # Detect low activity days
            low_threshold = max(0, mean_activity - (2 * std_activity))
            low_activity_days = daily_activity[daily_activity < low_threshold]
            
            return {
                'high_activity_days': {str(date): count for date, count in high_activity_days.items()},
                'low_activity_days': {str(date): count for date, count in low_activity_days.items()},
                'activity_variance': daily_activity.var(),
                'consistency_score': 1 - (std_activity / mean_activity) if mean_activity > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error detecting activity anomalies: {e}")
            return {}
    
    def _analyze_content_timing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze when different types of content are shared"""
        try:
            timing = {}
            
            # Media sharing timing
            if 'message_type' in df.columns:
                media_df = df[df['message_type'] == 'media']
                if not media_df.empty:
                    timing['media_sharing'] = {
                        'peak_hour': media_df.groupby('hour').size().idxmax(),
                        'peak_day': media_df.groupby('day_name').size().idxmax(),
                        'weekend_percentage': len(media_df[media_df['is_weekend'] == True]) / len(media_df) * 100 if 'is_weekend' in media_df.columns else 0
                    }
            
            # URL sharing timing
            if 'url_count' in df.columns:
                url_df = df[df['url_count'] > 0]
                if not url_df.empty:
                    timing['url_sharing'] = {
                        'peak_hour': url_df.groupby('hour').size().idxmax(),
                        'peak_day': url_df.groupby('day_name').size().idxmax()
                    }
            
            return timing
            
        except Exception as e:
            logger.error(f"Error analyzing content timing: {e}")
            return {}
    
    def _calculate_vocabulary_richness(self, df: pd.DataFrame) -> float:
        """Calculate vocabulary richness score"""
        try:
            all_text = ' '.join(df['message'].astype(str))
            words = re.findall(r'\b\w+\b', all_text.lower())
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.error(f"Error calculating vocabulary richness: {e}")
            return 0.0
    
    def _analyze_emoji_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze emoji usage patterns"""
        try:
            emoji_df = df[df['emoji_count'] > 0]
            
            if emoji_df.empty:
                return {}
            
            return {
                'total_emojis': df['emoji_count'].sum(),
                'messages_with_emojis': len(emoji_df),
                'emoji_usage_rate': len(emoji_df) / len(df) * 100,
                'top_emoji_users': emoji_df.groupby('user')['emoji_count'].sum().nlargest(3).to_dict(),
                'emoji_timing': {
                    'peak_hour': emoji_df.groupby('hour').size().idxmax(),
                    'peak_day': emoji_df.groupby('day_name').size().idxmax()
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing emoji patterns: {e}")
            return {}
    
    def _analyze_communication_style(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall communication style"""
        try:
            style = {}
            
            # Formality indicators
            if 'uppercase_ratio' in df.columns:
                high_caps = df[df['uppercase_ratio'] > 0.5]
                style['formality'] = {
                    'casual_indicators': len(df[df['uppercase_ratio'] < 0.1]),
                    'shouting_messages': len(high_caps),
                    'avg_capitalization': df['uppercase_ratio'].mean()
                }
            
            # Question asking patterns
            if 'is_question' in df.columns:
                questions = df[df['is_question'] == True]
                style['inquisitiveness'] = {
                    'total_questions': len(questions),
                    'question_rate': len(questions) / len(df) * 100,
                    'most_inquisitive_hour': questions.groupby('hour').size().idxmax() if not questions.empty else None
                }
            
            return style
            
        except Exception as e:
            logger.error(f"Error analyzing communication style: {e}")
            return {}
    
    def _analyze_conversation_starters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze who initiates conversations"""
        try:
            # Define conversation gaps (> 2 hours)
            df_sorted = df.sort_values('date')
            conversation_starts = []
            
            for i in range(len(df_sorted)):
                if i == 0:
                    conversation_starts.append(df_sorted.iloc[i]['user'])
                else:
                    time_gap = (df_sorted.iloc[i]['date'] - df_sorted.iloc[i-1]['date']).total_seconds() / 3600
                    if time_gap > 2:  # 2 hour gap
                        conversation_starts.append(df_sorted.iloc[i]['user'])
            
            starter_counts = Counter(conversation_starts)
            
            return {
                'conversation_starters': dict(starter_counts),
                'most_frequent_starter': starter_counts.most_common(1)[0][0] if starter_counts else None,
                'total_conversation_starts': len(conversation_starts)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conversation starters: {e}")
            return {}
    
    def _calculate_user_interactions(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calculate direct user interactions"""
        try:
            interactions = defaultdict(int)
            df_sorted = df.sort_values('date')
            
            for i in range(1, len(df_sorted)):
                current_user = df_sorted.iloc[i]['user']
                prev_user = df_sorted.iloc[i-1]['user']
                
                if current_user != prev_user and current_user != 'system' and prev_user != 'system':
                    pair = tuple(sorted([current_user, prev_user]))
                    interactions[pair] += 1
            
            return dict(interactions)
            
        except Exception as e:
            logger.error(f"Error calculating user interactions: {e}")
            return {}
    
    def _analyze_conversation_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze conversation flow and sustainability"""
        try:
            df_sorted = df.sort_values('date')
            
            # Calculate conversation lengths
            conversation_lengths = []
            current_length = 1
            
            for i in range(1, len(df_sorted)):
                time_gap = (df_sorted.iloc[i]['date'] - df_sorted.iloc[i-1]['date']).total_seconds() / 3600
                
                if time_gap < 2:  # Same conversation
                    current_length += 1
                else:  # New conversation
                    conversation_lengths.append(current_length)
                    current_length = 1
            
            conversation_lengths.append(current_length)
            
            return {
                'avg_conversation_length': np.mean(conversation_lengths) if conversation_lengths else 0,
                'longest_conversation': max(conversation_lengths) if conversation_lengths else 0,
                'total_conversations': len(conversation_lengths),
                'short_conversations': len([c for c in conversation_lengths if c <= 3]),
                'sustained_conversations': len([c for c in conversation_lengths if c > 10])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conversation flow: {e}")
            return {}
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        try:
            if len(series) < 2:
                return "insufficient_data"
            
            # Simple linear trend
            x = np.arange(len(series))
            y = series.values
            
            # Calculate correlation
            correlation = np.corrcoef(x, y)[0, 1]
            
            if correlation > 0.3:
                return "increasing"
            elif correlation < -0.3:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return "unknown"
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate activity consistency score for a user"""
        try:
            daily_activity = df.groupby(df['date'].dt.date).size()
            
            if len(daily_activity) < 2:
                return 0.0
            
            mean_activity = daily_activity.mean()
            std_activity = daily_activity.std()
            
            # Consistency score: higher is more consistent
            consistency = 1 - (std_activity / mean_activity) if mean_activity > 0 else 0
            return max(0, min(1, consistency))
            
        except Exception as e:
            logger.error(f"Error calculating consistency score: {e}")
            return 0.0
    
    def _detect_burst_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect message burst patterns"""
        try:
            # Group messages by 5-minute intervals
            df['time_group'] = df['date'].dt.floor('5min')
            burst_activity = df.groupby('time_group').size()
            
            # Define bursts as > 5 messages in 5 minutes
            bursts = burst_activity[burst_activity > 5]
            
            return {
                'total_bursts': len(bursts),
                'avg_burst_intensity': bursts.mean() if not bursts.empty else 0,
                'max_burst_intensity': bursts.max() if not bursts.empty else 0,
                'burst_frequency': len(bursts) / len(burst_activity) * 100 if len(burst_activity) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error detecting burst patterns: {e}")
            return {}
    
    def _empty_result(self) -> AnalysisResult:
        """Return empty analysis result"""
        return AnalysisResult(
            overview={},
            user_analysis={},
            temporal_analysis={},
            content_analysis={},
            interaction_analysis={},
            insights=["No data available for analysis"],
            recommendations=["Please upload a valid WhatsApp chat file"]
        )
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of last analysis"""
        if not self.analysis_results:
            return {"error": "No analysis available"}
        
        return {
            "total_insights": len(self.analysis_results.insights),
            "total_recommendations": len(self.analysis_results.recommendations),
            "analysis_sections": [
                "overview", "user_analysis", "temporal_analysis", 
                "content_analysis", "interaction_analysis"
            ],
            "key_metrics": {
                "total_messages": self.analysis_results.overview.get('total_messages', 0),
                "unique_users": self.analysis_results.overview.get('unique_users', 0),
                "avg_daily_messages": self.analysis_results.overview.get('activity_metrics', {}).get('avg_daily_messages', 0)
            }
        }