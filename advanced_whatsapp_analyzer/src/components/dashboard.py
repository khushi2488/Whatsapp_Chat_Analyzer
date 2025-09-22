"""
Dashboard Components Module
Contains reusable dashboard components for the chat analyzer.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DashboardComponents:
    """Reusable dashboard components"""
    
    def __init__(self):
        self.colors = {
            'primary': '#4CAF50',
            'secondary': '#2196F3', 
            'success': '#22c55e',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'info': '#3b82f6'
        }
    
    def metric_card(self, title: str, value: str, delta: Optional[str] = None, 
                   color: str = 'primary', help_text: Optional[str] = None):
        """Create a styled metric card"""
        st.metric(
            label=title,
            value=value,
            delta=delta,
            help=help_text
        )
    
    def create_kpi_row(self, metrics: List[Dict[str, Any]]):
        """Create a row of KPI metrics"""
        cols = st.columns(len(metrics))
        
        for i, metric in enumerate(metrics):
            with cols[i]:
                self.metric_card(**metric)
    
    def activity_timeline_chart(self, df: pd.DataFrame, date_column: str = 'date', 
                               title: str = "Activity Timeline") -> go.Figure:
        """Create an activity timeline chart"""
        try:
            # Group by date
            daily_counts = df.groupby(df[date_column].dt.date).size()
            
            # Create line chart
            fig = px.line(
                x=daily_counts.index,
                y=daily_counts.values,
                title=title,
                labels={'x': 'Date', 'y': 'Messages'}
            )
            
            # Add moving average
            if len(daily_counts) > 7:
                ma_7 = daily_counts.rolling(window=7).mean()
                fig.add_trace(go.Scatter(
                    x=ma_7.index,
                    y=ma_7.values,
                    mode='lines',
                    name='7-day Average',
                    line=dict(dash='dash', color='red', width=2)
                ))
            
            fig.update_layout(
                height=400,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating timeline chart: {e}")
            return go.Figure()
    
    def hourly_heatmap(self, df: pd.DataFrame, title: str = "Hourly Activity Heatmap") -> go.Figure:
        """Create hourly activity heatmap"""
        try:
            # Create pivot table for heatmap
            df['hour'] = df['date'].dt.hour
            df['day_name'] = df['date'].dt.day_name()
            
            heatmap_data = df.pivot_table(
                index='day_name',
                columns='hour',
                aggfunc='size',
                fill_value=0
            )
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex([day for day in day_order if day in heatmap_data.index])
            
            fig = px.imshow(
                heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                aspect="auto",
                color_continuous_scale="Viridis",
                title=title
            )
            
            fig.update_xaxis(title="Hour of Day")
            fig.update_yaxis(title="Day of Week")
            fig.update_layout(height=400)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return go.Figure()
    
    def sentiment_gauge(self, positive_pct: float, title: str = "Sentiment Score") -> go.Figure:
        """Create a sentiment gauge chart"""
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=positive_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=400)
            return fig
            
        except Exception as e:
            logger.error(f"Error creating gauge: {e}")
            return go.Figure()
    
    def user_activity_comparison(self, df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """Create user activity comparison chart"""
        try:
            user_stats = df[df['user'] != 'system']['user'].value_counts().head(top_n)
            
            fig = px.bar(
                x=user_stats.values,
                y=user_stats.index,
                orientation='h',
                title=f"Top {top_n} Most Active Users",
                labels={'x': 'Messages', 'y': 'User'}
            )
            
            fig.update_layout(height=400)
            return fig
            
        except Exception as e:
            logger.error(f"Error creating user comparison: {e}")
            return go.Figure()
    
    def response_time_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create response time distribution chart"""
        try:
            if 'response_time_minutes' not in df.columns:
                return go.Figure()
            
            # Filter reasonable response times (< 24 hours)
            response_data = df[
                (df['response_time_minutes'].notna()) & 
                (df['response_time_minutes'] < 1440)
            ]['response_time_minutes']
            
            if response_data.empty:
                return go.Figure()
            
            fig = px.histogram(
                response_data,
                nbins=50,
                title="Response Time Distribution",
                labels={'x': 'Response Time (minutes)', 'y': 'Frequency'}
            )
            
            # Add median line
            median_time = response_data.median()
            fig.add_vline(
                x=median_time,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Median: {median_time:.1f} min"
            )
            
            fig.update_layout(height=400)
            return fig
            
        except Exception as e:
            logger.error(f"Error creating response time chart: {e}")
            return go.Figure()
    
    def create_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        try:
            stats = {}
            
            # Basic counts
            stats['total_messages'] = len(df)
            stats['text_messages'] = len(df[df['message_type'] == 'text'])
            stats['media_messages'] = len(df[df['message_type'] == 'media'])
            stats['system_messages'] = len(df[df['message_type'] == 'system'])
            
            # Date range
            stats['start_date'] = df['date'].min().date()
            stats['end_date'] = df['date'].max().date()
            stats['duration_days'] = (df['date'].max() - df['date'].min()).days + 1
            
            # User statistics
            users = df[df['user'] != 'system']['user'].nunique()
            stats['unique_users'] = users
            stats['is_group_chat'] = users > 2
            
            # Activity patterns
            stats['avg_daily_messages'] = stats['total_messages'] / stats['duration_days'] if stats['duration_days'] > 0 else 0
            stats['peak_hour'] = df['hour'].mode().iloc[0] if len(df) > 0 else 0
            stats['peak_day'] = df['day_name'].mode().iloc[0] if len(df) > 0 else 'Unknown'
            
            # Content statistics
            if 'message_length' in df.columns:
                text_df = df[df['message_type'] == 'text']
                stats['avg_message_length'] = text_df['message_length'].mean() if len(text_df) > 0 else 0
                stats['total_characters'] = text_df['message_length'].sum()
            
            if 'word_count' in df.columns:
                stats['total_words'] = df['word_count'].sum()
                stats['avg_words_per_message'] = df['word_count'].mean()
            
            # Response time statistics
            if 'response_time_minutes' in df.columns:
                response_data = df[df['response_time_minutes'].notna() & (df['response_time_minutes'] < 1440)]
                if len(response_data) > 0:
                    stats['avg_response_time'] = response_data['response_time_minutes'].mean()
                    stats['median_response_time'] = response_data['response_time_minutes'].median()
                    stats['fast_responses_pct'] = len(response_data[response_data['response_time_minutes'] < 5]) / len(response_data) * 100
            
            # Sentiment statistics (if available)
            if 'sentiment_label' in df.columns:
                sentiment_data = df[df['sentiment_label'].notna()]
                if len(sentiment_data) > 0:
                    sentiment_counts = sentiment_data['sentiment_label'].value_counts()
                    total_sentiment = len(sentiment_data)
                    stats['positive_pct'] = sentiment_counts.get('positive', 0) / total_sentiment * 100
                    stats['negative_pct'] = sentiment_counts.get('negative', 0) / total_sentiment * 100
                    stats['neutral_pct'] = sentiment_counts.get('neutral', 0) / total_sentiment * 100
                    
                    if 'compound_score' in df.columns:
                        stats['avg_sentiment_score'] = sentiment_data['compound_score'].mean()
            
            # Media and link statistics
            if 'url_count' in df.columns:
                stats['total_links'] = df['url_count'].sum()
                stats['messages_with_links'] = len(df[df['url_count'] > 0])
            
            if 'emoji_count' in df.columns:
                stats['total_emojis'] = df['emoji_count'].sum()
                stats['messages_with_emojis'] = len(df[df['emoji_count'] > 0])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating summary stats: {e}")
            return {}
    
    def display_summary_dashboard(self, df: pd.DataFrame):
        """Display a comprehensive summary dashboard"""
        try:
            stats = self.create_summary_stats(df)
            
            if not stats:
                st.error("Unable to generate summary statistics")
                return
            
            # Header metrics
            st.markdown("### 📊 Chat Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Messages", f"{stats['total_messages']:,}")
            
            with col2:
                st.metric("Duration", f"{stats['duration_days']} days")
            
            with col3:
                st.metric("Participants", stats['unique_users'])
            
            with col4:
                st.metric("Daily Average", f"{stats['avg_daily_messages']:.1f}")
            
            # Activity charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Timeline chart
                timeline_fig = self.activity_timeline_chart(df)
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            with col2:
                # User activity chart
                if stats['unique_users'] > 1:
                    user_fig = self.user_activity_comparison(df)
                    st.plotly_chart(user_fig, use_container_width=True)
                else:
                    # Show hourly distribution for individual chats
                    hourly_data = df.groupby('hour').size()
                    fig = px.bar(x=hourly_data.index, y=hourly_data.values,
                               title="Hourly Activity Distribution")
                    fig.update_xaxis(title="Hour")
                    fig.update_yaxis(title="Messages")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap
            st.markdown("### 🗓️ Activity Heatmap")
            heatmap_fig = self.hourly_heatmap(df)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Additional metrics
            if 'avg_response_time' in stats:
                st.markdown("### ⚡ Response Time Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Response", f"{stats['avg_response_time']:.1f} min")
                
                with col2:
                    st.metric("Median Response", f"{stats['median_response_time']:.1f} min")
                
                with col3:
                    st.metric("Fast Responses", f"{stats['fast_responses_pct']:.1f}%")
                
                # Response time distribution
                response_fig = self.response_time_distribution(df)
                st.plotly_chart(response_fig, use_container_width=True)
            
            # Sentiment overview (if available)
            if 'positive_pct' in stats:
                st.markdown("### 😊 Sentiment Overview")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment gauge
                    gauge_fig = self.sentiment_gauge(stats['positive_pct'])
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                with col2:
                    # Sentiment breakdown
                    sentiment_data = {
                        'Sentiment': ['Positive', 'Negative', 'Neutral'],
                        'Percentage': [stats['positive_pct'], stats['negative_pct'], stats['neutral_pct']]
                    }
                    
                    fig = px.pie(
                        values=sentiment_data['Percentage'],
                        names=sentiment_data['Sentiment'],
                        title="Sentiment Distribution",
                        color_discrete_map={
                            'Positive': '#22c55e',
                            'Negative': '#ef4444',
                            'Neutral': '#6b7280'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error displaying dashboard: {e}")
            st.error(f"Dashboard error: {e}")
    
    def create_comparison_chart(self, data: Dict[str, float], title: str, 
                              chart_type: str = 'bar') -> go.Figure:
        """Create a comparison chart"""
        try:
            if chart_type == 'bar':
                fig = px.bar(
                    x=list(data.keys()),
                    y=list(data.values()),
                    title=title
                )
            elif chart_type == 'pie':
                fig = px.pie(
                    values=list(data.values()),
                    names=list(data.keys()),
                    title=title
                )
            else:
                fig = go.Figure()
            
            fig.update_layout(height=400)
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return go.Figure()
    
    def show_data_quality_report(self, df: pd.DataFrame):
        """Display data quality report"""
        try:
            st.markdown("### 🔍 Data Quality Report")
            
            # Basic data info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(df))
            
            with col2:
                missing_data = df.isnull().sum().sum()
                st.metric("Missing Values", missing_data)
            
            with col3:
                data_types = len(df.dtypes.unique())
                st.metric("Data Types", data_types)
            
            # Missing data analysis
            if missing_data > 0:
                missing_by_column = df.isnull().sum()
                missing_by_column = missing_by_column[missing_by_column > 0]
                
                if len(missing_by_column) > 0:
                    st.markdown("#### Missing Data by Column")
                    fig = px.bar(
                        x=missing_by_column.index,
                        y=missing_by_column.values,
                        title="Missing Values by Column"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Data distribution
            st.markdown("#### Message Type Distribution")
            msg_type_dist = df['message_type'].value_counts()
            fig = px.pie(
                values=msg_type_dist.values,
                names=msg_type_dist.index,
                title="Message Types"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error showing data quality report: {e}")
            st.error("Unable to generate data quality report")

# Utility functions for dashboard components
def format_number(num: float, decimal_places: int = 1) -> str:
    """Format numbers for display"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.{decimal_places}f}M"
    elif num >= 1_000:
        return f"{num/1_000:.{decimal_places}f}K"
    else:
        return f"{num:.{decimal_places}f}"

def calculate_trend(current: float, previous: float) -> Tuple[str, str]:
    """Calculate trend direction and percentage"""
    if previous == 0:
        return "→", "0%"
    
    change = ((current - previous) / previous) * 100
    
    if change > 0:
        return "↗", f"+{change:.1f}%"
    elif change < 0:
        return "↘", f"{change:.1f}%"
    else:
        return "→", "0%"