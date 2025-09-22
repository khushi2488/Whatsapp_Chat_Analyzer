"""
Advanced Visualizations Module
Contains sophisticated visualization components for the chat analyzer.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class AdvancedVisualizations:
    """Advanced visualization components"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17becf',
            'gradient': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        }
        
        self.sentiment_colors = {
            'positive': '#22c55e',
            'negative': '#ef4444',
            'neutral': '#6b7280'
        }
    
    def create_sentiment_timeline(self, df: pd.DataFrame, user: str = 'Overall') -> go.Figure:
        """Create advanced sentiment timeline with multiple metrics"""
        try:
            if 'sentiment_label' not in df.columns:
                return go.Figure()
            
            # Filter data
            if user != 'Overall':
                df = df[df['user'] == user]
            
            sentiment_df = df[df['sentiment_label'].notna()].copy()
            if sentiment_df.empty:
                return go.Figure()
            
            # Group by date and sentiment
            daily_sentiment = sentiment_df.groupby([
                sentiment_df['date'].dt.date, 
                'sentiment_label'
            ]).size().unstack(fill_value=0)
            
            # Calculate percentages
            daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Sentiment Counts', 'Sentiment Percentages'],
                vertical_spacing=0.15,
                shared_xaxes=True
            )
            
            # Add count traces
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in daily_sentiment.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=daily_sentiment.index,
                            y=daily_sentiment[sentiment],
                            mode='lines+markers',
                            name=f'{sentiment.title()} Count',
                            line=dict(color=self.sentiment_colors[sentiment]),
                            showlegend=True
                        ),
                        row=1, col=1
                    )
            
            # Add percentage traces
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in daily_sentiment_pct.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=daily_sentiment_pct.index,
                            y=daily_sentiment_pct[sentiment],
                            mode='lines',
                            name=f'{sentiment.title()} %',
                            line=dict(color=self.sentiment_colors[sentiment], dash='dot'),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
            
            fig.update_layout(
                title=f'Sentiment Analysis Timeline - {user}',
                height=600,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Message Count", row=1, col=1)
            fig.update_yaxes(title_text="Percentage", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sentiment timeline: {e}")
            return go.Figure()
    
    def create_user_interaction_network(self, df: pd.DataFrame) -> go.Figure:
        """Create user interaction network visualization"""
        try:
            # Filter text messages with response times
            interaction_df = df[
                (df['message_type'] == 'text') & 
                (df['response_time_minutes'].notna()) &
                (df['user'] != 'system')
            ].copy()
            
            if len(interaction_df) < 2:
                return go.Figure()
            
            # Create interaction pairs
            interactions = []
            for i in range(1, len(interaction_df)):
                current_user = interaction_df.iloc[i]['user']
                previous_user = interaction_df.iloc[i-1]['user']
                
                if current_user != previous_user:
                    interactions.append({
                        'from': previous_user,
                        'to': current_user,
                        'response_time': interaction_df.iloc[i]['response_time_minutes']
                    })
            
            if not interactions:
                return go.Figure()
            
            interaction_df = pd.DataFrame(interactions)
            
            # Count interactions and average response times
            interaction_stats = interaction_df.groupby(['from', 'to']).agg({
                'response_time': ['count', 'mean']
            }).round(2)
            
            interaction_stats.columns = ['count', 'avg_response_time']
            interaction_stats = interaction_stats.reset_index()
            
            # Get unique users
            users = list(set(interaction_stats['from'].tolist() + interaction_stats['to'].tolist()))
            
            # Create network visualization
            fig = go.Figure()
            
            # Add nodes (users)
            user_positions = {}
            n_users = len(users)
            for i, user in enumerate(users):
                angle = 2 * np.pi * i / n_users
                x = np.cos(angle)
                y = np.sin(angle)
                user_positions[user] = (x, y)
                
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=20, color='lightblue'),
                    text=user,
                    textposition='middle center',
                    name=user,
                    showlegend=False
                ))
            
            # Add edges (interactions)
            for _, row in interaction_stats.iterrows():
                from_pos = user_positions[row['from']]
                to_pos = user_positions[row['to']]
                
                # Line width based on interaction count
                line_width = min(row['count'] / 5, 10)
                
                fig.add_trace(go.Scatter(
                    x=[from_pos[0], to_pos[0], None],
                    y=[from_pos[1], to_pos[1], None],
                    mode='lines',
                    line=dict(width=line_width, color='gray'),
                    hovertemplate=f"From: {row['from']}<br>To: {row['to']}<br>Interactions: {row['count']}<br>Avg Response: {row['avg_response_time']:.1f}min<extra></extra>",
                    showlegend=False
                ))
            
            fig.update_layout(
                title="User Interaction Network",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interaction network: {e}")
            return go.Figure()
    
    def create_activity_flow_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create activity flow visualization showing conversation patterns"""
        try:
            # Create hourly activity data
            df['hour'] = df['date'].dt.hour
            df['day_name'] = df['date'].dt.day_name()
            
            # Get activity flow (hour to hour transitions)
            flow_data = []
            sorted_df = df.sort_values('date')
            
            for i in range(1, len(sorted_df)):
                prev_hour = sorted_df.iloc[i-1]['hour']
                curr_hour = sorted_df.iloc[i]['hour']
                time_diff = (sorted_df.iloc[i]['date'] - sorted_df.iloc[i-1]['date']).total_seconds() / 3600
                
                # Only consider transitions within 3 hours
                if time_diff <= 3:
                    flow_data.append({
                        'from_hour': prev_hour,
                        'to_hour': curr_hour,
                        'count': 1
                    })
            
            if not flow_data:
                return go.Figure()
            
            flow_df = pd.DataFrame(flow_data)
            flow_summary = flow_df.groupby(['from_hour', 'to_hour']).count().reset_index()
            
            # Create Sankey diagram
            hours = list(range(24))
            hour_labels = [f"{h:02d}:00" for h in hours]
            
            # Prepare data for Sankey
            sources = []
            targets = []
            values = []
            
            for _, row in flow_summary.iterrows():
                sources.append(row['from_hour'])
                targets.append(row['to_hour'])
                values.append(row['count'])
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=hour_labels,
                    color="blue"
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(255, 0, 255, 0.4)"
                )
            )])
            
            fig.update_layout(
                title_text="Conversation Flow by Hour",
                font_size=10,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating activity flow chart: {e}")
            return go.Figure()
    
    def create_emotion_radar_chart(self, df: pd.DataFrame, user: str = 'Overall') -> go.Figure:
        """Create radar chart for emotion analysis"""
        try:
            if 'emotion' not in df.columns:
                return go.Figure()
            
            # Filter data
            if user != 'Overall':
                df = df[df['user'] == user]
            
            emotion_df = df[df['emotion'].notna()].copy()
            if emotion_df.empty:
                return go.Figure()
            
            # Get emotion counts
            emotion_counts = emotion_df['emotion'].value_counts()
            
            # Ensure we have common emotions
            common_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'love', 'neutral']
            emotion_data = []
            
            for emotion in common_emotions:
                count = emotion_counts.get(emotion, 0)
                emotion_data.append(count)
            
            # Normalize to percentages
            total = sum(emotion_data) if sum(emotion_data) > 0 else 1
            emotion_percentages = [count / total * 100 for count in emotion_data]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=emotion_percentages,
                theta=common_emotions,
                fill='toself',
                name=f'{user} Emotions',
                line=dict(color='rgb(1,90,200)')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(emotion_percentages) * 1.1] if emotion_percentages else [0, 100]
                    )),
                showlegend=False,
                title=f"Emotion Profile - {user}",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating emotion radar chart: {e}")
            return go.Figure()
    
    def create_response_time_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create heatmap showing response times by hour and day"""
        try:
            if 'response_time_minutes' not in df.columns:
                return go.Figure()
            
            # Filter reasonable response times
            response_df = df[
                (df['response_time_minutes'].notna()) & 
                (df['response_time_minutes'] < 480)  # Less than 8 hours
            ].copy()
            
            if response_df.empty:
                return go.Figure()
            
            response_df['hour'] = response_df['date'].dt.hour
            response_df['day_name'] = response_df['date'].dt.day_name()
            
            # Create pivot table
            heatmap_data = response_df.pivot_table(
                index='day_name',
                columns='hour',
                values='response_time_minutes',
                aggfunc='mean'
            )
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex([day for day in day_order if day in heatmap_data.index])
            
            fig = px.imshow(
                heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                aspect="auto",
                color_continuous_scale="RdYlBu_r",
                title="Average Response Time Heatmap (minutes)",
                labels=dict(color="Response Time (min)")
            )
            
            fig.update_xaxis(title="Hour of Day")
            fig.update_yaxis(title="Day of Week")
            fig.update_layout(height=400)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating response time heatmap: {e}")
            return go.Figure()
    
    def create_message_length_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create message length distribution with statistical overlay"""
        try:
            if 'message_length' not in df.columns:
                return go.Figure()
            
            text_df = df[df['message_type'] == 'text'].copy()
            if text_df.empty:
                return go.Figure()
            
            lengths = text_df['message_length']
            
            # Create histogram
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=lengths,
                nbinsx=50,
                name='Message Length',
                opacity=0.7,
                marker_color='lightblue'
            ))
            
            # Add statistical lines
            mean_length = lengths.mean()
            median_length = lengths.median()
            
            fig.add_vline(
                x=mean_length,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_length:.1f}"
            )
            
            fig.add_vline(
                x=median_length,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Median: {median_length:.1f}"
            )
            
            # Add percentiles
            p25 = lengths.quantile(0.25)
            p75 = lengths.quantile(0.75)
            
            fig.add_vline(
                x=p25,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"25th: {p25:.1f}"
            )
            
            fig.add_vline(
                x=p75,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"75th: {p75:.1f}"
            )
            
            fig.update_layout(
                title="Message Length Distribution",
                xaxis_title="Message Length (characters)",
                yaxis_title="Frequency",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating message length distribution: {e}")
            return go.Figure()
    
    def create_user_sentiment_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Create comparative sentiment analysis across users"""
        try:
            if 'sentiment_label' not in df.columns:
                return go.Figure()
            
            # Filter data
            sentiment_df = df[
                (df['sentiment_label'].notna()) & 
                (df['user'] != 'system')
            ].copy()
            
            if sentiment_df.empty:
                return go.Figure()
            
            # Get user sentiment distribution
            user_sentiment = sentiment_df.groupby(['user', 'sentiment_label']).size().unstack(fill_value=0)
            
            # Calculate percentages
            user_sentiment_pct = user_sentiment.div(user_sentiment.sum(axis=1), axis=0) * 100
            
            # Create stacked bar chart
            fig = go.Figure()
            
            sentiments = ['positive', 'neutral', 'negative']
            colors = [self.sentiment_colors.get(s, '#666666') for s in sentiments]
            
            for i, sentiment in enumerate(sentiments):
                if sentiment in user_sentiment_pct.columns:
                    fig.add_trace(go.Bar(
                        name=sentiment.title(),
                        x=user_sentiment_pct.index,
                        y=user_sentiment_pct[sentiment],
                        marker_color=colors[i]
                    ))
            
            fig.update_layout(
                barmode='stack',
                title='Sentiment Distribution by User',
                xaxis_title='User',
                yaxis_title='Percentage',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating user sentiment comparison: {e}")
            return go.Figure()
    
    def create_conversation_momentum(self, df: pd.DataFrame) -> go.Figure:
        """Create conversation momentum visualization"""
        try:
            # Calculate conversation momentum based on message frequency
            df_sorted = df.sort_values('date')
            
            # Calculate rolling message count (momentum)
            df_sorted['rolling_count'] = df_sorted.set_index('date').rolling('1H').size().values
            
            # Calculate time gaps between messages
            df_sorted['time_gap'] = df_sorted['date'].diff().dt.total_seconds() / 60  # in minutes
            df_sorted['momentum_score'] = np.where(
                df_sorted['time_gap'] < 5, 
                df_sorted['rolling_count'] * 2,  # High momentum for quick responses
                df_sorted['rolling_count']
            )
            
            # Group by hour for visualization
            hourly_momentum = df_sorted.groupby(df_sorted['date'].dt.floor('H')).agg({
                'momentum_score': 'mean',
                'message_id': 'count'
            })
            
            # Create dual axis chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add momentum score
            fig.add_trace(
                go.Scatter(
                    x=hourly_momentum.index,
                    y=hourly_momentum['momentum_score'],
                    mode='lines',
                    name='Momentum Score',
                    line=dict(color='red', width=2)
                ),
                secondary_y=False,
            )
            
            # Add message count
            fig.add_trace(
                go.Scatter(
                    x=hourly_momentum.index,
                    y=hourly_momentum['message_id'],
                    mode='lines',
                    name='Messages per Hour',
                    line=dict(color='blue', width=1),
                    opacity=0.6
                ),
                secondary_y=True,
            )
            
            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Momentum Score", secondary_y=False)
            fig.update_yaxes(title_text="Messages per Hour", secondary_y=True)
            
            fig.update_layout(
                title_text="Conversation Momentum Over Time",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating conversation momentum: {e}")
            return go.Figure()
    
    def create_topic_evolution_chart(self, df: pd.DataFrame, topics: Dict[str, List[str]] = None) -> go.Figure:
        """Create topic evolution over time (placeholder for topic modeling)"""
        try:
            if not topics:
                # Simple keyword-based topic detection as placeholder
                topics = {
                    'work': ['work', 'job', 'office', 'meeting', 'project'],
                    'social': ['party', 'fun', 'friend', 'hang', 'weekend'],
                    'family': ['family', 'mom', 'dad', 'home', 'kids'],
                    'tech': ['app', 'phone', 'computer', 'software', 'tech']
                }
            
            text_df = df[df['message_type'] == 'text'].copy()
            if text_df.empty:
                return go.Figure()
            
            # Simple topic assignment based on keywords
            topic_data = []
            
            for _, row in text_df.iterrows():
                message = str(row['message']).lower()
                date = row['date']
                
                for topic, keywords in topics.items():
                    if any(keyword in message for keyword in keywords):
                        topic_data.append({
                            'date': date,
                            'topic': topic,
                            'count': 1
                        })
            
            if not topic_data:
                return go.Figure()
            
            topic_df = pd.DataFrame(topic_data)
            
            # Group by date and topic
            daily_topics = topic_df.groupby([
                topic_df['date'].dt.date,
                'topic'
            ]).sum().reset_index()
            
            # Create line chart for each topic
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, topic in enumerate(topics.keys()):
                topic_data = daily_topics[daily_topics['topic'] == topic]
                if not topic_data.empty:
                    fig.add_trace(go.Scatter(
                        x=topic_data['date'],
                        y=topic_data['count'],
                        mode='lines+markers',
                        name=topic.title(),
                        line=dict(color=colors[i % len(colors)])
                    ))
            
            fig.update_layout(
                title="Topic Evolution Over Time",
                xaxis_title="Date",
                yaxis_title="Topic Mentions",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating topic evolution chart: {e}")
            return go.Figure()
    
    def create_communication_health_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """Create comprehensive communication health dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Daily Activity Trend',
                    'Response Time Distribution',
                    'Sentiment Balance',
                    'User Participation'
                ],
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ]
            )
            
            # 1. Daily Activity Trend
            daily_activity = df.groupby(df['date'].dt.date).size()
            ma_7 = daily_activity.rolling(window=7).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_activity.index,
                    y=daily_activity.values,
                    mode='lines',
                    name='Daily Messages',
                    opacity=0.3,
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=ma_7.index,
                    y=ma_7.values,
                    mode='lines',
                    name='7-day Average',
                    line=dict(color='red', width=2),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 2. Response Time Distribution
            if 'response_time_minutes' in df.columns:
                response_data = df[
                    (df['response_time_minutes'].notna()) & 
                    (df['response_time_minutes'] < 240)
                ]['response_time_minutes']
                
                if not response_data.empty:
                    fig.add_trace(
                        go.Histogram(
                            x=response_data,
                            nbinsx=30,
                            showlegend=False,
                            marker_color='lightblue'
                        ),
                        row=1, col=2
                    )
            
            # 3. Sentiment Balance
            if 'sentiment_label' in df.columns:
                sentiment_data = df[df['sentiment_label'].notna()]
                if not sentiment_data.empty:
                    sentiment_counts = sentiment_data['sentiment_label'].value_counts()
                    
                    fig.add_trace(
                        go.Pie(
                            values=sentiment_counts.values,
                            labels=sentiment_counts.index,
                            showlegend=False,
                            marker_colors=[
                                self.sentiment_colors.get(label, '#666666') 
                                for label in sentiment_counts.index
                            ]
                        ),
                        row=2, col=1
                    )
            
            # 4. User Participation
            user_activity = df[df['user'] != 'system']['user'].value_counts().head(5)
            
            fig.add_trace(
                go.Bar(
                    x=user_activity.index,
                    y=user_activity.values,
                    showlegend=False,
                    marker_color='lightgreen'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="Communication Health Dashboard",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating health dashboard: {e}")
            return go.Figure()
    
    def create_custom_metric_chart(self, data: Dict[str, float], title: str, 
                                 chart_type: str = 'gauge') -> go.Figure:
        """Create custom metric visualization"""
        try:
            if chart_type == 'gauge':
                value = list(data.values())[0] if data else 0
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=value,
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
                
            elif chart_type == 'bar':
                fig = px.bar(
                    x=list(data.keys()),
                    y=list(data.values()),
                    title=title
                )
                
            else:  # default to line chart
                fig = px.line(
                    x=list(data.keys()),
                    y=list(data.values()),
                    title=title
                )
            
            fig.update_layout(height=400)
            return fig
            
        except Exception as e:
            logger.error(f"Error creating custom metric chart: {e}")
            return go.Figure()

# Utility functions for visualizations
def prepare_time_series_data(df: pd.DataFrame, date_column: str, 
                           value_column: str, freq: str = 'D') -> pd.DataFrame:
    """Prepare time series data for visualization"""
    try:
        return df.groupby(pd.Grouper(key=date_column, freq=freq))[value_column].count().reset_index()
    except Exception as e:
        logger.error(f"Error preparing time series data: {e}")
        return pd.DataFrame()

def calculate_moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    """Calculate moving average for trend analysis"""
    try:
        return series.rolling(window=window).mean()
    except Exception as e:
        logger.error(f"Error calculating moving average: {e}")
        return series

def detect_anomalies(series: pd.Series, threshold: float = 2.0) -> pd.Series:
    """Detect anomalies in time series data"""
    try:
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return pd.Series(dtype=bool)