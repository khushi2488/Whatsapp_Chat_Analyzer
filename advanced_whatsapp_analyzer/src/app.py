"""
Advanced WhatsApp Chat Analyzer - Main Streamlit Application
This is the main application file with enterprise-level features and UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
try:
    from preprocessor import AdvancedPreprocessor
    from components.sentiment_analyzer import SentimentAnalysisEngine
    from components.dashboard import DashboardComponents
    from components.visualizations import AdvancedVisualizations
    from components.export_manager import ExportManager
    from config.settings import settings, COLOR_SCHEMES
    from utils.helpers import format_duration, calculate_growth_rate, get_time_greeting
    from utils.validators import validate_file_type, validate_file_size
    from utils.security import sanitize_user_input, encrypt_sensitive_data
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.info("Please ensure all required modules are installed and properly configured.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=settings.ui.page_title,
    page_icon=settings.ui.page_icon,
    layout=settings.ui.layout,
    initial_sidebar_state=settings.ui.sidebar_state,
    menu_items={
        'Get Help': 'https://github.com/your-repo/help',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': f"{settings.app_name} v{settings.app_version}"
    }
)

# Custom CSS for better styling
def load_custom_css():
    """Load custom CSS for enhanced UI"""
    st.markdown("""
    <style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    
    /* Metrics styling */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e6e6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 0.5rem;
    }
    
    /* Alert boxes */
    .alert-success {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        color: #155724;
        margin: 1rem 0;
    }
    
    .alert-warning {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        color: #856404;
        margin: 1rem 0;
    }
    
    .alert-info {
        padding: 1rem;
        background-color: #cce7ff;
        border: 1px solid #99d6ff;
        border-radius: 0.5rem;
        color: #004085;
        margin: 1rem 0;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        text-align: center;
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

class ChatAnalyzerApp:
    """Main application class"""
    
    def __init__(self):
        self.preprocessor = None
        self.sentiment_engine = None
        self.dashboard = None
        self.visualizations = None
        self.export_manager = None
        
        # Initialize session state
        self._initialize_session_state()
        
        # Load components
        self._load_components()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        default_states = {
            'df': None,
            'processed': False,
            'analysis_complete': False,
            'processing_stats': None,
            'metadata': None,
            'selected_user': 'Overall',
            'analysis_type': 'overview',
            'export_ready': False,
            'sentiment_analyzed': False,
            'current_insights': None,
            'last_file_hash': None,
            'processing_progress': 0
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def _load_components(self):
        """Load application components"""
        try:
            self.preprocessor = AdvancedPreprocessor()
            
            if settings.features.enable_sentiment_analysis:
                self.sentiment_engine = SentimentAnalysisEngine()
            
            self.dashboard = DashboardComponents()
            self.visualizations = AdvancedVisualizations()
            
            if settings.features.enable_export_features:
                self.export_manager = ExportManager()
                
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            st.error(f"❌ Failed to load components: {e}")
    
    def render_header(self):
        """Render application header"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("🚀 Advanced WhatsApp Chat Analyzer")
            st.markdown(f"*v{settings.app_version} - Enterprise Edition*")
            
            # Add greeting based on time
            greeting = get_time_greeting()
            st.markdown(f"*{greeting}! Ready to analyze your conversations?*")
        
        # Show feature status
        if settings.debug:
            st.markdown("---")
            feature_cols = st.columns(5)
            features = [
                ("📊", "Sentiment Analysis", settings.features.enable_sentiment_analysis),
                ("🤖", "Topic Modeling", settings.features.enable_topic_modeling),
                ("📈", "Predictive Analytics", settings.features.enable_predictive_analytics),
                ("📤", "Export Features", settings.features.enable_export_features),
                ("⚡", "Real-time Processing", settings.features.enable_real_time_processing)
            ]
            
            for i, (icon, name, enabled) in enumerate(features):
                with feature_cols[i]:
                    status = "✅" if enabled else "❌"
                    st.markdown(f"{icon} {name}<br/>{status}", unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with file upload and controls"""
        st.sidebar.markdown("## 📂 File Upload")
        
        # File uploader
        uploaded_file = st.sidebar.file_uploader(
            "Choose a WhatsApp chat file",
            type=['txt', 'csv'],
            help="Export your WhatsApp chat (without media) as a .txt file"
        )
        
        if uploaded_file is not None:
            # Validate file
            validation_result = self._validate_uploaded_file(uploaded_file)
            
            if not validation_result['valid']:
                st.sidebar.error(validation_result['error'])
                return
            
            # File info
            file_details = {
                "filename": uploaded_file.name,
                "filetype": uploaded_file.type,
                "filesize": f"{uploaded_file.size / 1024:.2f} KB"
            }
            
            st.sidebar.markdown("### 📋 File Details")
            for key, value in file_details.items():
                st.sidebar.text(f"{key.capitalize()}: {value}")
            
            # Processing options
            st.sidebar.markdown("### ⚙️ Processing Options")
            
            # Language selection
            language = st.sidebar.selectbox(
                "Language",
                options=['auto', 'en', 'es', 'fr', 'de', 'it', 'pt'],
                index=0,
                help="Select chat language (auto-detection recommended)"
            )
            
            # Sentiment model selection
            if settings.features.enable_sentiment_analysis:
                sentiment_model = st.sidebar.selectbox(
                    "Sentiment Model",
                    options=['vader', 'textblob', 'transformers', 'multi'],
                    index=0,
                    help="Choose sentiment analysis model"
                )
            else:
                sentiment_model = 'vader'
            
            # Processing button
            if st.sidebar.button("🔄 Process Chat", type="primary"):
                self._process_uploaded_file(uploaded_file, language, sentiment_model)
        
        # Analysis controls (if data is processed)
        if st.session_state.processed and st.session_state.df is not None:
            st.sidebar.markdown("---")
            st.sidebar.markdown("## 👥 User Selection")
            
            # User selection
            users = ['Overall'] + list(st.session_state.df['user'].unique())
            users = [user for user in users if user != 'system']
            
            selected_user = st.sidebar.selectbox(
                "Select User for Analysis",
                options=users,
                index=users.index(st.session_state.selected_user) if st.session_state.selected_user in users else 0
            )
            st.session_state.selected_user = selected_user
            
            # Analysis type selection
            st.sidebar.markdown("## 📊 Analysis Type")
            analysis_types = [
                ("overview", "📈 Overview"),
                ("sentiment", "😊 Sentiment Analysis"),
                ("activity", "⏰ Activity Patterns"),
                ("users", "👥 User Analytics"),
                ("content", "💬 Content Analysis"),
                ("insights", "🤖 AI Insights")
            ]
            
            for value, label in analysis_types:
                if st.sidebar.button(label, key=f"btn_{value}"):
                    st.session_state.analysis_type = value
            
            # Export section
            if settings.features.enable_export_features and st.session_state.analysis_complete:
                st.sidebar.markdown("---")
                st.sidebar.markdown("## 📤 Export Options")
                
                export_format = st.sidebar.selectbox(
                    "Export Format",
                    options=['pdf', 'excel', 'csv', 'json'],
                    format_func=lambda x: {
                        'pdf': '📄 PDF Report',
                        'excel': '📊 Excel Workbook', 
                        'csv': '📋 CSV Data',
                        'json': '🔧 JSON Data'
                    }[x]
                )
                
                if st.sidebar.button("📤 Generate Export", type="secondary"):
                    self._generate_export(export_format)
    
    def _validate_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """Validate uploaded file"""
        try:
            # Check file type
            if not validate_file_type(uploaded_file.name, ['.txt', '.csv']):
                return {'valid': False, 'error': 'Invalid file type. Please upload a .txt or .csv file.'}
            
            # Check file size
            if not validate_file_size(uploaded_file.size, settings.files.max_file_size_mb * 1024 * 1024):
                return {'valid': False, 'error': f'File too large. Maximum size: {settings.files.max_file_size_mb} MB'}
            
            # Check if file is empty
            if uploaded_file.size == 0:
                return {'valid': False, 'error': 'File is empty.'}
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def _process_uploaded_file(self, uploaded_file, language: str, sentiment_model: str):
        """Process the uploaded WhatsApp chat file"""
        try:
            # Create file hash to detect changes
            file_content = uploaded_file.getvalue()
            import hashlib
            file_hash = hashlib.md5(file_content).hexdigest()
            
            # Check if file already processed
            if st.session_state.last_file_hash == file_hash:
                st.info("This file has already been processed. Showing existing results.")
                return
            
            # Show processing message
            with st.spinner("🔄 Processing chat file..."):
                data = file_content.decode("utf-8")
                file_size = len(file_content)
                
                # Sanitize input data
                data = sanitize_user_input(data)
                
                # Initialize preprocessor with language
                if language != 'auto':
                    self.preprocessor = AdvancedPreprocessor()
                
                # Process data with progress tracking
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Step 1: Preprocessing
                progress_text.text("📝 Parsing messages...")
                progress_bar.progress(25)
                df = self.preprocessor.preprocess(data, file_size)
                
                # Step 2: Sentiment Analysis
                if settings.features.enable_sentiment_analysis:
                    progress_text.text("😊 Analyzing sentiment...")
                    progress_bar.progress(50)
                    self.sentiment_engine = SentimentAnalysisEngine(sentiment_model, language)
                    df = self.sentiment_engine.analyze_dataframe(df)
                    st.session_state.sentiment_analyzed = True
                
                # Step 3: Additional Analysis
                progress_text.text("🔍 Extracting insights...")
                progress_bar.progress(75)
                time.sleep(0.5)  # Small delay for UX
                
                # Step 4: Finalization
                progress_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                # Update session state
                st.session_state.df = df
                st.session_state.processed = True
                st.session_state.analysis_complete = True
                st.session_state.processing_stats = self.preprocessor.get_processing_summary()
                st.session_state.metadata = self.preprocessor.metadata
                st.session_state.last_file_hash = file_hash
                
                # Clear progress indicators
                progress_bar.empty()
                progress_text.empty()
                
                # Show success message
                st.success(f"✅ Successfully processed {len(df)} messages!")
                
                # Show processing summary
                self._display_processing_summary()
                
                # Auto-scroll to results
                st.rerun()
                
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            st.error(f"❌ Error processing file: {e}")
            st.info("💡 Please check that your file is a valid WhatsApp chat export.")
            
            # Show troubleshooting tips
            with st.expander("🔧 Troubleshooting Tips"):
                st.markdown("""
                **Common issues:**
                1. File format not recognized - ensure it's a WhatsApp chat export
                2. Encoding issues - try saving the file as UTF-8
                3. File too large - consider splitting large chat histories
                4. Empty file - check that the chat contains messages
                
                **How to export WhatsApp chat:**
                1. Open WhatsApp on your phone
                2. Go to the chat you want to analyze
                3. Tap on chat name → More → Export Chat
                4. Choose "Without Media"
                5. Save the .txt file
                """)
    
    def _display_processing_summary(self):
        """Display processing summary"""
        if st.session_state.processing_stats:
            stats = st.session_state.processing_stats
            
            st.markdown("### 📊 Processing Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Messages Processed", 
                    stats['stats']['messages_processed'],
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Participants", 
                    stats['metadata']['participants'],
                    delta=None
                )
            
            with col3:
                duration_days = stats['metadata']['duration_days']
                st.metric(
                    "Duration", 
                    format_duration(duration_days),
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Processing Time", 
                    stats['stats']['processing_time'],
                    delta=None
                )
            
            # Detailed breakdown
            with st.expander("🔍 Detailed Breakdown", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Message Types:**")
                    st.text(f"• Text: {stats['stats']['messages_processed'] - stats['stats']['messages_filtered']}")
                    st.text(f"• Media: {stats['stats']['media_messages']}")
                    st.text(f"• System: {stats['stats']['system_messages']}")
                    st.text(f"• Deleted: {stats['stats']['deleted_messages']}")
                
                with col2:
                    st.markdown("**Chat Info:**")
                    st.text(f"• Language: {stats['metadata']['language']}")
                    st.text(f"• Chat Type: {stats['metadata']['chat_type']}")
                    st.text(f"• Date Range: {stats['metadata']['date_range']}")
    
    def render_main_content(self):
        """Render main analysis content based on selected analysis type"""
        if not st.session_state.processed or st.session_state.df is None:
            self._render_welcome_screen()
            return
        
        df = st.session_state.df
        selected_user = st.session_state.selected_user
        analysis_type = st.session_state.analysis_type
        
        # Filter data based on selected user
        if selected_user != 'Overall':
            filtered_df = df[df['user'] == selected_user].copy()
        else:
            filtered_df = df.copy()
        
        # Render content based on analysis type
        if analysis_type == 'overview':
            self._render_overview(filtered_df)
        elif analysis_type == 'sentiment':
            self._render_sentiment_analysis(filtered_df)
        elif analysis_type == 'activity':
            self._render_activity_analysis(filtered_df)
        elif analysis_type == 'users':
            self._render_user_analysis(df)  # Use full dataset for user comparison
        elif analysis_type == 'content':
            self._render_content_analysis(filtered_df)
        elif analysis_type == 'insights':
            self._render_ai_insights(filtered_df)
    
    def _render_welcome_screen(self):
        """Render welcome screen when no data is loaded"""
        st.markdown("# 👋 Welcome to Advanced WhatsApp Chat Analyzer")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ### 🚀 Enterprise-Grade Chat Analytics
            
            Transform your WhatsApp conversations into actionable insights with our advanced analytics platform.
            
            **✨ Features:**
            - 📊 **Advanced Sentiment Analysis** - Multi-model emotion detection
            - 🤖 **AI-Powered Insights** - Automated pattern recognition  
            - 📈 **Predictive Analytics** - Trend forecasting and risk assessment
            - 👥 **Team Productivity Metrics** - Response times and collaboration patterns
            - 🔒 **Enterprise Security** - GDPR compliant with data encryption
            - 📤 **Professional Reports** - Export to PDF, Excel, and more
            
            ---
            
            ### 🔧 How to Get Started:
            
            1. **Export your WhatsApp chat:**
               - Open WhatsApp on your phone
               - Go to the chat you want to analyze
               - Tap on chat name → More → Export Chat
               - Choose "Without Media"
               - Save the .txt file
            
            2. **Upload the file:**
               - Use the file uploader in the sidebar
               - Select your exported .txt file
               - Choose analysis options
               - Click "Process Chat"
            
            3. **Explore insights:**
               - View comprehensive analytics
               - Generate professional reports
               - Export data in multiple formats
            
            ---
            
            ### 📊 Sample Analytics Dashboard
            """)
            
            # Show sample metrics
            sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
            
            with sample_col1:
                st.metric("Messages Analyzed", "10,247", "+15%")
            
            with sample_col2:
                st.metric("Sentiment Score", "78%", "+5.2%")
            
            with sample_col3:
                st.metric("Avg Response Time", "2.4 min", "-12%")
            
            with sample_col4:
                st.metric("Active Users", "8", "+2")
            
            # Sample chart
            st.markdown("### 📈 Sample Sentiment Timeline")
            
            # Create sample data
            dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
            sample_data = pd.DataFrame({
                'Date': dates,
                'Positive': np.random.normal(0.6, 0.1, len(dates)),
                'Negative': np.random.normal(0.2, 0.05, len(dates)),
                'Neutral': np.random.normal(0.2, 0.05, len(dates))
            })
            
            fig = px.line(sample_data, x='Date', y=['Positive', 'Negative', 'Neutral'],
                         title="Sentiment Trends Over Time",
                         color_discrete_map={
                             'Positive': '#22c55e',
                             'Negative': '#ef4444',
                             'Neutral': '#6b7280'
                         })
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.info("💡 **Pro Tip:** For best results, use chat files with at least 100 messages spanning multiple days.")
    
    def _render_overview(self, df: pd.DataFrame):
        """Render overview analytics"""
        st.markdown(f"# 📈 Overview Analytics")
        if st.session_state.selected_user != 'Overall':
            st.markdown(f"*Analysis for: **{st.session_state.selected_user}***")
        
        # Use dashboard component for comprehensive overview
        self.dashboard.display_summary_dashboard(df)
        
        # Additional insights section
        st.markdown("### 💡 Quick Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            # Most active period
            hourly_activity = df.groupby('hour').size()
            peak_hour = hourly_activity.idxmax()
            peak_count = hourly_activity.max()
            
            st.info(f"🕐 **Peak Activity:** {peak_hour}:00-{peak_hour+1}:00 ({peak_count} messages)")
            
            # Most active day
            daily_activity = df.groupby('day_name').size()
            peak_day = daily_activity.idxmax()
            
            st.info(f"📅 **Most Active Day:** {peak_day}")
        
        with insights_col2:
            # Response time insight
            if 'response_time_minutes' in df.columns:
                avg_response = df['response_time_minutes'].mean()
                if not pd.isna(avg_response):
                    st.info(f"⚡ **Avg Response Time:** {avg_response:.1f} minutes")
            
            # Engagement insight
            if 'emoji_count' in df.columns:
                emoji_rate = len(df[df['emoji_count'] > 0]) / len(df) * 100
                st.info(f"😊 **Emoji Usage:** {emoji_rate:.1f}% of messages")
    
    def _render_sentiment_analysis(self, df: pd.DataFrame):
        """Render sentiment analysis section"""
        st.markdown("# 😊 Sentiment Analysis")
        
        if not st.session_state.sentiment_analyzed:
            st.warning("⚠️ Sentiment analysis not available. Please reprocess the chat with sentiment analysis enabled.")
            return
        
        # Check if sentiment data exists
        if 'sentiment_label' not in df.columns:
            st.error("❌ Sentiment data not found. Please reprocess the chat.")
            return
        
        # Filter text messages with sentiment data
        sentiment_df = df[df['sentiment_label'].notna()].copy()
        
        if sentiment_df.empty:
            st.warning("⚠️ No sentiment data available for the selected user.")
            return
        
        # Overall sentiment metrics
        col1, col2, col3, col4 = st.columns(4)
        
        sentiment_counts = sentiment_df['sentiment_label'].value_counts()
        total_messages = len(sentiment_df)
        
        with col1:
            positive_pct = (sentiment_counts.get('positive', 0) / total_messages * 100) if total_messages > 0 else 0
            st.metric("Positive Sentiment", f"{positive_pct:.1f}%", delta=None)
        
        with col2:
            negative_pct = (sentiment_counts.get('negative', 0) / total_messages * 100) if total_messages > 0 else 0
            st.metric("Negative Sentiment", f"{negative_pct:.1f}%", delta=None)
        
        with col3:
            neutral_pct = (sentiment_counts.get('neutral', 0) / total_messages * 100) if total_messages > 0 else 0
            st.metric("Neutral Sentiment", f"{neutral_pct:.1f}%", delta=None)
        
        with col4:
            if 'compound_score' in sentiment_df.columns:
                avg_compound = sentiment_df['compound_score'].mean()
                st.metric("Average Sentiment", f"{avg_compound:.2f}", delta=None)
        
        # Sentiment visualizations
        sentiment_fig = self.visualizations.create_sentiment_timeline(df)
        st.plotly_chart(sentiment_fig, use_container_width=True)
        
        # Sentiment distribution pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Sentiment Distribution")
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                        color_discrete_map=COLOR_SCHEMES['sentiment'],
                        title="Overall Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Sentiment Trends")
            # Create a gauge chart for overall sentiment health
            sentiment_score = positive_pct - negative_pct + 50  # Normalize to 0-100
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = sentiment_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sentiment Health Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_activity_analysis(self, df: pd.DataFrame):
        """Render activity analysis section"""
        st.markdown("# ⏰ Activity Patterns Analysis")
        
        # Activity timeline
        timeline_fig = self.visualizations.create_activity_timeline(df)
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Activity heatmap
        heatmap_fig = self.visualizations.create_hourly_heatmap(df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Response time analysis
        if 'response_time_minutes' in df.columns:
            response_fig = self.visualizations.create_response_time_analysis(df)
            st.plotly_chart(response_fig, use_container_width=True)
    
    def _render_user_analysis(self, df: pd.DataFrame):
        """Render user analysis section"""
        st.markdown("# 👥 User Analytics")
        
        # Filter out system messages
        user_df = df[df['user'] != 'system'].copy()
        
        if user_df.empty:
            st.warning("⚠️ No user data available for analysis.")
            return
        
        # User comparison chart
        user_fig = self.visualizations.create_user_comparison(df)
        st.plotly_chart(user_fig, use_container_width=True)
        
        # User statistics table
        user_stats = user_df.groupby('user').agg({
            'message_id': 'count',
            'message_length': 'mean',
            'word_count': 'sum' if 'word_count' in user_df.columns else 'count',
            'emoji_count': 'sum' if 'emoji_count' in user_df.columns else 'count',
            'url_count': 'sum' if 'url_count' in user_df.columns else 'count'
        }).round(2)
        
        user_stats.columns = ['Messages', 'Avg_Length', 'Total_Words', 'Emojis_Used', 'URLs_Shared']
        user_stats = user_stats.sort_values('Messages', ascending=False)
        
        st.markdown("### 📊 Detailed User Statistics")
        st.dataframe(user_stats, use_container_width=True)
    
    def _render_content_analysis(self, df: pd.DataFrame):
        """Render content analysis section"""
        st.markdown("# 💬 Content Analysis")
        
        # Word frequency analysis
        word_freq_fig = self.visualizations.create_word_frequency_chart(df)
        st.plotly_chart(word_freq_fig, use_container_width=True)
        
        # Message length analysis
        length_fig = self.visualizations.create_message_length_distribution(df)
        st.plotly_chart(length_fig, use_container_width=True)
        
        # Emoji analysis (if available)
        if 'emoji_count' in df.columns and df['emoji_count'].sum() > 0:
            emoji_fig = self.visualizations.create_emoji_analysis(df)
            st.plotly_chart(emoji_fig, use_container_width=True)
    
    def _render_ai_insights(self, df: pd.DataFrame):
        """Render AI-powered insights section"""
        st.markdown("# 🤖 AI-Powered Insights")
        
        # Generate comprehensive insights
        insights = self._generate_ai_insights(df)
        
        # Key insights cards
        st.markdown("### 🎯 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Communication Patterns")
            for insight in insights.get('communication_patterns', []):
                st.info(f"💡 {insight}")
        
        with col2:
            st.markdown("#### 👥 User Behavior")
            for insight in insights.get('user_behavior', []):
                st.info(f"👤 {insight}")
        
        # Recommendations
        st.markdown("### 🚀 Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Positive Trends")
            for rec in insights.get('positive_trends', []):
                st.success(f"✨ {rec}")
        
        with col2:
            st.markdown("#### ⚠️ Areas for Improvement")
            for rec in insights.get('improvements', []):
                st.warning(f"🔧 {rec}")
        
        # Advanced analytics
        if settings.features.enable_predictive_analytics:
            st.markdown("### 📈 Predictive Analytics")
            
            # Simple trend prediction
            daily_counts = df.groupby(df['date'].dt.date).size()
            
            if len(daily_counts) > 7:
                # Calculate moving average trend
                ma_7 = daily_counts.rolling(window=7).mean()
                trend = ma_7.iloc[-3:].mean() - ma_7.iloc[-10:-7].mean()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    trend_direction = "📈 Increasing" if trend > 5 else "📉 Decreasing" if trend < -5 else "➡️ Stable"
                    trend_value = f"+{trend:.1f}" if trend > 0 else f"{trend:.1f}"
                    st.metric("Activity Trend", trend_direction, f"{trend_value} msgs/day")
                
                with col2:
                    # Predict next week's activity
                    predicted_activity = ma_7.iloc[-1] + (trend * 7) if not ma_7.empty else 0
                    st.metric("Predicted Weekly Activity", f"{predicted_activity:.0f} messages")
                
                with col3:
                    # Communication health score
                    health_score = self._calculate_health_score(df)
                    st.metric("Communication Health", f"{health_score}/100")
    
    def _generate_ai_insights(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate AI-powered insights from the data"""
        insights = {
            'communication_patterns': [],
            'user_behavior': [],
            'positive_trends': [],
            'improvements': []
        }
        
        try:
            # Basic statistics
            total_messages = len(df)
            text_messages = len(df[df['message_type'] == 'text'])
            date_range_days = (df['date'].max() - df['date'].min()).days + 1
            
            # Communication patterns
            if date_range_days > 0:
                avg_daily_messages = total_messages / date_range_days
                insights['communication_patterns'].append(f"Average of {avg_daily_messages:.1f} messages per day over {date_range_days} days")
            
            # Peak activity analysis
            hourly_activity = df.groupby('hour').size()
            peak_hour = hourly_activity.idxmax()
            insights['communication_patterns'].append(f"Most active during {peak_hour}:00-{peak_hour+1}:00 with {hourly_activity.max()} messages")
            
            # Weekend vs weekday analysis
            if 'is_weekend' in df.columns:
                weekend_msgs = len(df[df['is_weekend'] == True])
                weekday_msgs = len(df[df['is_weekend'] == False])
                weekend_ratio = weekend_msgs / (weekend_msgs + weekday_msgs) * 100
                
                if weekend_ratio > 30:
                    insights['communication_patterns'].append(f"High weekend activity ({weekend_ratio:.1f}% of messages)")
                else:
                    insights['communication_patterns'].append(f"Primary weekday communication ({100-weekend_ratio:.1f}% weekday messages)")
            
            # User behavior analysis
            users = df[df['user'] != 'system']['user'].nunique()
            if users > 1:
                insights['user_behavior'].append(f"Group chat with {users} active participants")
                
                # User activity distribution
                user_activity = df[df['user'] != 'system']['user'].value_counts()
                top_user_pct = user_activity.iloc[0] / user_activity.sum() * 100
                
                if top_user_pct > 50:
                    insights['user_behavior'].append(f"Communication dominated by one user ({top_user_pct:.1f}% of messages)")
                else:
                    insights['user_behavior'].append(f"Well-balanced participation across users")
            else:
                insights['user_behavior'].append("Individual chat conversation")
            
            # Response time analysis
            if 'response_time_minutes' in df.columns:
                response_times = df[df['response_time_minutes'].notna() & (df['response_time_minutes'] < 1440)]
                if not response_times.empty:
                    avg_response = response_times['response_time_minutes'].mean()
                    if avg_response < 10:
                        insights['positive_trends'].append(f"Very responsive communication (avg {avg_response:.1f} min response time)")
                    elif avg_response > 60:
                        insights['improvements'].append(f"Slow response times (avg {avg_response:.1f} min)")
            
            # Sentiment analysis
            if 'sentiment_label' in df.columns and df['sentiment_label'].notna().any():
                sentiment_data = df[df['sentiment_label'].notna()]
                positive_pct = len(sentiment_data[sentiment_data['sentiment_label'] == 'positive']) / len(sentiment_data) * 100
                
                if positive_pct > 60:
                    insights['positive_trends'].append(f"Very positive communication tone ({positive_pct:.1f}% positive sentiment)")
                elif positive_pct < 40:
                    insights['improvements'].append(f"Consider more positive communication ({positive_pct:.1f}% positive sentiment)")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights['communication_patterns'].append("Unable to generate detailed insights due to data processing error")
            return insights
    
    def _calculate_health_score(self, df: pd.DataFrame) -> int:
        """Calculate overall communication health score"""
        try:
            score = 50  # Base score
            
            # Response time factor
            if 'response_time_minutes' in df.columns:
                response_times = df[df['response_time_minutes'].notna() & (df['response_time_minutes'] < 1440)]
                if not response_times.empty:
                    avg_response = response_times['response_time_minutes'].mean()
                    if avg_response < 30:
                        score += 20
                    elif avg_response < 120:
                        score += 10
                    elif avg_response > 480:
                        score -= 10
            
            # Sentiment factor
            if 'sentiment_label' in df.columns and df['sentiment_label'].notna().any():
                sentiment_data = df[df['sentiment_label'].notna()]
                positive_pct = len(sentiment_data[sentiment_data['sentiment_label'] == 'positive']) / len(sentiment_data)
                negative_pct = len(sentiment_data[sentiment_data['sentiment_label'] == 'negative']) / len(sentiment_data)
                
                score += int(positive_pct * 30)
                score -= int(negative_pct * 20)
            
            # Activity consistency factor
            daily_activity = df.groupby(df['date'].dt.date).size()
            if len(daily_activity) > 1:
                consistency = 1 - (daily_activity.std() / daily_activity.mean()) if daily_activity.mean() > 0 else 0
                score += int(consistency * 20)
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 50
    
    def _generate_export(self, export_format: str):
        """Generate export in specified format"""
        try:
            if not self.export_manager:
                st.error("❌ Export functionality not available")
                return
            
            with st.spinner(f"🔄 Generating {export_format.upper()} export..."):
                # Generate export using the ExportManager
                result = self.export_manager.export_data(
                    st.session_state.df, 
                    export_format,
                    include_analysis=True
                )
                
                if result.get('success'):
                    st.success(f"✅ {export_format.upper()} export generated successfully!")
                    
                    # Provide download button
                    st.download_button(
                        label=f"📥 Download {export_format.upper()}",
                        data=result['data'],
                        file_name=result['filename'],
                        mime=EXPORT_FORMATS.get(export_format, {}).get('mime_type', 'application/octet-stream')
                    )
                else:
                    st.error(f"❌ Export failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
    
    def run(self):
        """Run the main application"""
        try:
            # Load custom CSS
            load_custom_css()
            
            # Render header
            self.render_header()
            
            # Create main layout
            main_col, sidebar_col = st.columns([3, 1])
            
            with sidebar_col:
                self.render_sidebar()
            
            with main_col:
                self.render_main_content()
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error(f"❌ Application Error: {e}")
            
            if settings.debug:
                st.exception(e)

# Import EXPORT_FORMATS from config if not already imported
try:
    from config.settings import EXPORT_FORMATS
except ImportError:
    EXPORT_FORMATS = {
        'pdf': {'mime_type': 'application/pdf'},
        'excel': {'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'},
        'csv': {'mime_type': 'text/csv'},
        'json': {'mime_type': 'application/json'}
    }

# Initialize and run the application
if __name__ == "__main__":
    try:
        app = ChatAnalyzerApp()
        app.run()
    except Exception as e:
        st.error(f"💥 Critical Error: {e}")
        st.info("Please check your configuration and try again.")
        if settings.debug:
            st.exception(e)