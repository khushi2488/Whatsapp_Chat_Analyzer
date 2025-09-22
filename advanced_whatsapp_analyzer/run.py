#!/usr/bin/env python3
"""
Main runner for Advanced WhatsApp Chat Analyzer
This file handles application startup, logging setup, and error handling.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from loguru import logger
    from config.settings import settings
    # import streamlit.cli as stcli
    import streamlit.web.cli as stcli

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Packages installed. Please run again.")
    sys.exit(1)

def setup_logging():
    """Configure logging for the application"""
    try:
        # Remove default logger
        logger.remove()
        
        # Add console logger
        logger.add(
            sys.stdout,
            level=settings.logging.level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
            colorize=True
        )
        
        # Add file logger
        logger.add(
            settings.logging.log_file,
            level=settings.logging.level,
            format=settings.logging.format,
            rotation=f"{settings.logging.max_size_mb} MB",
            retention=f"{settings.logging.backup_count} files",
            compression="zip"
        )
        
        logger.info("🔧 Logging configured successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to setup logging: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "streamlit", "pandas", "numpy", "scikit-learn", 
        "transformers", "plotly", "nltk", "spacy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {missing_packages}")
        logger.info("📦 Run: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All dependencies are installed")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        
        # Required NLTK data
        nltk_data = [
            'punkt', 'vader_lexicon', 'stopwords', 
            'wordnet', 'averaged_perceptron_tagger'
        ]
        
        for data in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                logger.info(f"📥 Downloading NLTK data: {data}")
                nltk.download(data, quiet=True)
        
        logger.info("✅ NLTK data ready")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download NLTK data: {e}")
        return False

def setup_spacy_model():
    """Download and setup spaCy language model"""
    try:
        import spacy
        
        model_name = "en_core_web_sm"
        try:
            spacy.load(model_name)
            logger.info(f"✅ spaCy model '{model_name}' is available")
        except OSError:
            logger.info(f"📥 Downloading spaCy model: {model_name}")
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", model_name
            ])
            logger.info(f"✅ spaCy model '{model_name}' downloaded")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to setup spaCy model: {e}")
        logger.info("📝 You can continue without spaCy, but some features may be limited")
        return False

def create_required_directories():
    """Create necessary directories"""
    directories = [
        "logs", "uploads", "temp", "exports", 
        "models/trained", "data/processed"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("📁 Created required directories")

def validate_configuration():
    """Validate application configuration"""
    try:
        # Test database connection
        if settings.database.url.startswith("sqlite"):
            # For SQLite, just check if we can create the file
            db_path = Path(settings.database.url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
        # Test file permissions
        test_file = settings.files.upload_folder / "test.txt"
        test_file.write_text("test")
        test_file.unlink()
        
        # Validate feature flags
        if not any([
            settings.features.enable_sentiment_analysis,
            settings.features.enable_topic_modeling,
            settings.features.enable_predictive_analytics
        ]):
            logger.warning("⚠️  No analysis features enabled!")
        
        logger.info("✅ Configuration validated")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False

def print_startup_banner():
    """Print application startup banner"""
    banner = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                          ║
    ║                   🚀 {settings.app_name}                   ║
    ║                                    Version {settings.app_version}                                    ║
    ║                                                                                          ║
    ║  📊 Advanced Analytics  •  🤖 AI-Powered Insights  •  🔒 Enterprise Security         ║
    ║                                                                                          ║
    ║  Environment: {settings.environment:<10} | Debug: {str(settings.debug):<5} | Features: {sum(1 for f in [settings.features.enable_sentiment_analysis, settings.features.enable_topic_modeling, settings.features.enable_predictive_analytics, settings.features.enable_export_features] if f)}/4      ║
    ║                                                                                          ║
    ║  🌐 Access: http://localhost:8501                                                       ║
    ║  📚 Docs: /docs  |  🔧 Admin: /admin  |  📊 Health: /health                          ║
    ║                                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Feature status
    features_status = []
    if settings.features.enable_sentiment_analysis:
        features_status.append("✅ Sentiment Analysis")
    if settings.features.enable_topic_modeling:
        features_status.append("✅ Topic Modeling") 
    if settings.features.enable_predictive_analytics:
        features_status.append("✅ Predictive Analytics")
    if settings.features.enable_export_features:
        features_status.append("✅ Export Features")
    if settings.features.enable_real_time_processing:
        features_status.append("✅ Real-time Processing")
    
    if features_status:
        print("🎯 Enabled Features:")
        for feature in features_status:
            print(f"   {feature}")
        print()

def run_streamlit_app():
    """Run the Streamlit application"""
    try:
        # Streamlit configuration
        streamlit_args = [
            "streamlit", "run", "src/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true" if not settings.debug else "false",
            "--server.enableXsrfProtection", "true",
            "--server.enableCORS", "false",
            "--theme.base", settings.ui.theme,
            "--ui.hideTopBar", "false",
        ]
        
        # Add development-specific args
        if settings.debug:
            streamlit_args.extend([
                "--server.runOnSave", "true",
                "--server.allowRunOnSave", "true"
            ])
        
        logger.info("🚀 Starting Streamlit application...")
        logger.info(f"🌐 Application will be available at: http://localhost:8501")
        
        # Use stcli.main instead of subprocess for better integration
        sys.argv = streamlit_args
        stcli.main()
        
    except KeyboardInterrupt:
        logger.info("⏹️  Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        sys.exit(1)

def main():
    """Main application entry point"""
    print("🔄 Initializing Advanced WhatsApp Chat Analyzer...")
    
    # Setup logging first
    if not setup_logging():
        print("❌ Failed to setup logging. Continuing without file logging...")
    
    logger.info("🚀 Starting application initialization...")
    
    # Create required directories
    create_required_directories()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Download required data
    logger.info("📥 Setting up language models and data...")
    download_nltk_data()
    setup_spacy_model()
    
    # Validate configuration
    if not validate_configuration():
        logger.error("❌ Configuration validation failed. Please check your settings.")
        sys.exit(1)
    
    # Print startup banner
    print_startup_banner()
    
    # Log system information
    logger.info(f"🖥️  Python version: {sys.version}")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"🔧 Environment: {settings.environment}")
    logger.info(f"💾 Database: {settings.database.url}")
    logger.info(f"🌐 Languages supported: {len(settings.analytics.default_language)}")
    
    # Start the application
    try:
        run_streamlit_app()
    except Exception as e:
        logger.critical(f"💥 Critical error during startup: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)