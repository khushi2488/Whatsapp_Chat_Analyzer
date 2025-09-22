"""
Setup configuration for Advanced WhatsApp Chat Analyzer
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = this_directory / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            requirements = f.read().splitlines()
        # Filter out comments and empty lines
        return [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
    return []

# Get version from environment or default
version = os.getenv('VERSION', '2.0.0')

setup(
    name="advanced-whatsapp-chat-analyzer",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="Enterprise-grade WhatsApp chat analysis with advanced AI-powered insights",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/advanced-whatsapp-chat-analyzer",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/advanced-whatsapp-chat-analyzer/issues",
        "Source": "https://github.com/yourusername/advanced-whatsapp-chat-analyzer",
        "Documentation": "https://github.com/yourusername/advanced-whatsapp-chat-analyzer/docs",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "Topic :: Communications :: Chat",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "tensorflow-gpu>=2.13.0",
        ],
        "enterprise": [
            "redis>=4.5.0",
            "celery>=5.3.0",
            "gunicorn>=21.0.0",
            "prometheus-client>=0.19.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "whatsapp-analyzer=run:main",
            "chat-analyzer=run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.json"],
        "assets": ["**/*"],
        "config": ["**/*"],
    },
    zip_safe=False,
    keywords=[
        "whatsapp", "chat", "analysis", "sentiment", "nlp", "ai", "machine-learning",
        "data-visualization", "streamlit", "business-intelligence", "communication"
    ],
    platforms=["any"],
)

# Post-install message
print("""
🚀 Advanced WhatsApp Chat Analyzer installed successfully!

To get started:
1. Run: python run.py
2. Or use: whatsapp-analyzer
3. Open your browser to: http://localhost:8501

For documentation, visit:
https://github.com/yourusername/advanced-whatsapp-chat-analyzer

Need help? Create an issue at:
https://github.com/yourusername/advanced-whatsapp-chat-analyzer/issues
""")