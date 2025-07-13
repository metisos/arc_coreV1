"""
ARC Core - Adaptive Recursive Consciousness Engine

A modular continual learning system for language models with biological learning mechanisms.
"""

from setuptools import setup, find_packages
import os
import sys

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Version info
version_info = {}
with open(os.path.join('arc_core', '__init__.py'), 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            exec(line, version_info)
            break

setup(
    name="metisos-arc-core",
    version=version_info.get('__version__', '0.1.0'),
    author="Metis AI Research",
    author_email="research@metisai.dev",
    description="Adaptive Recursive Consciousness Engine - Modular continual learning for language models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/metisai/arc-core",
    project_urls={
        "Bug Tracker": "https://github.com/metisai/arc-core/issues",
        "Documentation": "https://arc-core.readthedocs.io/",
        "Source Code": "https://github.com/metisai/arc-core",
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'arc_core': ['packs/**/*'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "pre-commit>=2.15",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.12",
        ],
        "gpu": [
            "torch[cuda]>=1.12.0",
        ],
        "apple": [
            "torch>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "arc=cli.arc:cli",
        ],
    },
    keywords=[
        "artificial intelligence",
        "machine learning",
        "continual learning",
        "language models",
        "cognitive science",
        "neural networks",
        "biologically inspired",
        "recursive consciousness",
        "adaptive learning",
        "memory systems",
    ],
    zip_safe=False,
)
