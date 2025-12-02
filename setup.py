"""
Setup script for Agent Forge V2
Use: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    req_file = Path(__file__).parent / filename
    if not req_file.exists():
        return []
    with open(req_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="agent-forge-v2",
    version="1.0.0",
    description="8-phase AI agent creation pipeline with 25M parameter models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Agent Forge Team",
    url="https://github.com/agent-forge/agent-forge-v2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "peft>=0.7.0",
        "wandb>=0.16.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": read_requirements("config/requirements-dev.txt"),
        "ui": read_requirements("config/requirements-ui.txt"),
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
