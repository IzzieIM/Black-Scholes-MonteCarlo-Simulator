"""
visualization/__init__.py
Makes the 'visualization' directory a Python package.

Provides easy access to plotting functions.
"""

# Import the main visualizer class
from visualization.plots import OptionVisualizer

# Define what gets imported with "from visualization import *"
__all__ = [
    'OptionVisualizer'
]

# Package metadata
__version__ = '1.0.0'
__description__ = 'Visualization dashboard for options pricing system'