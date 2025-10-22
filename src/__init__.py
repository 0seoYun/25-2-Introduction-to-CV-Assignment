"""
KNN CIFAR-10 Classification Package

This package provides modular components for KNN classification on CIFAR-10 dataset.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import data_loader
from . import knn_models
from . import metrics
from . import visualization
from . import report_generator

__all__ = [
    'data_loader',
    'knn_models',
    'metrics',
    'visualization',
    'report_generator'
]
