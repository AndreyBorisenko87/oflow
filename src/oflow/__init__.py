"""
Scalping Strategy - Основной модуль

Торговая стратегия для скальпинга на криптовалютных рынках
с использованием детекторов паттернов и машинного обучения.
"""

from .pipeline import run_full_pipeline
from .blocks import (
    # Основные блоки
    DataLoaderBlock,
    ImportRawBlock,
    NormalizationBlock,
    
    # Функции запуска
    run_data_loader,
    import_raw_data,
    normalize_trades,
    normalize_depth,
    normalize_quotes
)

__version__ = "0.1.0"
__author__ = "Scalping Strategy Team"

__all__ = [
    # Основные функции
    "run_full_pipeline",
    
    # Блоки
    "DataLoaderBlock",
    "ImportRawBlock", 
    "NormalizationBlock",
    
    # Функции
    "run_data_loader",
    "import_raw_data",
    "normalize_trades",
    "normalize_depth",
    "normalize_quotes"
]
