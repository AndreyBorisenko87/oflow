"""
Блоки обработки данных для Scalping Strategy
"""

# Блок 01: Загрузчик данных
from .b01_data_loader import (
    DataLoaderBlock, run_data_loader
)

# Блок 02: Импорт сырья
from .b02_import_raw import (
    ImportRawBlock, import_raw_data
)

# Блок 03: Нормализация
from .b03_normalization import (
    NormalizationBlock, normalize_trades, normalize_depth, normalize_quotes
)

# Блок 04: Синхронизация времени
from .b04_sync_time import (
    estimate_exchange_lags, sync_timestamps, filter_drops,
    validate_sync_quality, run_time_sync
)

# Блок 05: Лучшие цены
from .b05_best_prices import (
    restore_quotes, calculate_spreads, aggregate_by_tick,
    validate_quotes, save_quotes, run_best_prices
)

# Блок 06: Топ-зона книги
from .b06_book_top import (
    analyze_top_levels, track_changes, run_book_top
)

# Блок 07: Лента сделок
from .b07_trade_tape import (
    run_trade_tape
)

# Блок 08: Кросс-биржевая агрегация
from .b08_cross_exchange import (
    run_cross_exchange
)

# Блок 09: Базис спот-фьючерс
from .b09_basis import (
    run_basis
)

# Блок 10: Фичи паттернов
from .b10_features import (
    DetectorFeatureExtractor, run_features
)

# Блок 11: Канонический слой
from .b11_canonical import (
    CanonicalStorage, save_canonical, batch_by_days, run_canonical
)

# Блок 12: Детекторы
from .b12_detectors import (
    BaseDetector, run_detectors, create_detectors_from_config
)

# Блок 13: Сканер
from .b13_scanner import (
    DataScanner, run_scanner
)

# Блок 14: Бэктест
from .b14_backtest import (
    BacktestBlock, BacktestEngine, run_backtest
)

# Блок 15: Экспорт разметки
from .b15_export import (
    ExportBlock, export_data, export_to_csv, format_for_tradingview, format_for_atas
)

__version__ = "0.1.0"
__all__ = [
    # Блок 01
    "DataLoaderBlock", "run_data_loader",
    
    # Блок 02
    "ImportRawBlock", "import_raw_data",
    
    # Блок 03
    "NormalizationBlock", "normalize_trades", "normalize_depth", "normalize_quotes",
    
    # Блок 04
    "estimate_exchange_lags", "sync_timestamps", "filter_drops",
    "validate_sync_quality", "run_time_sync",
    
    # Блок 05
    "restore_quotes", "calculate_spreads", "aggregate_by_tick",
    "validate_quotes", "save_quotes", "run_best_prices",
    
    # Блок 06
    "analyze_top_levels", "track_changes", "run_book_top",
    
    # Блок 07
    "run_trade_tape",
    
    # Блок 08
    "run_cross_exchange",
    
    # Блок 09
    "run_basis",
    
    # Блок 10
    "DetectorFeatureExtractor", "run_features",
    
    # Блок 11
    "CanonicalStorage", "save_canonical", "batch_by_days", "run_canonical",
    
    # Блок 12
    "BaseDetector", "run_detectors", "create_detectors_from_config",
    
    # Блок 13
    "DataScanner", "run_scanner",
    
    # Блок 14
    "BacktestBlock", "BacktestEngine", "run_backtest",
    
    # Блок 15
    "ExportBlock", "export_data", "export_to_csv", "format_for_tradingview", "format_for_atas",
]
