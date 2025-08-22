"""
Блоки обработки данных для Scalping Strategy
"""

# Блок 02: Импорт сырья
from .import_raw import (
    load_yaml, collect_paths, read_trades, read_depth,
    normalize_trades, normalize_depth, run, run_test
)

# Блок 03: Нормализация (пока в import_raw.py)
# from .normalize import normalize_trades, normalize_depth

# Блок 04: Синхронизация времени
from .sync_time import (
    estimate_exchange_lags, sync_timestamps, filter_drops,
    validate_sync_quality, run_time_sync
)

# Блок 05: Лучшие цены
from .best_prices import (
    restore_quotes, calculate_spreads, aggregate_by_tick,
    validate_quotes, save_quotes, run_best_prices
)

# Блок 06: Топ-зона книги
from .book_top import (
    analyze_top_levels, track_changes, run_book_top
)

# Блок 07: Лента сделок
from .trade_tape import (
    run_trade_tape
)

# Блок 08: Кросс-биржевая агрегация
from .cross_exchange import (
    run_cross_exchange
)

# Блок 09: Базис спот-фьючерс
from .basis import (
    run_basis
)

# Блок 10: Фичи паттернов
from .features import (
    DetectorFeatureExtractor, run_features
)

# Блок 11: Канонический слой
from .canonical import (
    CanonicalStorage, save_canonical, batch_by_days, run_canonical
)

# Блок 12: Детекторы
from .detectors import (
    BaseDetector, D1LiquidityVacuumBreak, D2AbsorptionFlip,
    D3IcebergFade, D4StopRunContinuation, D5StopRunFailure,
    D6BookImbalance, D7SpoofPullTrap, D8MomentumIgnition,
    run_detectors, create_detectors_from_config
)

# Блок 13: Сканер
from .scanner import (
    DataScanner, run_scanner
)

# Блок 14: Бэктест (пока заглушки)
# from .backtest import (
#     BacktestEngine, generate_report, run_backtest
# )

# Блок 15: Экспорт разметки (пока заглушки)
# from .export import (
#     export_to_csv, format_for_tradingview, format_for_atas, run_export
# )

__version__ = "0.1.0"
__all__ = [
    # Блок 02
    "load_yaml", "collect_paths", "read_trades", "read_depth",
    "normalize_trades", "normalize_depth", "run", "run_test",
    
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
    "BaseDetector", "D1LiquidityVacuumBreak", "D2AbsorptionFlip",
    "D3IcebergFade", "D4StopRunContinuation", "D5StopRunFailure",
    "D6BookImbalance", "D7SpoofPullTrap", "D8MomentumIgnition",
    "run_detectors", "create_detectors_from_config",
    
    # Блок 13
    "DataScanner", "run_scanner",
    
    # Блок 14 (пока заглушки)
    # "BacktestEngine", "generate_report", "run_backtest",
    
    # Блок 15 (пока заглушки)
    # "export_to_csv", "format_for_tradingview", "format_for_atas", "run_export",
]
