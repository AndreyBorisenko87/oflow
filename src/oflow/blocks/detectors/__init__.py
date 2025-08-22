"""
Детекторы паттернов для торговой стратегии
"""

from .base_detector import BaseDetector
from .d1_liquidity_vacuum_break import D1LiquidityVacuumBreak
from .d2_absorption_flip import D2AbsorptionFlip
from .d3_iceberg_fade import D3IcebergFade
from .d4_stop_run_continuation import D4StopRunContinuation
from .d5_stop_run_failure import D5StopRunFailure
from .d6_book_imbalance import D6BookImbalance
from .d7_spoof_pull_trap import D7SpoofPullTrap
from .d8_momentum_ignition import D8MomentumIgnition

# Импорт функций для запуска детекторов
from .detector_runner import run_detectors, create_detectors_from_config

__all__ = [
    'BaseDetector',
    'D1LiquidityVacuumBreak',
    'D2AbsorptionFlip',
    'D3IcebergFade',
    'D4StopRunContinuation',
    'D5StopRunFailure',
    'D6BookImbalance',
    'D7SpoofPullTrap',
    'D8MomentumIgnition',
    'run_detectors',
    'create_detectors_from_config'
]
