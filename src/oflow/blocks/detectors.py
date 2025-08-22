"""
Блок 12: Детекторы
Модуль правил (интерфейс) + 8 детекторов паттернов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Protocol
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class BaseDetector(Protocol):
    """Базовый интерфейс для детекторов"""
    
    def detect(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Основной метод детекции"""
        ...
    
    def detect_with_progress(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Детекция с отслеживанием прогресса"""
        ...
    
    def get_name(self) -> str:
        """Название детектора"""
        ...

def run_detectors(
    detectors: List[BaseDetector],
    data: Dict[str, pd.DataFrame],
    output_dir: Path = Path("data/events"),
    use_progress: bool = True
) -> Dict[str, pd.DataFrame]:
    """Запуск всех детекторов с прогресс-логированием"""
    start_time = time.time()
    logger.info(f"=== Запуск {len(detectors)} детекторов ===")
    
    results = {}
    total_detectors = len(detectors)
    
    for i, detector in enumerate(detectors, 1):
        logger.info(f"[{i}/{total_detectors}] Запуск {detector.get_name()}")
        
        try:
            # Выбор метода детекции
            if use_progress:
                events = detector.detect_with_progress(data)
            else:
                events = detector.detect(data)
            
            if not events.empty:
                # Сохранение результатов
                output_path = output_dir / f"events_{detector.get_name().lower()}.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                events.to_parquet(output_path, engine="pyarrow", index=False)
                logger.info(f"✓ {detector.get_name()}: {len(events)} событий сохранено в {output_path}")
            else:
                logger.info(f"○ {detector.get_name()}: событий не найдено")
            
            results[detector.get_name()] = events
            
        except Exception as e:
            logger.error(f"✗ {detector.get_name()}: ошибка - {e}")
            results[detector.get_name()] = pd.DataFrame()
    
    # Итоговая статистика
    total_duration = time.time() - start_time
    total_events = sum(len(df) for df in results.values())
    successful_detectors = sum(1 for df in results.values() if not df.empty)
    
    logger.info(f"=== Детекция завершена ===")
    logger.info(f"Время выполнения: {total_duration:.2f}с")
    logger.info(f"Успешных детекторов: {successful_detectors}/{total_detectors}")
    logger.info(f"Всего событий: {total_events}")
    
    return results

def create_detectors_from_config(config: Dict) -> List[BaseDetector]:
    """Создание детекторов на основе конфигурации"""
    from .detectors import (
        D1LiquidityVacuumBreak,
        D2AbsorptionFlip,
        D3IcebergFade,
        D4StopRunContinuation,
        D5StopRunFailure,
        D6BookImbalance,
        D7SpoofPullTrap,
        D8MomentumIgnition
    )
    
    detectors = []
    
    # Создание детекторов с их конфигурациями
    if 'D1_LiquidityVacuumBreak' in config:
        detectors.append(D1LiquidityVacuumBreak(config['D1_LiquidityVacuumBreak']))
    
    if 'D2_AbsorptionFlip' in config:
        detectors.append(D2AbsorptionFlip(config['D2_AbsorptionFlip']))
    
    if 'D3_IcebergFade' in config:
        detectors.append(D3IcebergFade(config['D3_IcebergFade']))
    
    if 'D4_StopRunContinuation' in config:
        detectors.append(D4StopRunContinuation(config['D4_StopRunContinuation']))
    
    if 'D5_StopRunFailure' in config:
        detectors.append(D5StopRunFailure(config['D5_StopRunFailure']))
    
    if 'D6_BookImbalance' in config:
        detectors.append(D6BookImbalance(config['D6_BookImbalance']))
    
    if 'D7_SpoofPullTrap' in config:
        detectors.append(D7SpoofPullTrap(config['D7_SpoofPullTrap']))
    
    if 'D8_MomentumIgnition' in config:
        detectors.append(D8MomentumIgnition(config['D8_MomentumIgnition']))
    
    logger.info(f"Создано {len(detectors)} детекторов")
    return detectors
