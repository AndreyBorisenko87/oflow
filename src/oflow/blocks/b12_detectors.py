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


if __name__ == "__main__":
    """Запуск блока 12 для тестирования"""
    import pandas as pd
    import logging
    import sys
    import os
    import time
    
    # Добавляем путь для импорта base_block
    sys.path.append(os.path.dirname(__file__))
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    print("🧪 ТЕСТ БЛОКА 12: Detectors")
    print("=" * 50)
    
    try:
        # Загружаем данные от предыдущих блоков
        features_df = pd.read_parquet("../../../data/features/detector_features.parquet")
        
        print(f"✅ Загружены данные для детекторов:")
        print(f"   - Features: {len(features_df)} строк")
        
        # Проверяем какие биржи у нас есть
        if not features_df.empty:
            exchanges = features_df['exchange'].unique()
            print(f"   - Биржи в features: {exchanges}")
        
        # Конфигурация для детекторов
        config = {
            'D1_LiquidityVacuumBreak': {
                'threshold': 0.1,
                'empty_levels': 3
            },
            'D2_AbsorptionFlip': {
                'min_side_ratio': 0.7,
                'max_price_move_ticks': 2
            },
            'D3_IcebergFade': {
                'same_price_trades': 10,
                'time_window_ms': 2000
            },
            'D4_StopRunContinuation': {
                'min_price_move_percent': 0.3,
                'time_window_ms': 2000
            },
            'D5_StopRunFailure': {
                'reversal_threshold_ratio': 0.7,
                'reversal_window_ms': 5000
            },
            'D6_BookImbalance': {
                'bid_ask_ratio': 3.0,
                'level_wall_multiplier': 5.0
            },
            'D7_SpoofPullTrap': {
                'large_order_multiplier': 5.0,
                'cancel_time_window_ms': 1000
            },
            'D8_MomentumIgnition': {
                'acceleration_window_ms': 3000,
                'min_volume_ratio': 3.0
            }
        }
        
        # Создаем данные для детекторов
        data = {
            'features': features_df
        }
        
        # Запускаем блок 12
        print("🚀 Запускаем блок 12: Детекторы паттернов...")
        start_time = time.time()
        
        # Создаем детекторы
        detectors = create_detectors_from_config(config)
        print(f"   - Создано детекторов: {len(detectors)}")
        
        # Запускаем детекторы
        results = run_detectors(detectors, data)
        
        execution_time = time.time() - start_time
        
        print(f"✅ Блок 12 выполнен за {execution_time:.2f} секунд!")
        print(f"📊 Результат детекции:")
        
        # Анализируем результаты
        total_events = sum(len(df) for df in results.values())
        successful_detectors = sum(1 for df in results.values() if not df.empty)
        
        print(f"   - Успешных детекторов: {successful_detectors}/{len(detectors)}")
        print(f"   - Всего событий: {total_events}")
        
        for detector_name, events_df in results.items():
            if not events_df.empty:
                print(f"   - {detector_name}: {len(events_df)} событий")
            else:
                print(f"   - {detector_name}: событий не найдено")
        
        print("✅ Блок 12 успешно протестирован!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()