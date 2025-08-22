#!/usr/bin/env python3
"""
Скрипт для создания уменьшенных тестовых данных
Уменьшает объем данных для быстрого тестирования блоков
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def create_small_trades_data(input_file: str, output_file: str, max_records: int = 5000):
    """Создает уменьшенную версию trades данных"""
    logger.info(f"Создание уменьшенных trades данных: {max_records} записей")
    
    # Загружаем исходные данные
    trades_df = pd.read_parquet(input_file)
    logger.info(f"Загружено {len(trades_df)} исходных trades записей")
    
    if len(trades_df) <= max_records:
        logger.info("Данные уже достаточно малы, копируем как есть")
        trades_df.to_parquet(output_file, index=False)
        return
    
    # Создаем уменьшенную версию
    # Берем равномерно распределенные записи по времени и биржам
    small_trades = []
    
    for exchange in trades_df['exchange'].unique():
        ex_data = trades_df[trades_df['exchange'] == exchange]
        target_count = max(1, int(max_records * len(ex_data) / len(trades_df)))
        
        # Берем равномерно распределенные записи
        step = max(1, len(ex_data) // target_count)
        selected_indices = range(0, len(ex_data), step)[:target_count]
        
        small_trades.append(ex_data.iloc[selected_indices])
        logger.info(f"  {exchange}: {len(ex_data)} -> {len(selected_indices)} записей")
    
    # Объединяем и сохраняем
    small_df = pd.concat(small_trades, ignore_index=True)
    small_df = small_df.sort_values('ts_ns').reset_index(drop=True)
    
    small_df.to_parquet(output_file, index=False)
    logger.info(f"Сохранено {len(small_df)} trades записей в {output_file}")

def create_small_depth_data(input_file: str, output_file: str, max_records: int = 10000):
    """Создает уменьшенную версию depth данных"""
    logger.info(f"Создание уменьшенных depth данных: {max_records} записей")
    
    # Загружаем исходные данные
    depth_df = pd.read_parquet(input_file)
    logger.info(f"Загружено {len(depth_df)} исходных depth записей")
    
    if len(depth_df) <= max_records:
        logger.info("Данные уже достаточно малы, копируем как есть")
        depth_df.to_parquet(output_file, index=False)
        return
    
    # Создаем уменьшенную версию
    # Берем равномерно распределенные записи по времени и биржам
    small_depth = []
    
    for exchange in depth_df['exchange'].unique():
        ex_data = depth_df[depth_df['exchange'] == exchange]
        target_count = max(1, int(max_records * len(ex_data) / len(depth_df)))
        
        # Берем равномерно распределенные записи
        step = max(1, len(ex_data) // target_count)
        selected_indices = range(0, len(ex_data), step)[:target_count]
        
        small_depth.append(ex_data.iloc[selected_indices])
        logger.info(f"  {exchange}: {len(ex_data)} -> {len(selected_indices)} записей")
    
    # Объединяем и сохраняем
    small_df = pd.concat(small_depth, ignore_index=True)
    small_df = small_df.sort_values('ts_ns').reset_index(drop=True)
    
    small_df.to_parquet(output_file, index=False)
    logger.info(f"Сохранено {len(small_df)} depth записей в {output_file}")

def main():
    """Основная функция создания уменьшенных данных"""
    logger.info("🚀 СОЗДАНИЕ УМЕНЬШЕННЫХ ТЕСТОВЫХ ДАННЫХ")
    logger.info("=" * 60)
    
    # Пути к файлам
    input_dir = Path("../../../data/normalized")
    output_dir = Path("../../../data/test_small")
    
    # Создаем выходную директорию
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем уменьшенные данные
    try:
        # Trades данные
        trades_input = input_dir / "trades.parquet"
        trades_output = output_dir / "trades_small.parquet"
        
        if trades_input.exists():
            create_small_trades_data(str(trades_input), str(trades_output), max_records=5000)
        else:
            logger.warning(f"Файл {trades_input} не найден")
        
        # Depth данные
        depth_input = input_dir / "book_top.parquet"
        depth_output = output_dir / "book_top_small.parquet"
        
        if depth_input.exists():
            create_small_depth_data(str(depth_input), str(depth_output), max_records=10000)
        else:
            logger.warning(f"Файл {depth_input} не найден")
        
        logger.info("✅ Уменьшенные тестовые данные созданы успешно!")
        logger.info(f"📁 Выходная папка: {output_dir.absolute()}")
        
        # Показываем размеры файлов
        for file_path in output_dir.glob("*.parquet"):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"📊 {file_path.name}: {size_mb:.1f} MB")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при создании уменьшенных данных: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
