"""
Блок 03: Нормализация
Приведение данных к единому формату для всех бирж
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def normalize_trades(
    df: pd.DataFrame,
    config: Dict = None
) -> pd.DataFrame:
    """
    Нормализация trades данных к единому формату
    
    Args:
        df: Сырые trades данные
        config: Конфигурация нормализации
        
    Returns:
        Нормализованный DataFrame
    """
    if config is None:
        config = {}
    
    log_progress = config.get('log_progress', True)
    
    if log_progress:
        logger.info("Этап 1/3: Нормализация trades данных")
    else:
        logger.info("Нормализация trades данных...")
    
    if df.empty:
        logger.warning("Trades DataFrame пустой")
        return df
    
    # Создаем копию для модификации
    df_normalized = df.copy()
    
    # 1. Стандартизация колонок
    column_mapping = {
        'timestamp': 'ts_ns',
        'time': 'ts_ns',
        'ts': 'ts_ns',
        'price': 'price',
        'amount': 'size',
        'quantity': 'size',
        'qty': 'size',
        'volume': 'size',
        'side': 'side',
        'type': 'type',
        'trade_id': 'trade_id',
        'order_id': 'order_id'
    }
    
    # Переименовываем колонки если нужно
    for old_name, new_name in column_mapping.items():
        if old_name in df_normalized.columns and new_name not in df_normalized.columns:
            df_normalized = df_normalized.rename(columns={old_name: new_name})
            logger.debug(f"Переименована колонка: {old_name} -> {new_name}")
    
    # 2. Нормализация временных меток
    if 'ts_ns' in df_normalized.columns:
        # Конвертируем в наносекунды если нужно
        if df_normalized['ts_ns'].dtype == 'object':
            # Пробуем разные форматы времени
            try:
                df_normalized['ts_ns'] = pd.to_datetime(df_normalized['ts_ns']).astype(np.int64)
            except:
                logger.warning("Не удалось конвертировать временные метки")
        elif df_normalized['ts_ns'].dtype in ['int64', 'int32']:
            # Проверяем, что это наносекунды (обычно > 1e15)
            if df_normalized['ts_ns'].max() < 1e15:
                # Вероятно секунды или миллисекунды
                if df_normalized['ts_ns'].max() < 1e12:  # Секунды
                    df_normalized['ts_ns'] = df_normalized['ts_ns'] * 1_000_000_000
                else:  # Миллисекунды
                    df_normalized['ts_ns'] = df_normalized['ts_ns'] * 1_000_000
    
    # 3. Нормализация стороны сделки
    if 'side' in df_normalized.columns:
        # Приводим к нижнему регистру
        df_normalized['side'] = df_normalized['side'].str.lower()
        
        # Стандартизируем значения
        side_mapping = {
            'buy': 'buy',
            'b': 'buy',
            'long': 'buy',
            'sell': 'sell',
            's': 'sell',
            'short': 'sell'
        }
        
        df_normalized['side'] = df_normalized['side'].map(side_mapping).fillna(df_normalized['side'])
        
        # Проверяем уникальные значения
        unique_sides = df_normalized['side'].unique()
        logger.info(f"Уникальные стороны сделок: {unique_sides}")
    
    # 4. Нормализация типов сделок
    if 'type' in df_normalized.columns:
        df_normalized['type'] = df_normalized['type'].str.lower()
        
        type_mapping = {
            'market': 'market',
            'm': 'market',
            'limit': 'limit',
            'l': 'limit'
        }
        
        df_normalized['type'] = df_normalized['type'].map(type_mapping).fillna(df_normalized['type'])
    
    # 5. Валидация данных
    if 'price' in df_normalized.columns:
        # Удаляем записи с неположительными ценами
        invalid_prices = (df_normalized['price'] <= 0).sum()
        if invalid_prices > 0:
            logger.warning(f"Удалено {invalid_prices} записей с неположительными ценами")
            df_normalized = df_normalized[df_normalized['price'] > 0]
    
    if 'size' in df_normalized.columns:
        # Удаляем записи с неположительными объемами
        invalid_sizes = (df_normalized['size'] <= 0).sum()
        if invalid_sizes > 0:
            logger.warning(f"Удалено {invalid_sizes} записей с неположительными объемами")
            df_normalized = df_normalized[df_normalized['size'] > 0]
    
    # 6. Сортировка по времени
    if 'ts_ns' in df_normalized.columns:
        df_normalized = df_normalized.sort_values('ts_ns').reset_index(drop=True)
    
    logger.info(f"Нормализовано {len(df_normalized)} trades записей")
    return df_normalized

def normalize_depth(
    df: pd.DataFrame,
    config: Dict = None
) -> pd.DataFrame:
    """
    Нормализация depth данных к единому формату
    
    Args:
        df: Сырые depth данные
        config: Конфигурация нормализации
        
    Returns:
        Нормализованный DataFrame
    """
    if config is None:
        config = {}
    
    log_progress = config.get('log_progress', True)
    
    if log_progress:
        logger.info("Этап 2/3: Нормализация depth данных")
    else:
        logger.info("Нормализация depth данных...")
    
    if df.empty:
        logger.warning("Depth DataFrame пустой")
        return df
    
    # Создаем копию для модификации
    df_normalized = df.copy()
    
    # 1. Стандартизация колонок
    column_mapping = {
        'timestamp': 'ts_ns',
        'time': 'ts_ns',
        'ts': 'ts_ns',
        'price': 'price',
        'amount': 'size',
        'quantity': 'size',
        'qty': 'size',
        'volume': 'size',
        'side': 'side',
        'level': 'level',
        'order_id': 'order_id',
        'update_type': 'update_type'
    }
    
    # Переименовываем колонки если нужно
    for old_name, new_name in column_mapping.items():
        if old_name in df_normalized.columns and new_name not in df_normalized.columns:
            df_normalized = df_normalized.rename(columns={old_name: new_name})
            logger.debug(f"Переименована колонка: {old_name} -> {new_name}")
    
    # 2. Нормализация временных меток
    if 'ts_ns' in df_normalized.columns:
        # Конвертируем в наносекунды если нужно
        if df_normalized['ts_ns'].dtype == 'object':
            try:
                df_normalized['ts_ns'] = pd.to_datetime(df_normalized['ts_ns']).astype(np.int64)
            except:
                logger.warning("Не удалось конвертировать временные метки")
        elif df_normalized['ts_ns'].dtype in ['int64', 'int32']:
            if df_normalized['ts_ns'].max() < 1e15:
                if df_normalized['ts_ns'].max() < 1e12:  # Секунды
                    df_normalized['ts_ns'] = df_normalized['ts_ns'] * 1_000_000_000
                else:  # Миллисекунды
                    df_normalized['ts_ns'] = df_normalized['ts_ns'] * 1_000_000
    
    # 3. Нормализация стороны
    if 'side' in df_normalized.columns:
        df_normalized['side'] = df_normalized['side'].str.lower()
        
        side_mapping = {
            'bid': 'bid',
            'b': 'bid',
            'buy': 'bid',
            'ask': 'ask',
            'a': 'ask',
            'sell': 'ask'
        }
        
        df_normalized['side'] = df_normalized['side'].map(side_mapping).fillna(df_normalized['side'])
    
    # 4. Нормализация типа обновления
    if 'update_type' in df_normalized.columns:
        df_normalized['update_type'] = df_normalized['update_type'].str.lower()
        
        type_mapping = {
            'new': 'new',
            'update': 'update',
            'delete': 'delete',
            'remove': 'delete'
        }
        
        df_normalized['update_type'] = df_normalized['update_type'].map(type_mapping).fillna(df_normalized['update_type'])
    
    # 5. Валидация данных
    if 'price' in df_normalized.columns:
        invalid_prices = (df_normalized['price'] <= 0).sum()
        if invalid_prices > 0:
            logger.warning(f"Удалено {invalid_prices} записей с неположительными ценами")
            df_normalized = df_normalized[df_normalized['price'] > 0]
    
    if 'size' in df_normalized.columns:
        # Для depth размер может быть 0 (удаление уровня)
        negative_sizes = (df_normalized['size'] < 0).sum()
        if negative_sizes > 0:
            logger.warning(f"Удалено {negative_sizes} записей с отрицательными объемами")
            df_normalized = df_normalized[df_normalized['size'] >= 0]
    
    # 6. Сортировка по времени
    if 'ts_ns' in df_normalized.columns:
        df_normalized = df_normalized.sort_values('ts_ns').reset_index(drop=True)
    
    logger.info(f"Нормализовано {len(df_normalized)} depth записей")
    return df_normalized

def validate_normalized_data(
    trades_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    config: Dict = None
) -> Dict[str, float]:
    """
    Валидация нормализованных данных
    
    Args:
        trades_df: Нормализованные trades
        depth_df: Нормализованные depth
        config: Конфигурация
        
    Returns:
        Словарь с метриками качества
    """
    if config is None:
        config = {}
    
    log_progress = config.get('log_progress', True)
    
    if log_progress:
        logger.info("Этап 3/3: Валидация нормализованных данных")
    else:
        logger.info("Валидация нормализованных данных...")
    
    quality_metrics = {}
    
    # 1. Проверка trades
    if not trades_df.empty:
        # Проверяем обязательные колонки
        required_trades_columns = ['ts_ns', 'price', 'size', 'side']
        missing_trades_columns = [col for col in required_trades_columns if col not in trades_df.columns]
        
        if missing_trades_columns:
            logger.warning(f"Отсутствуют колонки в trades: {missing_trades_columns}")
            trades_completeness = (len(required_trades_columns) - len(missing_trades_columns)) / len(required_trades_columns)
        else:
            trades_completeness = 1.0
        
        quality_metrics["trades_completeness"] = trades_completeness
        
        # Проверяем типы данных
        if 'ts_ns' in trades_df.columns:
            is_timestamp_numeric = pd.api.types.is_numeric_dtype(trades_df['ts_ns'])
            quality_metrics["trades_timestamp_valid"] = 1.0 if is_timestamp_numeric else 0.0
        
        # Проверяем логичность данных
        if 'price' in trades_df.columns and 'size' in trades_df.columns:
            valid_prices = (trades_df['price'] > 0).sum()
            valid_sizes = (trades_df['size'] > 0).sum()
            total_trades = len(trades_df)
            
            quality_metrics["trades_price_valid"] = valid_prices / total_trades if total_trades > 0 else 0.0
            quality_metrics["trades_size_valid"] = valid_sizes / total_trades if total_trades > 0 else 0.0
    else:
        quality_metrics["trades_completeness"] = 0.0
        quality_metrics["trades_timestamp_valid"] = 0.0
        quality_metrics["trades_price_valid"] = 0.0
        quality_metrics["trades_size_valid"] = 0.0
    
    # 2. Проверка depth
    if not depth_df.empty:
        # Проверяем обязательные колонки
        required_depth_columns = ['ts_ns', 'price', 'size', 'side']
        missing_depth_columns = [col for col in required_depth_columns if col not in depth_df.columns]
        
        if missing_depth_columns:
            logger.warning(f"Отсутствуют колонки в depth: {missing_depth_columns}")
            depth_completeness = (len(required_depth_columns) - len(missing_depth_columns)) / len(required_depth_columns)
        else:
            depth_completeness = 1.0
        
        quality_metrics["depth_completeness"] = depth_completeness
        
        # Проверяем типы данных
        if 'ts_ns' in depth_df.columns:
            is_timestamp_numeric = pd.api.types.is_numeric_dtype(depth_df['ts_ns'])
            quality_metrics["depth_timestamp_valid"] = 1.0 if is_timestamp_numeric else 0.0
        
        # Проверяем логичность данных
        if 'price' in depth_df.columns and 'size' in depth_df.columns:
            valid_prices = (depth_df['price'] > 0).sum()
            valid_sizes = (depth_df['size'] >= 0).sum()  # Размер может быть 0
            total_depth = len(depth_df)
            
            quality_metrics["depth_price_valid"] = valid_prices / total_depth if total_depth > 0 else 0.0
            quality_metrics["depth_size_valid"] = valid_sizes / total_depth if total_depth > 0 else 0.0
    else:
        quality_metrics["depth_completeness"] = 0.0
        quality_metrics["depth_timestamp_valid"] = 0.0
        quality_metrics["depth_price_valid"] = 0.0
        quality_metrics["depth_size_valid"] = 0.0
    
    # 3. Общий скор качества
    overall_quality = np.mean(list(quality_metrics.values()))
    quality_metrics["overall_quality"] = overall_quality
    
    # Логируем результаты
    logger.info("=== МЕТРИКИ КАЧЕСТВА НОРМАЛИЗАЦИИ ===")
    for metric, score in quality_metrics.items():
        logger.info(f"{metric}: {score:.3f}")
    
    # Цветная оценка качества
    if overall_quality >= 0.9:
        logger.info("🟢 Качество нормализации: ОТЛИЧНОЕ")
    elif overall_quality >= 0.7:
        logger.info("🟡 Качество нормализации: ХОРОШЕЕ")
    elif overall_quality >= 0.5:
        logger.info("🟠 Качество нормализации: УДОВЛЕТВОРИТЕЛЬНОЕ")
    else:
        logger.warning("🔴 Качество нормализации: ПЛОХОЕ")
    
    return quality_metrics

def run_normalization(
    trades_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Основная функция нормализации данных
    
    Args:
        trades_df: Сырые trades данные
        depth_df: Сырые depth данные
        config: Конфигурация
        
    Returns:
        Кортеж (нормализованные trades, нормализованные depth)
    """
    logger.info("Запуск нормализации данных...")
    
    # Проверяем конфигурацию
    norm_config = config.get('normalization', {})
    log_progress = norm_config.get('log_progress', True)
    
    try:
        # 1. Нормализация trades
        normalized_trades = normalize_trades(trades_df, norm_config)
        
        # 2. Нормализация depth
        normalized_depth = normalize_depth(depth_df, norm_config)
        
        # 3. Валидация качества
        quality = validate_normalized_data(normalized_trades, normalized_depth, norm_config)
        
        # 4. Итоговая статистика
        logger.info("=== ИТОГИ НОРМАЛИЗАЦИИ ===")
        logger.info(f"Исходные trades: {len(trades_df):,}")
        logger.info(f"Нормализованные trades: {len(normalized_trades):,}")
        logger.info(f"Исходные depth: {len(depth_df):,}")
        logger.info(f"Нормализованные depth: {len(normalized_depth):,}")
        logger.info(f"Общее качество: {quality.get('overall_quality', 0):.3f}")
        
        # Статистика по биржам
        if 'exchange' in normalized_trades.columns:
            for exchange in normalized_trades['exchange'].unique():
                ex_count = len(normalized_trades[normalized_trades['exchange'] == exchange])
                logger.info(f"  {exchange} trades: {ex_count:,}")
        
        if 'exchange' in normalized_depth.columns:
            for exchange in normalized_depth['exchange'].unique():
                ex_count = len(normalized_depth[normalized_depth['exchange'] == exchange])
                logger.info(f"  {exchange} depth: {ex_count:,}")
        
        logger.info("✅ Нормализация данных завершена успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при нормализации: {e}")
        raise
    
    return normalized_trades, normalized_depth
