"""
Блок 06: Топ-зона книги
Пометить уровни 0…N тиков и считать глубину/изменения size/add/del
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def analyze_top_levels(
    depth_df: pd.DataFrame,
    top_levels: int = 10,
    tick_width: float = 0.01
) -> pd.DataFrame:
    """
    Анализ топ-уровней order book с группировкой по тикам
    
    Args:
        depth_df: Нормализованные depth данные
        top_levels: Количество топ-уровней для анализа
        tick_width: Ширина тика в единицах цены
        
    Returns:
        DataFrame с проанализированными топ-уровнями
    """
    logger.info(f"Анализ топ-{top_levels} уровней order book...")
    
    if depth_df.empty:
        logger.warning("Depth DataFrame пустой, возвращаем пустой результат")
        return pd.DataFrame()
    
    # Создаем копию для модификации
    depth_df = depth_df.copy()
    
    # 1. Группируем по биржам и временным окнам
    book_analysis = []
    
    for exchange in depth_df['exchange'].unique():
        ex_data = depth_df[depth_df['exchange'] == exchange]
        
        # Группируем по временным окнам (10мс)
        tick_size_ns = 10 * 1_000_000  # 10мс в наносекундах
        ex_data['time_bucket'] = (ex_data['ts_ns'] // tick_size_ns) * tick_size_ns
        
        for bucket_time in ex_data['time_bucket'].unique():
            bucket_data = ex_data[ex_data['time_bucket'] == bucket_time]
            
            # Разделяем на bid и ask
            bids = bucket_data[bucket_data['side'] == 'bid']
            asks = bucket_data[bucket_data['side'] == 'ask']
            
            # 2. Анализируем bid уровни (от высшей цены)
            bid_levels = []
            if not bids.empty:
                # Сортируем по цене (убывание) и берем топ-уровни
                sorted_bids = bids.sort_values('price', ascending=False)
                
                for i, (_, row) in enumerate(sorted_bids.head(top_levels).iterrows()):
                    if row['size'] > 0:  # Только активные уровни
                        level_info = {
                            'ts_ns': bucket_time,
                            'exchange': exchange,
                            'side': 'bid',
                            'level': i,  # 0 = best bid, 1 = second best, etc.
                            'price': row['price'],
                            'size': row['size'],
                            'time_bucket': bucket_time,  # Добавляем time_bucket
                            'price_tick': round(row['price'] / tick_width) * tick_width,  # Округление до тика
                            'level_depth': row['size'],  # Глубина на этом уровне
                            'cumulative_depth': sorted_bids.head(i + 1)['size'].sum()  # Накопительная глубина
                        }
                        bid_levels.append(level_info)
            
            # 3. Анализируем ask уровни (от низшей цены)
            ask_levels = []
            if not asks.empty:
                # Сортируем по цене (возрастание) и берем топ-уровни
                sorted_asks = asks.sort_values('price', ascending=True)
                
                for i, (_, row) in enumerate(sorted_asks.head(top_levels).iterrows()):
                    if row['size'] > 0:  # Только активные уровни
                        level_info = {
                            'ts_ns': bucket_time,
                            'exchange': exchange,
                            'side': 'ask',
                            'level': i,  # 0 = best ask, 1 = second best, etc.
                            'price': row['price'],
                            'size': row['size'],
                            'time_bucket': bucket_time,  # Добавляем time_bucket
                            'price_tick': round(row['price'] / tick_width) * tick_width,  # Округление до тика
                            'level_depth': row['size'],  # Глубина на этом уровне
                            'cumulative_depth': sorted_asks.head(i + 1)['size'].sum()  # Накопительная глубина
                        }
                        ask_levels.append(level_info)
            
            # 4. Добавляем все уровни в общий список
            book_analysis.extend(bid_levels)
            book_analysis.extend(ask_levels)
    
    # 5. Создаем DataFrame и сортируем
    if book_analysis:
        book_df = pd.DataFrame(book_analysis)
        book_df = book_df.sort_values(['ts_ns', 'exchange', 'side', 'level']).reset_index(drop=True)
        
        logger.info(f"Проанализировано {len(book_df)} уровней order book")
        logger.info(f"Временной диапазон: {book_df['ts_ns'].min()} - {book_df['ts_ns'].max()}")
        
        # Статистика по биржам
        for exchange in book_df['exchange'].unique():
            ex_data = book_df[book_df['exchange'] == exchange]
            logger.info(f"  {exchange}: {len(ex_data)} уровней")
            
            # Статистика по сторонам
            for side in ['bid', 'ask']:
                side_data = ex_data[ex_data['side'] == side]
                if not side_data.empty:
                    logger.info(f"    {side}: {len(side_data)} уровней, средняя глубина {side_data['level_depth'].mean():.2f}")
    else:
        book_df = pd.DataFrame()
        logger.warning("Не удалось проанализировать уровни order book")
    
    return book_df

def track_changes(
    book_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    min_depth_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Отслеживание изменений в order book (add/del/modify)
    
    Args:
        book_df: Проанализированные уровни order book
        depth_df: Исходные depth данные для сравнения
        min_depth_threshold: Минимальный порог глубины для учета изменений
        
    Returns:
        DataFrame с отслеженными изменениями
    """
    logger.info("Отслеживание изменений в order book...")
    
    if book_df.empty or depth_df.empty:
        logger.warning("Один из DataFrame пустой, возвращаем исходный book_df")
        return book_df
    
    # Создаем копию для модификации
    book_df = book_df.copy()
    
    # 1. Группируем depth данные по времени и биржам для сравнения
    depth_df = depth_df.copy()
    tick_size_ns = 10 * 1_000_000
    depth_df['time_bucket'] = (depth_df['ts_ns'] // tick_size_ns) * tick_size_ns
    
    # 2. Создаем словарь для хранения предыдущего состояния
    previous_state = {}
    
    # 3. Анализируем изменения по временным окнам
    changes_list = []
    
    for exchange in book_df['exchange'].unique():
        ex_book = book_df[book_df['exchange'] == exchange]
        ex_depth = depth_df[depth_df['exchange'] == exchange]
        
        for bucket_time in sorted(ex_book['time_bucket'].unique()):
            current_book = ex_book[ex_book['time_bucket'] == bucket_time]
            current_depth = ex_depth[ex_depth['time_bucket'] == bucket_time]
            
            # Создаем ключ для текущего состояния
            state_key = f"{exchange}_{bucket_time}"
            
            # 4. Анализируем изменения для каждого уровня
            for _, level_row in current_book.iterrows():
                if level_row['level_depth'] < min_depth_threshold:
                    continue
                
                # Ищем соответствующие изменения в depth данных
                level_changes = current_depth[
                    (current_depth['side'] == level_row['side']) &
                    (abs(current_depth['price'] - level_row['price']) < 0.001)  # Примерное совпадение цены
                ]
                
                change_info = {
                    'ts_ns': bucket_time,
                    'exchange': exchange,
                    'side': level_row['side'],
                    'level': level_row['level'],
                    'price': level_row['price'],
                    'current_size': level_row['level_depth'],
                    'price_tick': level_row['price_tick'],
                    'cumulative_depth': level_row['cumulative_depth']
                }
                
                # 5. Анализируем типы изменений
                if not level_changes.empty:
                    # Вычисляем изменения размера
                    size_changes = level_changes['size'].values
                    
                    # Определяем тип изменения
                    if len(size_changes) == 1:
                        if size_changes[0] > 0:
                            change_info['change_type'] = 'add'
                            change_info['size_change'] = size_changes[0]
                        else:
                            change_info['change_type'] = 'delete'
                            change_info['size_change'] = abs(size_changes[0])
                    else:
                        # Множественные изменения - считаем как modify
                        change_info['change_type'] = 'modify'
                        change_info['size_change'] = sum(size_changes)
                    
                    # Вычисляем скорость изменения (size/время)
                    if len(level_changes) > 1:
                        time_range = level_changes['ts_ns'].max() - level_changes['ts_ns'].min()
                        if time_range > 0:
                            change_info['change_speed'] = change_info['size_change'] / (time_range / 1_000_000_000)  # size/сек
                        else:
                            change_info['change_speed'] = 0.0
                    else:
                        change_info['change_speed'] = 0.0
                    
                    # Добавляем информацию о delta
                    change_info['delta_count'] = len(level_changes)
                    change_info['delta_sizes'] = size_changes.tolist()
                else:
                    # Нет изменений в этом временном окне
                    change_info['change_type'] = 'stable'
                    change_info['size_change'] = 0.0
                    change_info['change_speed'] = 0.0
                    change_info['delta_count'] = 0
                    change_info['delta_sizes'] = []
                
                changes_list.append(change_info)
    
    # 6. Создаем DataFrame с изменениями
    if changes_list:
        changes_df = pd.DataFrame(changes_list)
        changes_df = changes_df.sort_values(['ts_ns', 'exchange', 'side', 'level']).reset_index(drop=True)
        
        logger.info(f"Отслежено {len(changes_df)} изменений в order book")
        
        # Статистика по типам изменений
        change_types = changes_df['change_type'].value_counts()
        logger.info("Распределение типов изменений:")
        for change_type, count in change_types.items():
            logger.info(f"  {change_type}: {count}")
        
        # Статистика по биржам
        for exchange in changes_df['exchange'].unique():
            ex_changes = changes_df[changes_df['exchange'] == exchange]
            logger.info(f"  {exchange}: {len(ex_changes)} изменений")
            
            # Средняя скорость изменений
            avg_speed = ex_changes['change_speed'].mean()
            logger.info(f"    Средняя скорость изменений: {avg_speed:.2f} size/сек")
    else:
        changes_df = pd.DataFrame()
        logger.warning("Не удалось отследить изменения")
    
    return changes_df

def calculate_book_metrics(
    book_df: pd.DataFrame,
    changes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Вычисление дополнительных метрик order book
    
    Args:
        book_df: Проанализированные уровни order book
        changes_df: Отслеженные изменения
        
    Returns:
        DataFrame с дополнительными метриками
    """
    logger.info("Вычисление метрик order book...")
    
    if book_df.empty:
        return book_df
    
    # Создаем копию для модификации
    book_df = book_df.copy()
    
    # 1. Вычисляем метрики по уровням
    book_metrics = []
    
    for exchange in book_df['exchange'].unique():
        ex_book = book_df[book_df['exchange'] == exchange]
        
        for bucket_time in ex_book['time_bucket'].unique():
            bucket_book = ex_book[ex_book['time_bucket'] == bucket_time]
            
            # Разделяем на bid и ask
            bids = bucket_book[bucket_book['side'] == 'bid']
            asks = bucket_book[bucket_book['side'] == 'ask']
            
            metrics = {
                'ts_ns': bucket_time,
                'exchange': exchange
            }
            
            # 2. Метрики bid стороны
            if not bids.empty:
                metrics['bid_levels_count'] = len(bids)
                metrics['bid_total_depth'] = bids['level_depth'].sum()
                metrics['bid_avg_depth'] = bids['level_depth'].mean()
                metrics['bid_max_depth'] = bids['level_depth'].max()
                metrics['bid_depth_std'] = bids['level_depth'].std()
                
                # Глубина по уровням
                for i in range(min(5, len(bids))):  # Первые 5 уровней
                    level_data = bids[bids['level'] == i]
                    if not level_data.empty:
                        metrics[f'bid_level_{i}_depth'] = level_data.iloc[0]['level_depth']
                        metrics[f'bid_level_{i}_price'] = level_data.iloc[0]['price']
                    else:
                        metrics[f'bid_level_{i}_depth'] = 0.0
                        metrics[f'bid_level_{i}_price'] = 0.0
            else:
                # Заполняем нулями если нет bid данных
                metrics.update({
                    'bid_levels_count': 0, 'bid_total_depth': 0.0, 'bid_avg_depth': 0.0,
                    'bid_max_depth': 0.0, 'bid_depth_std': 0.0
                })
                for i in range(5):
                    metrics[f'bid_level_{i}_depth'] = 0.0
                    metrics[f'bid_level_{i}_price'] = 0.0
            
            # 3. Метрики ask стороны
            if not asks.empty:
                metrics['ask_levels_count'] = len(asks)
                metrics['ask_total_depth'] = asks['level_depth'].sum()
                metrics['ask_avg_depth'] = asks['level_depth'].mean()
                metrics['ask_max_depth'] = asks['level_depth'].max()
                metrics['ask_depth_std'] = asks['level_depth'].std()
                
                # Глубина по уровням
                for i in range(min(5, len(asks))):  # Первые 5 уровней
                    level_data = asks[asks['level'] == i]
                    if not level_data.empty:
                        metrics[f'ask_level_{i}_depth'] = level_data.iloc[0]['level_depth']
                        metrics[f'ask_level_{i}_price'] = level_data.iloc[0]['price']
                    else:
                        metrics[f'ask_level_{i}_depth'] = 0.0
                        metrics[f'ask_level_{i}_price'] = 0.0
            else:
                # Заполняем нулями если нет ask данных
                metrics.update({
                    'ask_levels_count': 0, 'ask_total_depth': 0.0, 'ask_avg_depth': 0.0,
                    'ask_max_depth': 0.0, 'ask_depth_std': 0.0
                })
                for i in range(5):
                    metrics[f'ask_level_{i}_depth'] = 0.0
                    metrics[f'ask_level_{i}_price'] = 0.0
            
            # 4. Сводные метрики
            if not bids.empty and not asks.empty:
                # Лучшие цены
                best_bid = bids[bids['level'] == 0]['price'].iloc[0] if not bids[bids['level'] == 0].empty else 0.0
                best_ask = asks[asks['level'] == 0]['price'].iloc[0] if not asks[asks['level'] == 0].empty else 0.0
                
                if best_bid > 0 and best_ask > 0:
                    metrics['spread'] = best_ask - best_bid
                    metrics['spread_rel'] = (best_ask - best_bid) / best_bid * 100
                    metrics['mid_price'] = (best_bid + best_ask) / 2
                else:
                    metrics['spread'] = 0.0
                    metrics['spread_rel'] = 0.0
                    metrics['mid_price'] = 0.0
                
                # Общая глубина
                metrics['total_book_depth'] = metrics['bid_total_depth'] + metrics['ask_total_depth']
                metrics['depth_imbalance'] = abs(metrics['bid_total_depth'] - metrics['ask_total_depth']) / metrics['total_book_depth']
            else:
                metrics.update({
                    'spread': 0.0, 'spread_rel': 0.0, 'mid_price': 0.0,
                    'total_book_depth': 0.0, 'depth_imbalance': 0.0
                })
            
            book_metrics.append(metrics)
    
    # 5. Создаем DataFrame с метриками
    if book_metrics:
        metrics_df = pd.DataFrame(book_metrics)
        metrics_df = metrics_df.sort_values(['ts_ns', 'exchange']).reset_index(drop=True)
        
        logger.info(f"Вычислено {len(metrics_df)} метрик order book")
        
        # Статистика по биржам
        for exchange in metrics_df['exchange'].unique():
            ex_metrics = metrics_df[metrics_df['exchange'] == exchange]
            logger.info(f"  {exchange}: {len(ex_metrics)} метрик")
            
            if 'spread' in ex_metrics.columns:
                avg_spread = ex_metrics['spread'].mean()
                logger.info(f"    Средний спред: {avg_spread:.4f}")
    else:
        metrics_df = pd.DataFrame()
        logger.warning("Не удалось вычислить метрики")
    
    return metrics_df

def save_book_top(
    book_df: pd.DataFrame,
    changes_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_path: Path,
    config: Dict
) -> None:
    """
    Сохранение результатов анализа топ-зоны order book
    
    Args:
        book_df: Проанализированные уровни
        changes_df: Отслеженные изменения
        metrics_df: Вычисленные метрики
        output_path: Путь для сохранения
        config: Конфигурация
    """
    logger.info(f"Сохранение результатов анализа order book в {output_path}...")
    
    try:
        # 1. Создать директорию если не существует
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 2. Получаем настройки сжатия из конфигурации
        compression = config.get("export", {}).get("compression", "snappy")
        
        # 3. Сохраняем основные уровни
        if not book_df.empty:
            book_path = output_path / "book_levels.parquet"
            book_df.to_parquet(
                book_path,
                engine="pyarrow",
                compression=compression,
                index=False
            )
            logger.info(f"Уровни order book сохранены: {len(book_df)} строк")
        
        # 4. Сохраняем изменения
        if not changes_df.empty:
            changes_path = output_path / "book_changes.parquet"
            changes_df.to_parquet(
                changes_path,
                engine="pyarrow",
                compression=compression,
                index=False
            )
            logger.info(f"Изменения order book сохранены: {len(changes_df)} строк")
        
        # 5. Сохраняем метрики
        if not metrics_df.empty:
            metrics_path = output_path / "book_metrics.parquet"
            metrics_df.to_parquet(
                metrics_path,
                engine="pyarrow",
                compression=compression,
                index=False
            )
            logger.info(f"Метрики order book сохранены: {len(metrics_df)} строк")
        
        # 6. Логируем информацию о сохранении
        logger.info(f"Результаты анализа order book сохранены в {output_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении результатов анализа order book: {e}")
        raise

def run_book_top(
    depth_df: pd.DataFrame,
    config: Dict,
    output_dir: Path = Path("data/book_top")
) -> Dict[str, pd.DataFrame]:
    """
    Основная функция анализа топ-зоны order book
    
    Args:
        depth_df: Нормализованные depth данные
        config: Конфигурация
        output_dir: Директория для сохранения
        
    Returns:
        Словарь с результатами анализа
    """
    logger.info("Запуск анализа топ-зоны order book...")
    
    try:
        # 1. Получаем параметры из конфигурации
        top_levels = config.get("order_book", {}).get("top_levels", 10)
        tick_width = config.get("order_book", {}).get("tick_width", 0.01)
        min_depth_threshold = config.get("order_book", {}).get("depth_threshold", 0.0)
        
        logger.info(f"Параметры анализа: top_levels={top_levels}, tick_width={tick_width}")
        
        # 2. Анализируем топ-уровни
        logger.info("Анализ топ-уровней order book...")
        book_df = analyze_top_levels(depth_df, top_levels, tick_width)
        
        if book_df.empty:
            logger.warning("Не удалось проанализировать уровни order book, завершаем работу")
            return {}
        
        # 3. Отслеживаем изменения
        logger.info("Отслеживание изменений в order book...")
        changes_df = track_changes(book_df, depth_df, min_depth_threshold)
        
        # 4. Вычисляем метрики
        logger.info("Вычисление метрик order book...")
        metrics_df = calculate_book_metrics(book_df, changes_df)
        
        # 5. Сохраняем результаты
        logger.info("Сохранение результатов анализа...")
        save_book_top(book_df, changes_df, metrics_df, output_dir, config)
        
        # 6. Итоговая статистика
        logger.info("=== ИТОГИ АНАЛИЗА ORDER BOOK ===")
        logger.info(f"Исходные depth записи: {len(depth_df):,}")
        logger.info(f"Проанализировано уровней: {len(book_df):,}")
        
        if not changes_df.empty:
            logger.info(f"Отслежено изменений: {len(changes_df):,}")
            
            # Статистика по типам изменений
            change_types = changes_df['change_type'].value_counts()
            for change_type, count in change_types.items():
                logger.info(f"  {change_type}: {count}")
        
        if not metrics_df.empty:
            logger.info(f"Вычислено метрик: {len(metrics_df):,}")
        
        # Статистика по биржам
        for exchange in book_df['exchange'].unique():
            ex_levels = book_df[book_df['exchange'] == exchange]
            logger.info(f"  {exchange}: {len(ex_levels)} уровней")
        
        logger.info("✅ Анализ топ-зоны order book завершен успешно!")
        
        # 7. Возвращаем результаты
        results = {
            'book_levels': book_df,
            'book_changes': changes_df,
            'book_metrics': metrics_df
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка при анализе order book: {e}")
        raise
    
    return results
