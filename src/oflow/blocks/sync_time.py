"""
Блок 04: Синхронизация времени
Оценить и сдвинуть лаги по биржам/источникам и отфильтровать дропы
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def estimate_exchange_lags(trades_df: pd.DataFrame, depth_df: pd.DataFrame, config: Dict = None) -> Dict[str, int]:
    """
    Оценить лаги между биржами на основе trades данных
    
    Args:
        trades_df: Нормализованные trades данные
        depth_df: Нормализованные depth данные
        config: Конфигурация синхронизации
        
    Returns:
        Словарь с лагами для каждой биржи (в наносекундах)
    """
    if config is None:
        config = {}
    
    logger.info("Оценка лагов между биржами...")
    logger.info("Этап 1/4: Анализ данных по биржам")
    
    if trades_df.empty:
        logger.warning("Trades данные пусты, возвращаем нулевые лаги")
        return {"binance": 0, "bybit": 0, "okx": 0}
    
    # 1. Получаем уникальные биржи
    exchanges = trades_df['exchange'].unique()
    logger.info(f"Найдены биржи: {exchanges}")
    
    # Применяем конфигурацию
    reference_exchange = config.get('reference_exchange', 'binance')
    max_lag_seconds = config.get('max_lag_seconds', 10)
    correlation_threshold = config.get('correlation_threshold', 0.7)
    time_bucket_ms = config.get('time_bucket_ms', 1000)
    min_data_points = config.get('min_data_points', 50)
    
    logger.info(f"Референсная биржа: {reference_exchange}")
    logger.info(f"Максимальный лаг: {max_lag_seconds} сек")
    logger.info(f"Порог корреляции: {correlation_threshold}")
    
    # 2. Создаем агрегированные временные ряды цен
    time_window_ns = time_bucket_ms * 1_000_000  # Конвертируем в наносекунды
    logger.info("Этап 2/4: Создание временных рядов цен")
    
    # Группируем данные по биржам и временным окнам
    price_series = {}
    
    for exchange in exchanges:
        ex_data = trades_df[trades_df['exchange'] == exchange].copy()
        if ex_data.empty:
            continue
            
        # Создаем временные окна (округляем до секунд)
        ex_data['time_bucket'] = (ex_data['ts_ns'] // time_window_ns) * time_window_ns
        
        # Агрегируем средневзвешенную цену по объему в каждом окне
        bucket_data = ex_data.groupby('time_bucket').agg({
            'price': 'mean',  # Средняя цена
            'size': 'sum'     # Общий объем
        }).reset_index()
        
        price_series[exchange] = bucket_data.set_index('time_bucket')['price']
        logger.info(f"{exchange}: {len(bucket_data)} временных окон")
    
    # 3. Находим пересечение временных окон между биржами
    if len(price_series) < 2:
        logger.warning("Недостаточно бирж для оценки лагов")
        return {ex: 0 for ex in exchanges}
    
    # Используем указанную референсную биржу
    if reference_exchange not in price_series:
        reference_exchange = list(price_series.keys())[0]
        logger.warning(f"Референсная биржа {reference_exchange} не найдена, используем {reference_exchange}")
    
    reference_series = price_series[reference_exchange]
    lags = {reference_exchange: 0}  # Референсная биржа имеет лаг 0
    
    logger.info("Этап 3/4: Анализ корреляций между биржами")
    
    # 4. Для каждой биржи вычисляем кросс-корреляцию с референсной
    for exchange in exchanges:
        if exchange == reference_exchange:
            continue
            
        target_series = price_series[exchange]
        
        # Находим общие временные окна
        common_times = reference_series.index.intersection(target_series.index)
        
        if len(common_times) < min_data_points:  # Используем конфигурацию
            logger.warning(f"Недостаточно общих точек для {exchange}: {len(common_times)} < {min_data_points}")
            lags[exchange] = 0
            continue
        
        ref_prices = reference_series.loc[common_times]
        target_prices = target_series.loc[common_times]
        
        # Вычисляем кросс-корреляцию для определения лага
        # Ищем лаг в диапазоне ±max_lag_seconds секунд
        max_lag_buckets = max_lag_seconds
        correlations = []
        
        for lag in range(-max_lag_buckets, max_lag_buckets + 1):
            if lag == 0:
                corr = np.corrcoef(ref_prices, target_prices)[0, 1]
            elif lag > 0:
                # Позитивный лаг: target отстает от reference
                if len(ref_prices) > lag:
                    corr = np.corrcoef(ref_prices[:-lag], target_prices[lag:])[0, 1]
                else:
                    corr = 0
            else:
                # Негативный лаг: target опережает reference
                lag_abs = abs(lag)
                if len(target_prices) > lag_abs:
                    corr = np.corrcoef(ref_prices[lag_abs:], target_prices[:-lag_abs])[0, 1]
                else:
                    corr = 0
            
            if not np.isnan(corr):
                correlations.append((lag, corr))
        
        # Находим лаг с максимальной корреляцией
        if correlations:
            best_lag, best_corr = max(correlations, key=lambda x: x[1])
            lag_ns = best_lag * time_window_ns  # Переводим в наносекунды
            
            logger.info(f"{exchange}: лаг = {best_lag}с ({lag_ns}нс), корреляция = {best_corr:.3f}")
            lags[exchange] = lag_ns
        else:
            logger.warning(f"Не удалось вычислить лаг для {exchange}")
            lags[exchange] = 0
    
    # 5. Валидация лагов (проверяем разумность)
    for exchange, lag in lags.items():
        if abs(lag) > 30_000_000_000:  # Более 30 секунд - подозрительно
            logger.warning(f"Подозрительно большой лаг для {exchange}: {lag/1e9:.1f}с, сброс в 0")
            lags[exchange] = 0
    
    logger.info(f"Итоговые лаги: {lags}")
    return lags

def sync_timestamps(
    trades_df: pd.DataFrame, 
    depth_df: pd.DataFrame,
    lags: Dict[str, int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Синхронизировать таймстемпы между биржами
    
    Args:
        trades_df: DataFrame с trades
        depth_df: DataFrame с depth
        lags: Словарь лагов по биржам
        
    Returns:
        Кортеж (синхронизированные trades, синхронизированные depth)
    """
    logger.info("Синхронизация таймстемпов...")
    
    # Применяем лаги к trades данным
    synced_trades = trades_df.copy() if not trades_df.empty else trades_df
    synced_depth = depth_df.copy() if not depth_df.empty else depth_df
    
    if not synced_trades.empty:
        logger.info("Применение лагов к trades данным...")
        for exchange, lag_ns in lags.items():
            if lag_ns == 0:
                continue
                
            # Находим строки для данной биржи
            exchange_mask = synced_trades['exchange'] == exchange
            if exchange_mask.any():
                # Применяем лаг (вычитаем лаг из времени для компенсации)
                synced_trades.loc[exchange_mask, 'ts_ns'] -= lag_ns
                logger.info(f"Применен лаг {lag_ns/1e6:.1f}мс к {exchange_mask.sum()} trades от {exchange}")
        
        # Пересортировываем по времени после применения лагов
        synced_trades = synced_trades.sort_values('ts_ns').reset_index(drop=True)
        logger.info("Trades данные пересортированы по синхронизированному времени")
    
    if not synced_depth.empty:
        logger.info("Применение лагов к depth данным...")
        for exchange, lag_ns in lags.items():
            if lag_ns == 0:
                continue
                
            # Находим строки для данной биржи
            exchange_mask = synced_depth['exchange'] == exchange
            if exchange_mask.any():
                # Применяем лаг (вычитаем лаг из времени для компенсации)
                synced_depth.loc[exchange_mask, 'ts_ns'] -= lag_ns
                logger.info(f"Применен лаг {lag_ns/1e6:.1f}мс к {exchange_mask.sum()} depth от {exchange}")
        
        # Пересортировываем по времени после применения лагов
        synced_depth = synced_depth.sort_values('ts_ns').reset_index(drop=True)
        logger.info("Depth данные пересортированы по синхронизированному времени")
    
    # Проверяем результат синхронизации
    total_lag_applied = sum(abs(lag) for lag in lags.values() if lag != 0)
    if total_lag_applied > 0:
        logger.info(f"Синхронизация завершена, применен общий лаг {total_lag_applied/1e6:.1f}мс")
        
        # Логируем новые временные диапазоны
        if not synced_trades.empty:
            logger.info(f"Новый диапазон trades: {synced_trades.ts_ns.min()} - {synced_trades.ts_ns.max()}")
        if not synced_depth.empty:
            logger.info(f"Новый диапазон depth: {synced_depth.ts_ns.min()} - {synced_depth.ts_ns.max()}")
    else:
        logger.info("Лаги не обнаружены, синхронизация не требуется")
    
    return synced_trades, synced_depth

def filter_drops(
    trades_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    min_overlap_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Отфильтровать дропы данных и неполные периоды
    
    Args:
        trades_df: DataFrame с trades
        depth_df: DataFrame с depth
        min_overlap_ratio: Минимальное перекрытие данных
        
    Returns:
        Кортеж (очищенные trades, очищенные depth)
    """
    logger.info("Фильтрация дропов данных...")
    
    filtered_trades = trades_df.copy() if not trades_df.empty else trades_df
    filtered_depth = depth_df.copy() if not depth_df.empty else depth_df
    
    if trades_df.empty and depth_df.empty:
        logger.warning("Оба DataFrame пусты, фильтрация не требуется")
        return filtered_trades, filtered_depth
    
    # 1. Определяем общий временной диапазон
    all_timestamps = []
    
    if not trades_df.empty:
        all_timestamps.extend(trades_df['ts_ns'].tolist())
    if not depth_df.empty:
        all_timestamps.extend(depth_df['ts_ns'].tolist())
    
    if not all_timestamps:
        logger.warning("Нет временных меток для анализа")
        return filtered_trades, filtered_depth
    
    global_start = min(all_timestamps)
    global_end = max(all_timestamps)
    total_duration = global_end - global_start
    
    logger.info(f"Общий временной диапазон: {total_duration/1e9:.1f} секунд")
    
    # 2. Анализируем покрытие данных по биржам
    time_window_ns = 10_000_000_000  # 10 секунд - окно для анализа покрытия
    exchanges = set()
    
    if not trades_df.empty:
        exchanges.update(trades_df['exchange'].unique())
    if not depth_df.empty:
        exchanges.update(depth_df['exchange'].unique())
    
    logger.info(f"Анализируем покрытие для бирж: {exchanges}")
    
    # 3. Создаем временную сетку для анализа покрытия
    time_buckets = np.arange(global_start, global_end + time_window_ns, time_window_ns)
    coverage_data = []
    
    for i in range(len(time_buckets) - 1):
        bucket_start = time_buckets[i]
        bucket_end = time_buckets[i + 1]
        
        # Считаем количество бирж с данными в этом окне
        exchanges_with_data = set()
        
        # Проверяем trades
        if not trades_df.empty:
            trades_in_bucket = trades_df[
                (trades_df['ts_ns'] >= bucket_start) & 
                (trades_df['ts_ns'] < bucket_end)
            ]
            if not trades_in_bucket.empty:
                exchanges_with_data.update(trades_in_bucket['exchange'].unique())
        
        # Проверяем depth
        if not depth_df.empty:
            depth_in_bucket = depth_df[
                (depth_df['ts_ns'] >= bucket_start) & 
                (depth_df['ts_ns'] < bucket_end)
            ]
            if not depth_in_bucket.empty:
                exchanges_with_data.update(depth_in_bucket['exchange'].unique())
        
        # Вычисляем коэффициент покрытия
        coverage_ratio = len(exchanges_with_data) / len(exchanges) if exchanges else 0
        coverage_data.append({
            'start': bucket_start,
            'end': bucket_end,
            'coverage': coverage_ratio,
            'exchanges': exchanges_with_data
        })
    
    # 4. Находим периоды с хорошим покрытием
    good_periods = []
    current_period_start = None
    
    for bucket in coverage_data:
        if bucket['coverage'] >= min_overlap_ratio:
            if current_period_start is None:
                current_period_start = bucket['start']
        else:
            if current_period_start is not None:
                # Завершаем текущий хороший период
                good_periods.append((current_period_start, bucket['start']))
                current_period_start = None
    
    # Закрываем последний период если нужно
    if current_period_start is not None:
        good_periods.append((current_period_start, global_end))
    
    logger.info(f"Найдено {len(good_periods)} периодов с покрытием >= {min_overlap_ratio:.1%}")
    
    # 5. Фильтруем данные по хорошим периодам
    if good_periods:
        # Создаем маску для хороших периодов
        def is_in_good_period(timestamp):
            return any(start <= timestamp < end for start, end in good_periods)
        
        if not trades_df.empty:
            good_trades_mask = trades_df['ts_ns'].apply(is_in_good_period)
            filtered_trades = trades_df[good_trades_mask].reset_index(drop=True)
            
            removed_trades = len(trades_df) - len(filtered_trades)
            if removed_trades > 0:
                logger.info(f"Удалено {removed_trades} trades ({removed_trades/len(trades_df)*100:.1f}%)")
            
        if not depth_df.empty:
            good_depth_mask = depth_df['ts_ns'].apply(is_in_good_period)
            filtered_depth = depth_df[good_depth_mask].reset_index(drop=True)
            
            removed_depth = len(depth_df) - len(filtered_depth)
            if removed_depth > 0:
                logger.info(f"Удалено {removed_depth} depth records ({removed_depth/len(depth_df)*100:.1f}%)")
        
        # Вычисляем итоговое покрытие
        total_good_time = sum(end - start for start, end in good_periods)
        final_coverage = total_good_time / total_duration
        logger.info(f"Итоговое покрытие времени: {final_coverage:.1%}")
        
    else:
        logger.warning("Не найдено периодов с достаточным покрытием, данные не изменены")
    
    # 6. Дополнительная фильтрация: удаляем одиночные точки
    if not filtered_trades.empty:
        # Удаляем trades, которые стоят изолированно (больше 60 секунд от ближайших)
        isolation_threshold = 60_000_000_000  # 60 секунд
        
        filtered_trades = filtered_trades.sort_values('ts_ns').reset_index(drop=True)
        to_keep = [True] * len(filtered_trades)
        
        for i in range(len(filtered_trades)):
            current_time = filtered_trades.iloc[i]['ts_ns']
            
            # Проверяем расстояние до ближайших точек
            prev_distance = float('inf')
            next_distance = float('inf')
            
            if i > 0:
                prev_distance = current_time - filtered_trades.iloc[i-1]['ts_ns']
            if i < len(filtered_trades) - 1:
                next_distance = filtered_trades.iloc[i+1]['ts_ns'] - current_time
            
            # Если точка слишком изолирована, помечаем для удаления
            if min(prev_distance, next_distance) > isolation_threshold:
                to_keep[i] = False
        
        isolated_count = sum(not keep for keep in to_keep)
        if isolated_count > 0:
            filtered_trades = filtered_trades[to_keep].reset_index(drop=True)
            logger.info(f"Удалено {isolated_count} изолированных trades")
    
    logger.info("Фильтрация дропов завершена")
    return filtered_trades, filtered_depth

def validate_sync_quality(
    trades_df: pd.DataFrame,
    depth_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Проверить качество синхронизации
    
    Args:
        trades_df: Синхронизированные trades
        depth_df: Синхронизированные depth
        
    Returns:
        Словарь с метриками качества
    """
    logger.info("Проверка качества синхронизации...")
    
    quality_metrics = {}
    
    # 1. Анализ покрытия trades данных
    if not trades_df.empty:
        exchanges = trades_df['exchange'].unique()
        
        # Вычисляем временное покрытие по биржам
        total_time_range = trades_df['ts_ns'].max() - trades_df['ts_ns'].min()
        
        # Анализируем равномерность распределения по биржам
        exchange_coverage = {}
        for exchange in exchanges:
            ex_data = trades_df[trades_df['exchange'] == exchange]
            ex_time_range = ex_data['ts_ns'].max() - ex_data['ts_ns'].min()
            coverage = ex_time_range / total_time_range if total_time_range > 0 else 0
            exchange_coverage[exchange] = coverage
        
        # Средняя покрытие по биржам
        avg_trades_coverage = np.mean(list(exchange_coverage.values()))
        quality_metrics["trades_coverage"] = min(avg_trades_coverage, 1.0)
        
        # Равномерность распределения (стандартное отклонение от равного распределения)
        expected_per_exchange = 1.0 / len(exchanges)
        actual_distribution = [len(trades_df[trades_df['exchange'] == ex]) / len(trades_df) 
                             for ex in exchanges]
        distribution_variance = np.var(actual_distribution)
        distribution_score = max(0, 1.0 - distribution_variance * 10)  # Штраф за неравномерность
        quality_metrics["trades_distribution"] = distribution_score
        
        logger.info(f"Trades покрытие по биржам: {exchange_coverage}")
    else:
        quality_metrics["trades_coverage"] = 0.0
        quality_metrics["trades_distribution"] = 0.0
    
    # 2. Анализ покрытия depth данных
    if not depth_df.empty:
        exchanges = depth_df['exchange'].unique()
        
        total_time_range = depth_df['ts_ns'].max() - depth_df['ts_ns'].min()
        
        exchange_coverage = {}
        for exchange in exchanges:
            ex_data = depth_df[depth_df['exchange'] == exchange]
            ex_time_range = ex_data['ts_ns'].max() - ex_data['ts_ns'].min()
            coverage = ex_time_range / total_time_range if total_time_range > 0 else 0
            exchange_coverage[exchange] = coverage
        
        avg_depth_coverage = np.mean(list(exchange_coverage.values()))
        quality_metrics["depth_coverage"] = min(avg_depth_coverage, 1.0)
        
        logger.info(f"Depth покрытие по биржам: {exchange_coverage}")
    else:
        quality_metrics["depth_coverage"] = 0.0
    
    # 3. Анализ временного выравнивания между биржами
    if not trades_df.empty and len(trades_df['exchange'].unique()) > 1:
        # Анализируем синхронность по временным окнам
        window_size = 1_000_000_000  # 1 секунда
        
        # Создаем временные окна
        min_time = trades_df['ts_ns'].min()
        max_time = trades_df['ts_ns'].max()
        
        time_buckets = np.arange(min_time, max_time + window_size, window_size)
        alignment_scores = []
        
        for i in range(len(time_buckets) - 1):
            bucket_start = time_buckets[i]
            bucket_end = time_buckets[i + 1]
            
            # Данные в текущем окне
            window_data = trades_df[
                (trades_df['ts_ns'] >= bucket_start) & 
                (trades_df['ts_ns'] < bucket_end)
            ]
            
            if len(window_data) > 0:
                # Считаем количество бирж в окне
                exchanges_in_window = len(window_data['exchange'].unique())
                total_exchanges = len(trades_df['exchange'].unique())
                
                # Скор выравнивания = доля бирж присутствующих в окне
                alignment_score = exchanges_in_window / total_exchanges
                alignment_scores.append(alignment_score)
        
        # Средний скор выравнивания
        avg_alignment = np.mean(alignment_scores) if alignment_scores else 0
        quality_metrics["time_alignment"] = avg_alignment
        
        logger.info(f"Анализировано {len(alignment_scores)} временных окон")
        logger.info(f"Средний скор выравнивания: {avg_alignment:.3f}")
    else:
        quality_metrics["time_alignment"] = 1.0  # Одна биржа всегда "выровнена"
    
    # 4. Анализ целостности данных
    integrity_score = 1.0
    
    # Проверка на дубликаты временных меток в пределах биржи
    if not trades_df.empty:
        for exchange in trades_df['exchange'].unique():
            ex_data = trades_df[trades_df['exchange'] == exchange]
            duplicates = ex_data['ts_ns'].duplicated().sum()
            if duplicates > 0:
                logger.warning(f"Найдены дублирующиеся timestamps в {exchange}: {duplicates}")
                integrity_score *= (1.0 - duplicates / len(ex_data))
    
    # Проверка на монотонность после сортировки
    if not trades_df.empty:
        is_sorted = trades_df['ts_ns'].is_monotonic_increasing
        if not is_sorted:
            logger.warning("Данные не отсортированы по времени")
            integrity_score *= 0.5
    
    # Проверка на разумные временные интервалы
    if not trades_df.empty and len(trades_df) > 1:
        time_diffs = trades_df['ts_ns'].diff().dropna()
        negative_diffs = (time_diffs < 0).sum()
        if negative_diffs > 0:
            logger.warning(f"Обнаружены отрицательные временные интервалы: {negative_diffs}")
            integrity_score *= (1.0 - negative_diffs / len(time_diffs))
    
    quality_metrics["data_integrity"] = max(0.0, integrity_score)
    
    # 5. Общий скор качества
    overall_score = np.mean(list(quality_metrics.values()))
    quality_metrics["overall_quality"] = overall_score
    
    # Логируем результаты
    logger.info("=== МЕТРИКИ КАЧЕСТВА СИНХРОНИЗАЦИИ ===")
    for metric, score in quality_metrics.items():
        logger.info(f"{metric}: {score:.3f}")
    
    if overall_score >= 0.9:
        logger.info("🟢 Качество синхронизации: ОТЛИЧНОЕ")
    elif overall_score >= 0.7:
        logger.info("🟡 Качество синхронизации: ХОРОШЕЕ")
    elif overall_score >= 0.5:
        logger.info("🟠 Качество синхронизации: УДОВЛЕТВОРИТЕЛЬНОЕ")
    else:
        logger.warning("🔴 Качество синхронизации: ПЛОХОЕ")
    
    return quality_metrics

def run_time_sync(
    trades_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Основная функция синхронизации времени
    
    Args:
        trades_df: Нормализованные trades
        depth_df: Нормализованные depth
        config: Конфигурация
        
    Returns:
        Кортеж (синхронизированные trades, синхронизированные depth)
    """
    logger.info("Запуск синхронизации времени...")
    
    # Проверяем конфигурацию
    sync_config = config.get('sync_time', {})
    log_progress = sync_config.get('log_progress', True)
    
    if log_progress:
        logger.info("Этап 1/4: Оценка лагов между биржами")
    
    # 1. Оценить лаги
    lags = estimate_exchange_lags(trades_df, depth_df, sync_config)
    
    if log_progress:
        logger.info("Этап 2/4: Синхронизация временных меток")
    
    # 2. Синхронизировать таймстемпы
    synced_trades, synced_depth = sync_timestamps(trades_df, depth_df, lags)
    
    if log_progress:
        logger.info("Этап 3/4: Фильтрация дропов данных")
    
    # 3. Отфильтровать дропы
    filtered_trades, filtered_depth = filter_drops(
        synced_trades, synced_depth, 
        sync_config.get("min_data_coverage", 0.8)
    )
    
    if log_progress:
        logger.info("Этап 4/4: Валидация качества синхронизации")
    
    # 4. Проверить качество
    quality = validate_sync_quality(filtered_trades, filtered_depth)
    
    logger.info("Синхронизация времени завершена")
    return filtered_trades, filtered_depth
