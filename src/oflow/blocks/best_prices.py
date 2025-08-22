"""
Блок 05: Лучшие цены
Восстановить best bid/ask/mid/спред с шагом 10 мс
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def restore_quotes(
    depth_df: pd.DataFrame,
    tick_size_ms: int = 10,
    config: Dict = None
) -> pd.DataFrame:
    """
    Восстановить best bid/ask/mid цены из depth данных
    
    Args:
        depth_df: Нормализованные depth данные
        tick_size_ms: Шаг агрегации в миллисекундах
        config: Конфигурация восстановления цен
        
    Returns:
        DataFrame с quotes (ts_ns, exchange, best_bid, best_ask, mid, spread)
    """
    if config is None:
        config = {}
    
    log_progress = config.get('log_progress', True)
    
    if log_progress:
        logger.info("Этап 1/4: Восстановление best bid/ask/mid цен")
    else:
        logger.info("Восстановление best bid/ask/mid цен...")
    
    if depth_df.empty:
        logger.warning("Depth DataFrame пустой, возвращаем пустой результат")
        return pd.DataFrame(columns=[
            'ts_ns', 'exchange', 'best_bid', 'best_ask', 'mid', 'spread'
        ])
    
    # 1. Создаем временные окна для агрегации
    tick_size_ns = tick_size_ms * 1_000_000  # переводим в наносекунды
    
    # Округляем timestamps до границ тиков
    depth_df = depth_df.copy()
    depth_df['time_bucket'] = (depth_df['ts_ns'] // tick_size_ns) * tick_size_ns
    
    logger.info(f"Агрегация по тикам {tick_size_ms}ms ({tick_size_ns}нс)")
    
    # 2. Группируем данные по временным окнам и биржам
    quotes_list = []
    
    for exchange in depth_df['exchange'].unique():
        ex_data = depth_df[depth_df['exchange'] == exchange]
        
        # Группируем по временным окнам
        for bucket_time in ex_data['time_bucket'].unique():
            bucket_data = ex_data[ex_data['time_bucket'] == bucket_time]
            
            # Разделяем на bid и ask
            bids = bucket_data[bucket_data['side'] == 'bid']
            asks = bucket_data[bucket_data['side'] == 'ask']
            
            best_bid = None
            best_ask = None
            
            # 3. Находим best bid (максимальная цена)
            if not bids.empty:
                # Фильтруем только активные уровни (size > 0)
                active_bids = bids[bids['size'] > 0]
                if not active_bids.empty:
                    best_bid = active_bids['price'].max()
            
            # 4. Находим best ask (минимальная цена)
            if not asks.empty:
                # Фильтруем только активные уровни (size > 0)
                active_asks = asks[asks['size'] > 0]
                if not active_asks.empty:
                    best_ask = active_asks['price'].min()
            
            # 5. Создаем quote только если есть и bid и ask
            if best_bid is not None and best_ask is not None:
                # Проверяем логичность (bid < ask)
                if best_bid < best_ask:
                    mid = (best_bid + best_ask) / 2
                    spread = best_ask - best_bid
                    
                    quotes_list.append({
                        'ts_ns': bucket_time,
                        'exchange': exchange,
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'mid': mid,
                        'spread': spread
                    })
                else:
                    logger.debug(f"Пропущен нелогичный quote: bid={best_bid}, ask={best_ask}")
            else:
                logger.debug(f"Пропущен неполный quote: bid={best_bid}, ask={best_ask}")
    
    # 6. Создаем DataFrame и сортируем по времени
    if quotes_list:
        quotes_df = pd.DataFrame(quotes_list)
        quotes_df = quotes_df.sort_values('ts_ns').reset_index(drop=True)
        
        exchanges_count = len(quotes_df['exchange'].unique())
        logger.info(f"Восстановлено {len(quotes_df)} quotes для {exchanges_count} бирж")
        logger.info(f"Временной диапазон: {quotes_df['ts_ns'].min()} - {quotes_df['ts_ns'].max()}")
        
        # Логируем статистику по биржам
        for exchange in quotes_df['exchange'].unique():
            ex_quotes = quotes_df[quotes_df['exchange'] == exchange]
            logger.info(f"  {exchange}: {len(ex_quotes)} quotes")
    else:
        quotes_df = pd.DataFrame(columns=[
            'ts_ns', 'exchange', 'best_bid', 'best_ask', 'mid', 'spread'
        ])
        logger.warning("Не удалось восстановить ни одного quote")
    
    return quotes_df

def calculate_spreads(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычислить спреды и дополнительные метрики
    
    Args:
        quotes_df: DataFrame с quotes
        
    Returns:
        DataFrame с добавленными метриками спредов
    """
    logger.info("Вычисление спредов...")
    
    if quotes_df.empty:
        return quotes_df
    
    # Создаем копию для модификации
    quotes_df = quotes_df.copy()
    
    # 1. Абсолютный спред (ask - bid) - уже есть в колонке 'spread'
    quotes_df['spread_abs'] = quotes_df['spread']
    
    # 2. Относительный спред (%)
    quotes_df['spread_rel'] = quotes_df['spread'] / quotes_df['mid'] * 100
    
    # 3. Скользящие средние спредов (по биржам)
    for exchange in quotes_df['exchange'].unique():
        ex_mask = quotes_df['exchange'] == exchange
        ex_data = quotes_df[ex_mask].copy()
        
        if len(ex_data) > 0:
            # Сортируем по времени
            ex_data = ex_data.sort_values('ts_ns')
            
            # Скользящие средние для абсолютного спреда
            quotes_df.loc[ex_mask, 'spread_ma_5'] = ex_data['spread'].rolling(window=5, min_periods=1).mean()
            quotes_df.loc[ex_mask, 'spread_ma_20'] = ex_data['spread'].rolling(window=20, min_periods=1).mean()
            
            # Скользящие средние для относительного спреда
            quotes_df.loc[ex_mask, 'spread_rel_ma_5'] = ex_data['spread_rel'].rolling(window=5, min_periods=1).mean()
            quotes_df.loc[ex_mask, 'spread_rel_ma_20'] = ex_data['spread_rel'].rolling(window=20, min_periods=1).mean()
    
    # 4. Волатильность спредов (скользящее стандартное отклонение)
    for exchange in quotes_df['exchange'].unique():
        ex_mask = quotes_df['exchange'] == exchange
        ex_data = quotes_df[ex_mask].copy()
        
        if len(ex_data) > 0:
            ex_data = ex_data.sort_values('ts_ns')
            
            # Волатильность абсолютного спреда
            quotes_df.loc[ex_mask, 'spread_volatility'] = ex_data['spread'].rolling(window=20, min_periods=1).std()
            
            # Волатильность относительного спреда
            quotes_df.loc[ex_mask, 'spread_rel_volatility'] = ex_data['spread_rel'].rolling(window=20, min_periods=1).std()
    
    # 5. Дополнительные метрики
    quotes_df['bid_ask_ratio'] = quotes_df['best_bid'] / quotes_df['best_ask']
    quotes_df['mid_change'] = quotes_df['mid'].diff()
    quotes_df['mid_change_pct'] = quotes_df['mid_change'] / quotes_df['mid'].shift(1) * 100
    
    # 6. Логируем статистику
    logger.info("Статистика спредов:")
    logger.info(f"  Абсолютный спред: {quotes_df['spread'].min():.4f} - {quotes_df['spread'].max():.4f}")
    logger.info(f"  Относительный спред: {quotes_df['spread_rel'].min():.4f}% - {quotes_df['spread_rel'].max():.4f}%")
    
    # Статистика по биржам
    for exchange in quotes_df['exchange'].unique():
        ex_data = quotes_df[quotes_df['exchange'] == exchange]
        logger.info(f"  {exchange}: средний спред {ex_data['spread'].mean():.4f} ({ex_data['spread_rel'].mean():.4f}%)")
    
    return quotes_df

def aggregate_by_tick(
    quotes_df: pd.DataFrame,
    tick_size_ms: int = 10
) -> pd.DataFrame:
    """
    Агрегировать quotes по временным тикам
    
    Args:
        quotes_df: DataFrame с quotes
        tick_size_ms: Размер тика в миллисекундах
        
    Returns:
        Агрегированный DataFrame
    """
    logger.info(f"Агрегация quotes по тикам {tick_size_ms}ms...")
    
    if quotes_df.empty:
        return quotes_df
    
    # Создаем копию для модификации
    quotes_df = quotes_df.copy()
    
    # 1. Округлить timestamps до границ тиков
    tick_size_ns = tick_size_ms * 1_000_000
    quotes_df['tick_bucket'] = (quotes_df['ts_ns'] // tick_size_ns) * tick_size_ns
    
    logger.info(f"Создано {quotes_df['tick_bucket'].nunique()} тиковых окон")
    
    # 2. Агрегируем данные внутри каждого тика
    aggregated_quotes = []
    
    for exchange in quotes_df['exchange'].unique():
        ex_data = quotes_df[quotes_df['exchange'] == exchange]
        
        for tick_time in ex_data['tick_bucket'].unique():
            tick_data = ex_data[ex_data['tick_bucket'] == tick_time]
            
            if len(tick_data) > 0:
                # Агрегируем данные внутри тика
                aggregated_quote = {
                    'ts_ns': tick_time,
                    'exchange': exchange,
                    'best_bid': tick_data['best_bid'].mean(),  # Среднее значение
                    'best_ask': tick_data['best_ask'].mean(),
                    'mid': tick_data['mid'].mean(),
                    'spread': tick_data['spread'].mean(),
                    'spread_abs': tick_data['spread_abs'].mean(),
                    'spread_rel': tick_data['spread_rel'].mean(),
                    'quotes_count': len(tick_data)  # Количество quotes в тике
                }
                
                # Добавляем скользящие средние если есть
                if 'spread_ma_5' in tick_data.columns:
                    aggregated_quote['spread_ma_5'] = tick_data['spread_ma_5'].mean()
                    aggregated_quote['spread_ma_20'] = tick_data['spread_ma_5'].mean()
                    aggregated_quote['spread_rel_ma_5'] = tick_data['spread_rel_ma_5'].mean()
                    aggregated_quote['spread_rel_ma_20'] = tick_data['spread_rel_ma_20'].mean()
                
                # Добавляем волатильность если есть
                if 'spread_volatility' in tick_data.columns:
                    aggregated_quote['spread_volatility'] = tick_data['spread_volatility'].mean()
                    aggregated_quote['spread_rel_volatility'] = tick_data['spread_rel_volatility'].mean()
                
                # Добавляем дополнительные метрики если есть
                if 'bid_ask_ratio' in tick_data.columns:
                    aggregated_quote['bid_ask_ratio'] = tick_data['bid_ask_ratio'].mean()
                    aggregated_quote['mid_change'] = tick_data['mid_change'].sum()  # Сумма изменений
                    aggregated_quote['mid_change_pct'] = tick_data['mid_change_pct'].mean()
                
                aggregated_quotes.append(aggregated_quote)
    
    # 3. Создаем агрегированный DataFrame
    if aggregated_quotes:
        aggregated_df = pd.DataFrame(aggregated_quotes)
        aggregated_df = aggregated_df.sort_values('ts_ns').reset_index(drop=True)
        
        logger.info(f"Агрегировано {len(aggregated_df)} тиковых quotes")
        logger.info(f"Среднее количество quotes на тик: {aggregated_df['quotes_count'].mean():.1f}")
        
        # Статистика по биржам
        for exchange in aggregated_df['exchange'].unique():
            ex_data = aggregated_df[aggregated_df['exchange'] == exchange]
            logger.info(f"  {exchange}: {len(ex_data)} тиковых quotes")
    else:
        aggregated_df = pd.DataFrame()
        logger.warning("Не удалось агрегировать quotes")
    
    return aggregated_df

def validate_quotes(quotes_df: pd.DataFrame) -> Dict[str, float]:
    """
    Валидация качества восстановленных quotes
    
    Args:
        quotes_df: DataFrame с quotes
        
    Returns:
        Словарь с метриками качества
    """
    logger.info("Валидация quotes...")
    
    if quotes_df.empty:
        return {"quality": 0.0, "completeness": 0.0}
    
    quality_metrics = {}
    
    # 1. Проверить логичность цен (bid < ask)
    valid_bid_ask = (quotes_df['best_bid'] < quotes_df['best_ask']).sum()
    total_quotes = len(quotes_df)
    bid_ask_validity = valid_bid_ask / total_quotes if total_quotes > 0 else 0
    
    quality_metrics["bid_ask_valid"] = bid_ask_validity
    
    if bid_ask_validity < 1.0:
        invalid_count = total_quotes - valid_bid_ask
        logger.warning(f"Найдено {invalid_count} нелогичных quotes (bid >= ask)")
    
    # 2. Оценить полноту данных
    # Проверяем наличие всех необходимых колонок
    required_columns = ['ts_ns', 'exchange', 'best_bid', 'best_ask', 'mid', 'spread']
    missing_columns = [col for col in required_columns if col not in quotes_df.columns]
    
    if missing_columns:
        logger.warning(f"Отсутствуют колонки: {missing_columns}")
        completeness = (len(required_columns) - len(missing_columns)) / len(required_columns)
    else:
        completeness = 1.0
    
    quality_metrics["completeness"] = completeness
    
    # 3. Проверить временные разрывы
    if len(quotes_df) > 1:
        # Сортируем по времени
        sorted_quotes = quotes_df.sort_values('ts_ns')
        
        # Вычисляем временные интервалы между quotes
        time_diffs = sorted_quotes['ts_ns'].diff().dropna()
        
        if len(time_diffs) > 0:
            # Находим большие разрывы (> 1 секунды)
            large_gaps = (time_diffs > 1_000_000_000).sum()
            gap_ratio = large_gaps / len(time_diffs)
            
            quality_metrics["time_coverage"] = 1.0 - gap_ratio
            
            if large_gaps > 0:
                logger.warning(f"Обнаружено {large_gaps} больших временных разрывов (>1с)")
        else:
            quality_metrics["time_coverage"] = 1.0
    else:
        quality_metrics["time_coverage"] = 1.0
    
    # 4. Проверить качество спредов
    if 'spread' in quotes_df.columns:
        # Проверяем на отрицательные спреды
        negative_spreads = (quotes_df['spread'] < 0).sum()
        if negative_spreads > 0:
            logger.warning(f"Найдено {negative_spreads} отрицательных спредов")
        
        # Проверяем на подозрительно большие спреды (>10% от цены)
        if 'spread_rel' in quotes_df.columns:
            large_spreads = (quotes_df['spread_rel'] > 10.0).sum()
            if large_spreads > 0:
                logger.warning(f"Найдено {large_spreads} подозрительно больших спредов (>10%)")
    
    # 5. Проверить распределение по биржам
    exchange_counts = quotes_df['exchange'].value_counts()
    if len(exchange_counts) > 1:
        # Вычисляем равномерность распределения
        total = exchange_counts.sum()
        expected_per_exchange = total / len(exchange_counts)
        variance = ((exchange_counts - expected_per_exchange) ** 2).sum() / len(exchange_counts)
        distribution_score = max(0, 1.0 - variance / (expected_per_exchange ** 2))
        
        quality_metrics["distribution_balance"] = distribution_score
        
        if distribution_score < 0.8:
            logger.warning(f"Неравномерное распределение по биржам: {exchange_counts.to_dict()}")
    else:
        quality_metrics["distribution_balance"] = 1.0
    
    # 6. Общий скор качества
    overall_quality = np.mean(list(quality_metrics.values()))
    quality_metrics["overall_quality"] = overall_quality
    
    # Логируем результаты валидации
    logger.info("=== МЕТРИКИ КАЧЕСТВА QUOTES ===")
    for metric, score in quality_metrics.items():
        logger.info(f"{metric}: {score:.3f}")
    
    # Цветная оценка качества
    if overall_quality >= 0.9:
        logger.info("🟢 Качество quotes: ОТЛИЧНОЕ")
    elif overall_quality >= 0.7:
        logger.info("🟡 Качество quotes: ХОРОШЕЕ")
    elif overall_quality >= 0.5:
        logger.info("🟠 Качество quotes: УДОВЛЕТВОРИТЕЛЬНОЕ")
    else:
        logger.warning("🔴 Качество quotes: ПЛОХОЕ")
    
    return quality_metrics

def save_quotes(
    quotes_df: pd.DataFrame,
    output_path: Path,
    config: Dict
) -> None:
    """
    Сохранить quotes в parquet файл
    
    Args:
        quotes_df: DataFrame с quotes
        output_path: Путь для сохранения
        config: Конфигурация
    """
    logger.info(f"Сохранение quotes в {output_path}...")
    
    if quotes_df.empty:
        logger.warning("Quotes DataFrame пустой, сохранение пропущено")
        return
    
    try:
        # 1. Создать директорию если не существует
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 2. Получаем настройки сжатия из конфигурации
        compression = config.get("export", {}).get("compression", "snappy")
        
        # 3. Сохраняем в parquet с настройками
        quotes_df.to_parquet(
            output_path, 
            engine="pyarrow", 
            compression=compression,
            index=False
        )
        
        # 4. Логируем информацию о сохранении
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Quotes сохранены: {len(quotes_df)} строк")
        logger.info(f"Размер файла: {file_size_mb:.1f} MB")
        logger.info(f"Путь: {output_path}")
        
        # 5. Логируем статистику по биржам
        if 'exchange' in quotes_df.columns:
            exchange_counts = quotes_df['exchange'].value_counts()
            logger.info("Распределение по биржам:")
            for exchange, count in exchange_counts.items():
                logger.info(f"  {exchange}: {count:,} quotes")
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении quotes: {e}")
        raise

def run_best_prices(
    depth_df: pd.DataFrame,
    config: Dict,
    output_dir: Path = Path("data/quotes")
) -> pd.DataFrame:
    """
    Основная функция восстановления лучших цен
    
    Args:
        depth_df: Нормализованные depth данные
        config: Конфигурация
        output_dir: Директория для сохранения
        
    Returns:
        DataFrame с quotes
    """
    logger.info("Запуск восстановления лучших цен...")
    
    # Проверяем конфигурацию
    best_prices_config = config.get('best_prices', {})
    log_progress = best_prices_config.get('log_progress', True)
    
    try:
        if log_progress:
            logger.info("Этап 1/4: Восстановление quotes из depth данных")
        
        # 1. Восстановить quotes
        tick_size_ms = best_prices_config.get("tick_size_ms", 10)
        logger.info(f"Шаг агрегации: {tick_size_ms}ms")
        
        quotes_df = restore_quotes(depth_df, tick_size_ms, best_prices_config)
        
        if quotes_df.empty:
            logger.warning("Не удалось восстановить quotes, завершаем работу")
            return quotes_df
        
        if log_progress:
            logger.info("Этап 2/4: Вычисление спредов и метрик")
        
        # 2. Вычислить спреды и дополнительные метрики
        quotes_df = calculate_spreads(quotes_df, best_prices_config)
        
        if log_progress:
            logger.info("Этап 3/4: Агрегация по временным тикам")
        
        # 3. Агрегировать по тикам
        quotes_df = aggregate_by_tick(quotes_df, tick_size_ms)
        
        if quotes_df.empty:
            logger.warning("Агрегация не дала результатов")
            return quotes_df
        
        if log_progress:
            logger.info("Этап 4/4: Валидация качества и сохранение")
        
        # 4. Валидировать качество
        quality = validate_quotes(quotes_df)
        
        # 5. Сохранить результат
        output_path = output_dir / "quotes.parquet"
        save_quotes(quotes_df, output_path, config)
        
        # 6. Итоговая статистика
        logger.info("=== ИТОГИ ВОССТАНОВЛЕНИЯ QUOTES ===")
        logger.info(f"Исходные depth записи: {len(depth_df):,}")
        logger.info(f"Восстановлено quotes: {len(quotes_df):,}")
        logger.info(f"Коэффициент восстановления: {len(quotes_df)/len(depth_df)*100:.2f}%")
        logger.info(f"Общее качество: {quality.get('overall_quality', 0):.3f}")
        
        # Статистика по биржам
        if 'exchange' in quotes_df.columns:
            for exchange in quotes_df['exchange'].unique():
                ex_count = len(quotes_df[quotes_df['exchange'] == exchange])
                logger.info(f"  {exchange}: {ex_count:,} quotes")
        
        logger.info("✅ Восстановление лучших цен завершено успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при восстановлении quotes: {e}")
        raise
    
    return quotes_df
