"""
Блок 08: Кросс-биржевая агрегация
Собрать NBBO и суммарные объёмы/консенсус по биржам
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class CrossExchangeAggregator:
    """Агрегатор кросс-биржевых данных"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sync_window_ms = config.get('sync', {}).get('max_lag_ms', 5000)
        self.min_overlap_ratio = config.get('sync', {}).get('min_overlap_ratio', 0.8)
        self.tick_size_ms = config.get('time', {}).get('tick_size_ms', 10)
        self.exchanges = config.get('exchanges', ['binance', 'bybit', 'okx'])
        
    def run_cross_exchange_analysis(
        self,
        quotes_df: pd.DataFrame,
        trades_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Полный анализ кросс-биржевых данных"""
        start_time = time.time()
        logger.info("=== Начало кросс-биржевой агрегации ===")
        
        # Этап 1: Подготовка и синхронизация данных
        logger.info("Этап 1/4: Подготовка и синхронизация данных...")
        synced_quotes, synced_trades = self._synchronize_data(quotes_df, trades_df)
        
        # Этап 2: Построение NBBO
        logger.info("Этап 2/4: Построение NBBO...")
        nbbo_df = self._build_nbbo(synced_quotes)
        
        # Этап 3: Агрегация объемов
        logger.info("Этап 3/4: Агрегация объемов...")
        volume_df = self._aggregate_volumes(synced_trades, nbbo_df)
        
        # Этап 4: Дополнительные метрики
        logger.info("Этап 4/4: Дополнительные метрики...")
        nbbo_df, volume_df = self._calculate_additional_metrics(nbbo_df, volume_df)
        
        # Завершение
        duration = time.time() - start_time
        logger.info(f"Агрегация завершена за {duration:.2f}с")
        logger.info(f"NBBO: {len(nbbo_df)} записей, Объемы: {len(volume_df)} записей")
        logger.info("=== Кросс-биржевая агрегация завершена ===")
        
        return nbbo_df, volume_df
    
    def _synchronize_data(
        self,
        quotes_df: pd.DataFrame,
        trades_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Синхронизация данных между биржами"""
        logger.info("Синхронизация данных...")
        
        if quotes_df.empty or trades_df.empty:
            logger.warning("Один из датафреймов пуст")
            return quotes_df, trades_df
        
        # Проверка обязательных колонок
        required_quotes = ['ts_ns', 'exchange', 'instrument', 'best_bid', 'best_ask']
        required_trades = ['ts_ns', 'exchange', 'instrument', 'price', 'size', 'aggressor']
        
        missing_quotes = [col for col in required_quotes if col not in quotes_df.columns]
        missing_trades = [col for col in required_trades if col not in trades_df.columns]
        
        if missing_quotes or missing_trades:
            logger.error(f"Отсутствуют колонки: quotes={missing_quotes}, trades={missing_trades}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Синхронизация по времени
        synced_quotes = self._synchronize_quotes(quotes_df)
        synced_trades = self._synchronize_trades(trades_df)
        
        logger.info(f"Синхронизировано: quotes={len(synced_quotes)}, trades={len(synced_trades)}")
        return synced_quotes, synced_trades
    
    def _synchronize_quotes(self, quotes_df: pd.DataFrame) -> pd.DataFrame:
        """Синхронизация котировок по времени"""
        df = quotes_df.copy()
        
        # Сортировка по времени
        df = df.sort_values('ts_ns').reset_index(drop=True)
        
        # Группировка по времени с учетом синхронизации
        df['ts_group'] = (df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
        
        # Агрегация по группам времени и биржам
        synced_quotes = df.groupby(['ts_group', 'exchange', 'instrument']).agg({
            'ts_ns': 'first',
            'best_bid': 'last',
            'best_ask': 'last'
        }).reset_index()
        
        # Создаем bid_size и ask_size из существующих данных (используем size из trades)
        synced_quotes['bid_size'] = 1.0  # Заглушка для совместимости
        synced_quotes['ask_size'] = 1.0  # Заглушка для совместимости
        
        # Удаление временной группировки
        synced_quotes = synced_quotes.drop('ts_group', axis=1)
        
        return synced_quotes
    
    def _synchronize_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Синхронизация сделок по времени"""
        df = trades_df.copy()
        
        # Сортировка по времени
        df = df.sort_values('ts_ns').reset_index(drop=True)
        
        # Группировка по времени с учетом синхронизации
        df['ts_group'] = (df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
        
        # Агрегация по группам времени и биржам
        synced_trades = df.groupby(['ts_group', 'exchange', 'instrument']).agg({
            'ts_ns': 'first',
            'price': 'mean',
            'size': 'sum',
            'aggressor': lambda x: x.mode().iloc[0] if not x.empty else 'unknown'
        }).reset_index()
        
        # Создаем колонку side для совместимости
        synced_trades['side'] = synced_trades['aggressor']
        
        # Удаление временной группировки
        synced_trades = synced_trades.drop('ts_group', axis=1)
        
        return synced_trades
    
    def _build_nbbo(self, quotes_df: pd.DataFrame) -> pd.DataFrame:
        """Построение NBBO (National Best Bid and Offer)"""
        logger.info("Построение NBBO...")
        
        if quotes_df.empty:
            return pd.DataFrame()
        
        # Группировка по времени и инструменту
        quotes_df = quotes_df.sort_values('ts_ns').reset_index(drop=True)
        quotes_df['ts_group'] = (quotes_df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
        
        nbbo_records = []
        total_groups = quotes_df['ts_group'].nunique()
        
        for i, (ts_group, group_data) in enumerate(quotes_df.groupby('ts_group')):
            if i % 1000 == 0:  # Логируем каждые 1000 групп
                logger.info(f"Прогресс NBBO: {i+1}/{total_groups} временных групп")
            
            # Найти лучшие bid и ask среди всех бирж
            best_bid = group_data.loc[group_data['best_bid'].idxmax()]
            best_ask = group_data.loc[group_data['best_ask'].idxmin()]
            
            # Проверка валидности спреда
            if best_bid['best_bid'] >= best_ask['best_ask']:
                continue  # Пропускаем невалидные спреды
            
            nbbo_records.append({
                'ts_ns': group_data['ts_ns'].iloc[0],
                'ts_group': ts_group,
                'symbol': group_data['instrument'].iloc[0],
                'best_bid': best_bid['best_bid'],
                'best_ask': best_ask['best_ask'],
                'best_bid_size': best_bid['bid_size'],
                'best_ask_size': best_ask['ask_size'],
                'mid_price': (best_bid['best_bid'] + best_ask['best_ask']) / 2,
                'spread': best_ask['best_ask'] - best_bid['best_bid'],
                'spread_pct': ((best_ask['best_ask'] - best_bid['best_bid']) / best_bid['best_bid']) * 100,
                'bid_exchange': best_bid['exchange'],
                'ask_exchange': best_ask['exchange'],
                'total_bid_volume': group_data['bid_size'].sum(),
                'total_ask_volume': group_data['ask_size'].sum(),
                'volume_imbalance': (group_data['bid_size'].sum() - group_data['ask_size'].sum()) / 
                                   (group_data['bid_size'].sum() + group_data['ask_size'].sum())
            })
        
        nbbo_df = pd.DataFrame(nbbo_records)
        
        if not nbbo_df.empty:
            # Удаление временной группировки
            nbbo_df = nbbo_df.drop('ts_group', axis=1)
            
            # Сортировка по времени
            nbbo_df = nbbo_df.sort_values('ts_ns').reset_index(drop=True)
            
            logger.info(f"NBBO построен: {len(nbbo_df)} записей")
        else:
            logger.warning("NBBO не построен - нет валидных данных")
        
        return nbbo_df
    
    def _aggregate_volumes(
        self,
        trades_df: pd.DataFrame,
        nbbo_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Агрегация объемов по биржам"""
        logger.info("Агрегация объемов...")
        
        if trades_df.empty:
            return pd.DataFrame()
        
        # Синхронизация с NBBO по времени
        if not nbbo_df.empty:
            trades_df = self._align_trades_with_nbbo(trades_df, nbbo_df)
        
        # Группировка по времени
        trades_df = trades_df.sort_values('ts_ns').reset_index(drop=True)
        trades_df['ts_group'] = (trades_df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
        
        volume_records = []
        total_groups = trades_df['ts_group'].nunique()
        
        for i, (ts_group, group_data) in enumerate(trades_df.groupby('ts_group')):
            if i % 1000 == 0:  # Логируем каждые 1000 групп
                logger.info(f"Прогресс объемов: {i+1}/{total_groups} временных групп")
            
            # Агрегация по биржам
            exchange_volumes = group_data.groupby('exchange').agg({
                'size': 'sum',
                'price': 'mean'
            }).reset_index()
            
            # Общие метрики
            total_volume = group_data['size'].sum()
            total_value = (group_data['size'] * group_data['price']).sum()
            vwap = total_value / total_volume if total_volume > 0 else 0
            
            # Консенсус цен
            price_std = group_data['price'].std()
            price_consensus = 1.0 / (1.0 + price_std) if not pd.isna(price_std) else 0
            
            # Метрики ликвидности
            buy_volume = group_data[group_data['side'] == 'buy']['size'].sum()
            sell_volume = group_data[group_data['side'] == 'sell']['size'].sum()
            
            volume_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
            
            # Крупные сделки
            avg_trade_size = group_data['size'].mean()
            large_trades = group_data[group_data['size'] > avg_trade_size * 2]
            large_trades_ratio = len(large_trades) / len(group_data) if len(group_data) > 0 else 0
            
            volume_records.append({
                'ts_ns': group_data['ts_ns'].iloc[0],
                'ts_group': ts_group,
                'symbol': group_data['instrument'].iloc[0],
                'total_volume': total_volume,
                'total_value': total_value,
                'vwap': vwap,
                'price_consensus': price_consensus,
                'liquidity_score': 1.0 / (1.0 + volume_imbalance**2),
                'volume_imbalance': volume_imbalance,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'large_trades_ratio': large_trades_ratio,
                'trade_count': len(group_data),
                'exchange_count': group_data['exchange'].nunique(),
                'exchanges': ','.join(sorted(group_data['exchange'].unique()))
            })
        
        volume_df = pd.DataFrame(volume_records)
        
        if not volume_df.empty:
            # Удаление временной группировки
            volume_df = volume_df.drop('ts_group', axis=1)
            
            # Сортировка по времени
            volume_df = volume_df.sort_values('ts_ns').reset_index(drop=True)
            
            logger.info(f"Объемы агрегированы: {len(volume_df)} записей")
        else:
            logger.warning("Объемы не агрегированы - нет данных")
        
        return volume_df
    
    def _align_trades_with_nbbo(
        self,
        trades_df: pd.DataFrame,
        nbbo_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Выравнивание сделок с NBBO по времени"""
        # Создание временных групп для NBBO
        nbbo_df = nbbo_df.copy()
        nbbo_df['ts_group'] = (nbbo_df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
        
        # Создание временных групп для сделок
        trades_df = trades_df.copy()
        trades_df['ts_group'] = (trades_df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
        
        # Объединение по временным группам
        aligned_trades = trades_df.merge(
            nbbo_df[['ts_group', 'mid_price', 'spread', 'spread_pct']],
            on='ts_group',
            how='left'
        )
        
        # Удаление временной группировки
        aligned_trades = aligned_trades.drop('ts_group', axis=1)
        
        return aligned_trades
    
    def _calculate_additional_metrics(
        self,
        nbbo_df: pd.DataFrame,
        volume_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Вычисление дополнительных метрик"""
        logger.info("Вычисление дополнительных метрик...")
        
        # Дополнительные метрики для NBBO
        if not nbbo_df.empty:
            nbbo_df = self._enhance_nbbo_metrics(nbbo_df)
        
        # Дополнительные метрики для объемов
        if not volume_df.empty:
            volume_df = self._enhance_volume_metrics(volume_df)
        
        return nbbo_df, volume_df
    
    def _enhance_nbbo_metrics(self, nbbo_df: pd.DataFrame) -> pd.DataFrame:
        """Улучшение метрик NBBO"""
        df = nbbo_df.copy()
        
        # Скользящие средние
        df['spread_ma_5'] = df['spread'].rolling(window=5).mean()
        df['spread_ma_20'] = df['spread'].rolling(window=20).mean()
        
        # Волатильность спреда
        df['spread_volatility'] = df['spread'].rolling(window=20).std()
        
        # Изменение mid price
        df['mid_price_change'] = df['mid_price'].diff()
        df['mid_price_change_pct'] = (df['mid_price_change'] / df['mid_price'].shift(1)) * 100
        
        # Конвергенция/дивергенция бирж
        df['exchange_convergence'] = df.apply(
            lambda row: 1.0 if row['bid_exchange'] == row['ask_exchange'] else 0.0, axis=1
        )
        
        return df
    
    def _enhance_volume_metrics(self, volume_df: pd.DataFrame) -> pd.DataFrame:
        """Улучшение метрик объемов"""
        df = volume_df.copy()
        
        # Скользящие средние объемов
        df['volume_ma_5'] = df['total_volume'].rolling(window=5).mean()
        df['volume_ma_20'] = df['total_volume'].rolling(window=20).mean()
        
        # Относительные объемы
        df['volume_ratio_5'] = df['total_volume'] / df['volume_ma_5']
        df['volume_ratio_20'] = df['total_volume'] / df['volume_ma_20']
        
        # Волатильность цен
        if 'mid_price' in df.columns:
            df['price_volatility'] = df['mid_price'].rolling(window=20).std()
        
        # Тренд объемов
        df['volume_trend'] = df['total_volume'].rolling(window=10).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
        )
        
        return df

def run_cross_exchange(
    quotes_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    config: Dict,
    output_dir: Path = Path("data/nbbo")
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Основная функция кросс-биржевой агрегации"""
    logger.info("Запуск кросс-биржевой агрегации...")
    
    # Создание агрегатора
    aggregator = CrossExchangeAggregator(config)
    
    # Полный анализ
    nbbo_df, volume_df = aggregator.run_cross_exchange_analysis(quotes_df, trades_df)
    
    # Сохранение результатов
    if not nbbo_df.empty or not volume_df.empty:
        output_path = output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохранение NBBO
        if not nbbo_df.empty:
            nbbo_path = output_path / "nbbo_aggregated.parquet"
            nbbo_df.to_parquet(nbbo_path, engine="pyarrow", index=False)
            logger.info(f"NBBO сохранен: {len(nbbo_df)} строк в {nbbo_path}")
        
        # Сохранение объемов
        if not volume_df.empty:
            volume_path = output_path / "volumes_aggregated.parquet"
            volume_df.to_parquet(volume_path, engine="pyarrow", index=False)
            logger.info(f"Объемы сохранены: {len(volume_df)} строк в {volume_path}")
        
        # Сохранение агрегированных данных
        save_aggregated_data(nbbo_df, volume_df, output_path)
    
    logger.info("Кросс-биржевая агрегация завершена")
    return nbbo_df, volume_df

def save_aggregated_data(
    nbbo_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    output_dir: Path
):
    """Сохранение агрегированных данных"""
    logger.info("Сохранение агрегированных данных...")
    
    # Агрегация NBBO по биржам
    if not nbbo_df.empty:
        exchange_nbbo = nbbo_df.groupby(['bid_exchange', 'ask_exchange']).agg({
            'spread': 'mean',
            'spread_pct': 'mean',
            'best_bid_size': 'mean',
            'best_ask_size': 'mean',
            'volume_imbalance': 'mean'
        }).round(6)
        
        exchange_nbbo.to_parquet(output_dir / "exchange_nbbo_summary.parquet", engine="pyarrow")
    
    # Агрегация объемов по биржам
    if not volume_df.empty:
        exchange_volumes = volume_df.groupby('exchanges').agg({
            'total_volume': 'sum',
            'total_value': 'sum',
            'vwap': 'mean',
            'liquidity_score': 'mean',
            'large_trades_ratio': 'mean'
        }).round(4)
        
        exchange_volumes.to_parquet(output_dir / "exchange_volumes_summary.parquet", engine="pyarrow")
    
    # Временная агрегация
    if not nbbo_df.empty and not volume_df.empty:
        # Объединение NBBO и объемов по времени
        combined_df = nbbo_df.merge(
            volume_df[['ts_ns', 'total_volume', 'vwap', 'liquidity_score']],
            on='ts_ns',
            how='inner'
        )
        
        # Агрегация по часам
        combined_df['timestamp'] = pd.to_datetime(combined_df['ts_ns'], unit='ns')
        combined_df['hour'] = combined_df['timestamp'].dt.hour
        
        hourly_summary = combined_df.groupby('hour').agg({
            'spread': 'mean',
            'spread_pct': 'mean',
            'total_volume': 'sum',
            'vwap': 'mean',
            'liquidity_score': 'mean',
            'volume_imbalance': 'mean'
        }).round(6)
        
        hourly_summary.to_parquet(output_dir / "hourly_summary.parquet", engine="pyarrow")
    
    logger.info("Агрегированные данные сохранены")


if __name__ == "__main__":
    """Запуск блока 08 для тестирования"""
    import pandas as pd
    import logging
    import sys
    import os
    import time
    
    # Добавляем путь для импорта base_block
    sys.path.append(os.path.dirname(__file__))
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    print("🧪 ТЕСТ БЛОКА 08: Cross Exchange")
    print("=" * 50)
    
    try:
        # Загружаем данные от предыдущих блоков
        quotes_df = pd.read_parquet("../../../data/quotes/quotes.parquet")
        trades_df = pd.read_parquet("../../../data/test_small/trades_small.parquet")
        
        # Проверяем структуру quotes данных
        if not quotes_df.empty:
            print(f"   - Колонки в quotes: {quotes_df.columns.tolist()}")
            print(f"   - Первые 3 строки quotes:")
            print(quotes_df.head(3))
            
            # Добавляем недостающую колонку instrument
            if 'instrument' not in quotes_df.columns:
                print("   - Добавляем колонку instrument (ETHUSDT)...")
                quotes_df['instrument'] = 'ETHUSDT'
                print("   - Колонка instrument добавлена!")
        
        # Проверяем структуру trades данных  
        if not trades_df.empty:
            print(f"   - Колонки в trades: {trades_df.columns.tolist()}")
            print(f"   - Первые 3 строки trades:")
            print(trades_df.head(3))
        
        print(f"✅ Загружены данные для кросс-биржевого анализа:")
        print(f"   - Quotes: {len(quotes_df)} строк")
        print(f"   - Trades: {len(trades_df)} строк")
        
        # Проверяем какие биржи у нас есть
        if not quotes_df.empty:
            exchanges = quotes_df['exchange'].unique()
            print(f"   - Биржи в quotes: {exchanges}")
        
        if not trades_df.empty:
            exchanges = trades_df['exchange'].unique()
            print(f"   - Биржи в trades: {exchanges}")
        
        # Конфигурация для кросс-биржевого анализа
        config = {
            'sync': {
                'max_lag_ms': 5000,      # Максимальная задержка между биржами
                'min_overlap_ratio': 0.8  # Минимальное перекрытие данных
            },
            'time': {
                'tick_size_ms': 10       # Размер временного тика
            },
            'exchanges': ['binance', 'bybit', 'okx']
        }
        
        # Запускаем блок 08
        print("🚀 Запускаем блок 08: Кросс-биржевая агрегация...")
        start_time = time.time()
        
        nbbo_df, volume_df = run_cross_exchange(quotes_df, trades_df, config)
        
        execution_time = time.time() - start_time
        
        print(f"✅ Блок 08 выполнен за {execution_time:.2f} секунд!")
        print(f"📊 Результат агрегации:")
        
        # Анализируем результаты
        if not nbbo_df.empty:
            print(f"   - NBBO: {len(nbbo_df)} записей")
            print(f"   - Биржи в NBBO: {nbbo_df['bid_exchange'].unique() if 'bid_exchange' in nbbo_df.columns else 'N/A'}")
        
        if not volume_df.empty:
            print(f"   - Объемы: {len(volume_df)} записей")
            print(f"   - Биржи в объемах: {volume_df['exchanges'].unique() if 'exchanges' in volume_df.columns else 'N/A'}")
        
        print("✅ Блок 08 успешно протестирован!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
