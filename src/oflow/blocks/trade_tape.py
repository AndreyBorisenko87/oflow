"""
Блок 07: Лента сделок
Анализ агрессии, объемы по окнам, квантили и метрики торговой активности
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class TradeTapeAnalyzer:
    """Анализатор ленты сделок"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.windows = config.get('windows', {'short': 1000, 'medium': 10000, 'long': 60000})
        self.tick_size_ms = config.get('time', {}).get('tick_size_ms', 10)
        self.aggression_threshold = config.get('aggression', {}).get('threshold', 0.6)
        
    def analyze_trade_tape(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Полный анализ ленты сделок"""
        start_time = time.time()
        logger.info("=== Начало анализа ленты сделок ===")
        
        if trades_df.empty:
            logger.warning("Лента сделок пуста")
            return trades_df
        
        # Этап 1: Подготовка данных
        logger.info("Этап 1/4: Подготовка данных...")
        tape_df = self._prepare_data(trades_df)
        
        # Этап 2: Анализ агрессии
        logger.info("Этап 2/4: Анализ агрессии...")
        tape_df = self._analyze_aggression(tape_df)
        
        # Этап 3: Вычисление объемов по окнам
        logger.info("Этап 3/4: Вычисление объемов по окнам...")
        tape_df = self._calculate_volumes_by_windows(tape_df)
        
        # Этап 4: Дополнительные метрики
        logger.info("Этап 4/4: Дополнительные метрики...")
        tape_df = self._calculate_additional_metrics(tape_df)
        
        # Завершение
        duration = time.time() - start_time
        logger.info(f"Анализ завершен за {duration:.2f}с, обработано {len(tape_df)} сделок")
        logger.info("=== Анализ ленты сделок завершен ===")
        
        return tape_df
    
    def _prepare_data(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка данных для анализа"""
        logger.info("Подготовка данных...")
        
        # Копия для безопасного изменения
        df = trades_df.copy()
        
        # Проверка обязательных колонок
        required_columns = ['ts_ns', 'exchange', 'symbol', 'side', 'price', 'size']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Отсутствуют обязательные колонки: {missing_columns}")
            return pd.DataFrame()
        
        # Сортировка по времени
        df = df.sort_values('ts_ns').reset_index(drop=True)
        
        # Добавление временных меток
        df['timestamp'] = pd.to_datetime(df['ts_ns'], unit='ns')
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Вычисление базовых метрик
        df['trade_value'] = df['price'] * df['size']
        
        # Определение направления движения цены
        df['price_change'] = df['price'].diff()
        df['price_change_pct'] = (df['price_change'] / df['price'].shift(1)) * 100
        
        logger.info(f"Подготовлено {len(df)} сделок")
        return df
    
    def _analyze_aggression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Анализ стороны агрессии"""
        logger.info("Анализ агрессии...")
        
        # Определение агрессора по side и price
        df['aggression_type'] = 'unknown'
        df['aggression_score'] = 0.0
        df['aggressor_side'] = 'unknown'
        
        # Анализ по биржам
        exchanges = df['exchange'].unique()
        total_exchanges = len(exchanges)
        
        for i, exchange in enumerate(exchanges):
            if i % 3 == 0:  # Логируем каждые 3 биржи
                logger.info(f"Прогресс агрессии: {i+1}/{total_exchanges} бирж")
                
            exchange_data = df[df['exchange'] == exchange].copy()
            
            if exchange_data.empty:
                continue
            
            # Анализ агрессии для биржи
            exchange_data = self._analyze_exchange_aggression(exchange_data)
            
            # Обновляем основной DataFrame
            df.loc[df['exchange'] == exchange, 'aggression_type'] = exchange_data['aggression_type']
            df.loc[df['exchange'] == exchange, 'aggression_score'] = exchange_data['aggression_score']
            df.loc[df['exchange'] == exchange, 'aggressor_side'] = exchange_data['aggressor_side']
        
        # Статистика агрессии
        aggression_stats = df['aggression_type'].value_counts()
        logger.info(f"Статистика агрессии: {dict(aggression_stats)}")
        
        return df
    
    def _analyze_exchange_aggression(self, exchange_data: pd.DataFrame) -> pd.DataFrame:
        """Анализ агрессии для конкретной биржи"""
        df = exchange_data.copy()
        
        # Сортировка по времени
        df = df.sort_values('ts_ns').reset_index(drop=True)
        
        for i in range(1, len(df)):
            current_trade = df.iloc[i]
            prev_trade = df.iloc[i-1]
            
            # Определение агрессора
            if current_trade['side'] == 'buy':
                # Покупатель агрессивный, если цена выше предыдущей
                if current_trade['price'] > prev_trade['price']:
                    df.loc[df.index[i], 'aggression_type'] = 'aggressive_buy'
                    df.loc[df.index[i], 'aggression_score'] = 1.0
                    df.loc[df.index[i], 'aggressor_side'] = 'buy'
                elif current_trade['price'] == prev_trade['price']:
                    # На том же уровне - умеренная агрессия
                    df.loc[df.index[i], 'aggression_type'] = 'moderate_buy'
                    df.loc[df.index[i], 'aggression_score'] = 0.5
                    df.loc[df.index[i], 'aggressor_side'] = 'buy'
                else:
                    # Цена ниже - пассивная покупка
                    df.loc[df.index[i], 'aggression_type'] = 'passive_buy'
                    df.loc[df.index[i], 'aggression_score'] = 0.0
                    df.loc[df.index[i], 'aggressor_side'] = 'buy'
            
            else:  # side == 'sell'
                # Продавец агрессивный, если цена ниже предыдущей
                if current_trade['price'] < prev_trade['price']:
                    df.loc[df.index[i], 'aggression_type'] = 'aggressive_sell'
                    df.loc[df.index[i], 'aggression_score'] = 1.0
                    df.loc[df.index[i], 'aggressor_side'] = 'sell'
                elif current_trade['price'] == prev_trade['price']:
                    # На том же уровне - умеренная агрессия
                    df.loc[df.index[i], 'aggression_type'] = 'moderate_sell'
                    df.loc[df.index[i], 'aggression_score'] = 0.5
                    df.loc[df.index[i], 'aggressor_side'] = 'sell'
                else:
                    # Цена выше - пассивная продажа
                    df.loc[df.index[i], 'aggression_type'] = 'passive_sell'
                    df.loc[df.index[i], 'aggression_score'] = 0.0
                    df.loc[df.index[i], 'aggressor_side'] = 'sell'
        
        return df
    
    def _calculate_volumes_by_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисление объемов по временным окнам"""
        logger.info("Вычисление объемов по окнам...")
        
        # Сортировка по времени
        df = df.sort_values('ts_ns').reset_index(drop=True)
        
        # Анализ по биржам
        exchanges = df['exchange'].unique()
        total_exchanges = len(exchanges)
        
        for i, exchange in enumerate(exchanges):
            if i % 3 == 0:  # Логируем каждые 3 биржи
                logger.info(f"Прогресс объемов: {i+1}/{total_exchanges} бирж")
                
            exchange_data = df[df['exchange'] == exchange].copy()
            
            if exchange_data.empty:
                continue
            
            # Вычисление объемов для биржи
            exchange_data = self._calculate_exchange_volumes(exchange_data)
            
            # Обновляем основной DataFrame
            for window_name, window_ms in self.windows.items():
                volume_col = f'volume_{window_name}_{window_ms}ms'
                speed_col = f'speed_{window_name}_{window_ms}ms'
                cumulative_col = f'cumulative_{window_name}_{window_ms}ms'
                
                if volume_col in exchange_data.columns:
                    df.loc[df['exchange'] == exchange, volume_col] = exchange_data[volume_col]
                    df.loc[df['exchange'] == exchange, speed_col] = exchange_data[speed_col]
                    df.loc[df['exchange'] == exchange, cumulative_col] = exchange_data[cumulative_col]
        
        return df
    
    def _calculate_exchange_volumes(self, exchange_data: pd.DataFrame) -> pd.DataFrame:
        """Вычисление объемов для конкретной биржи"""
        df = exchange_data.copy()
        
        # Сортировка по времени
        df = df.sort_values('ts_ns').reset_index(drop=True)
        
        for window_name, window_ms in self.windows.items():
            volume_col = f'volume_{window_name}_{window_ms}ms'
            speed_col = f'speed_{window_name}_{window_ms}ms'
            cumulative_col = f'cumulative_{window_name}_{window_ms}ms'
            
            # Инициализация колонок
            df[volume_col] = 0.0
            df[speed_col] = 0.0
            df[cumulative_col] = 0.0
            
            for i in range(len(df)):
                current_ts = df.iloc[i]['ts_ns']
                window_start = current_ts - (window_ms * 1_000_000)
                
                # Находим сделки в окне
                window_trades = df[
                    (df['ts_ns'] >= window_start) & 
                    (df['ts_ns'] <= current_ts)
                ]
                
                if not window_trades.empty:
                    # Объем в окне
                    window_volume = window_trades['size'].sum()
                    df.loc[df.index[i], volume_col] = window_volume
                    
                    # Скорость торгов (объем в секунду)
                    window_duration_sec = window_ms / 1000
                    if window_duration_sec > 0:
                        df.loc[df.index[i], speed_col] = window_volume / window_duration_sec
                    
                    # Кумулятивный объем
                    if i > 0:
                        prev_cumulative = df.iloc[i-1][cumulative_col]
                        df.loc[df.index[i], cumulative_col] = prev_cumulative + df.iloc[i]['size']
                    else:
                        df.loc[df.index[i], cumulative_col] = df.iloc[i]['size']
        
        return df
    
    def _calculate_additional_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисление дополнительных метрик"""
        logger.info("Вычисление дополнительных метрик...")
        
        # Анализ по биржам
        exchanges = df['exchange'].unique()
        total_exchanges = len(exchanges)
        
        for i, exchange in enumerate(exchanges):
            if i % 3 == 0:  # Логируем каждые 3 биржи
                logger.info(f"Прогресс метрик: {i+1}/{total_exchanges} бирж")
                
            exchange_data = df[df['exchange'] == exchange].copy()
            
            if exchange_data.empty:
                continue
            
            # Вычисление метрик для биржи
            exchange_data = self._calculate_exchange_metrics(exchange_data)
            
            # Обновляем основной DataFrame
            metric_columns = ['vwap', 'trade_flow_imbalance', 'large_trades_ratio', 'volatility']
            for col in metric_columns:
                if col in exchange_data.columns:
                    df.loc[df['exchange'] == exchange, col] = exchange_data[col]
        
        return df
    
    def _calculate_exchange_metrics(self, exchange_data: pd.DataFrame) -> pd.DataFrame:
        """Вычисление метрик для конкретной биржи"""
        df = exchange_data.copy()
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['trade_value'] * df['size']).cumsum() / df['size'].cumsum()
        
        # Trade Flow Imbalance
        buy_volume = df[df['side'] == 'buy']['size'].cumsum()
        sell_volume = df[df['side'] == 'sell']['size'].cumsum()
        total_volume = buy_volume + sell_volume
        
        df['trade_flow_imbalance'] = (buy_volume - sell_volume) / total_volume
        df['trade_flow_imbalance'] = df['trade_flow_imbalance'].fillna(0)
        
        # Large Trades Ratio (сделки больше среднего)
        avg_trade_size = df['size'].mean()
        large_trades = df[df['size'] > avg_trade_size * 2]
        
        if len(df) > 0:
            df['large_trades_ratio'] = len(large_trades) / len(df)
        else:
            df['large_trades_ratio'] = 0.0
        
        # Волатильность (скользящее стандартное отклонение цены)
        df['volatility'] = df['price'].rolling(window=20).std()
        df['volatility'] = df['volatility'].fillna(0)
        
        return df

def run_trade_tape(
    trades_df: pd.DataFrame,
    config: Dict,
    output_dir: Path = Path("data/tape")
) -> pd.DataFrame:
    """Основная функция анализа ленты сделок"""
    logger.info("Запуск анализа ленты сделок...")
    
    # Создание анализатора
    analyzer = TradeTapeAnalyzer(config)
    
    # Полный анализ
    tape_df = analyzer.analyze_trade_tape(trades_df)
    
    # Сохранение результатов
    if not tape_df.empty:
        output_path = output_dir / "tape_analyzed.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        tape_df.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(f"Анализированная лента сделок сохранена: {len(tape_df)} строк в {output_path}")
        
        # Сохранение агрегированных данных
        save_aggregated_data(tape_df, output_dir)
    
    logger.info("Анализ ленты сделок завершен")
    return tape_df

def save_aggregated_data(tape_df: pd.DataFrame, output_dir: Path):
    """Сохранение агрегированных данных"""
    logger.info("Сохранение агрегированных данных...")
    
    # Агрегация по биржам
    exchange_summary = tape_df.groupby('exchange').agg({
        'size': ['sum', 'mean', 'count'],
        'trade_value': ['sum', 'mean'],
        'aggression_score': 'mean',
        'volume_short_1000ms': 'mean',
        'volume_medium_10000ms': 'mean',
        'volume_long_60000ms': 'mean'
    }).round(4)
    
    exchange_summary.to_parquet(output_dir / "exchange_summary.parquet", engine="pyarrow")
    
    # Агрегация по часам
    hourly_summary = tape_df.groupby(['exchange', 'date', 'hour']).agg({
        'size': 'sum',
        'trade_value': 'sum',
        'aggression_score': 'mean',
        'volatility': 'mean'
    }).round(4)
    
    hourly_summary.to_parquet(output_dir / "hourly_summary.parquet", engine="pyarrow")
    
    # Агрегация по типам агрессии
    aggression_summary = tape_df.groupby(['exchange', 'aggression_type']).agg({
        'size': 'sum',
        'trade_value': 'sum',
        'aggression_score': 'mean'
    }).round(4)
    
    aggression_summary.to_parquet(output_dir / "aggression_summary.parquet", engine="pyarrow")
    
    logger.info("Агрегированные данные сохранены")
