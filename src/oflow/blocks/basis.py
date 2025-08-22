"""
Блок 09: Базис спот–фьючерс
Вычислить разницу/процент/скользящие и пометки аномалий
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class BasisAnalyzer:
    """Анализатор базиса спот-фьючерс"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sync_window_ms = config.get('sync', {}).get('max_lag_ms', 5000)
        self.tick_size_ms = config.get('time', {}).get('tick_size_ms', 10)
        self.anomaly_threshold = config.get('anomaly', {}).get('threshold', 2.0)
        self.ma_windows = config.get('moving_averages', [5, 10, 20, 50])
        self.volatility_window = config.get('volatility', {}).get('window', 20)
        
    def run_basis_analysis(
        self,
        spot_df: pd.DataFrame,
        futures_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Полный анализ базиса спот-фьючерс"""
        start_time = time.time()
        logger.info("=== Начало анализа базиса спот-фьючерс ===")
        
        # Этап 1: Подготовка и синхронизация данных
        logger.info("Этап 1/4: Подготовка и синхронизация данных...")
        synced_spot, synced_futures = self._synchronize_data(spot_df, futures_df)
        
        # Этап 2: Вычисление базиса
        logger.info("Этап 2/4: Вычисление базиса...")
        basis_df = self._calculate_basis(synced_spot, synced_futures)
        
        # Этап 3: Обнаружение аномалий
        logger.info("Этап 3/4: Обнаружение аномалий...")
        basis_df = self._detect_anomalies(basis_df)
        
        # Этап 4: Дополнительные метрики
        logger.info("Этап 4/4: Дополнительные метрики...")
        basis_df = self._calculate_additional_metrics(basis_df)
        
        # Завершение
        duration = time.time() - start_time
        logger.info(f"Анализ завершен за {duration:.2f}с")
        logger.info(f"Базис вычислен: {len(basis_df)} записей")
        logger.info("=== Анализ базиса спот-фьючерс завершен ===")
        
        return basis_df
    
    def _synchronize_data(
        self,
        spot_df: pd.DataFrame,
        futures_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Синхронизация данных спот и фьючерс по времени"""
        logger.info("Синхронизация данных...")
        
        if spot_df.empty or futures_df.empty:
            logger.warning("Один из датафреймов пуст")
            return spot_df, futures_df
        
        # Проверка обязательных колонок
        required_columns = ['ts_ns', 'exchange', 'symbol', 'price']
        missing_spot = [col for col in required_columns if col not in spot_df.columns]
        missing_futures = [col for col in required_columns if col not in futures_df.columns]
        
        if missing_spot or missing_futures:
            logger.error(f"Отсутствуют колонки: spot={missing_spot}, futures={missing_futures}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Синхронизация по времени
        synced_spot = self._synchronize_quotes(spot_df, 'spot')
        synced_futures = self._synchronize_quotes(futures_df, 'futures')
        
        logger.info(f"Синхронизировано: spot={len(synced_spot)}, futures={len(synced_futures)}")
        return synced_spot, synced_futures
    
    def _synchronize_quotes(self, quotes_df: pd.DataFrame, market_type: str) -> pd.DataFrame:
        """Синхронизация котировок по времени"""
        df = quotes_df.copy()
        
        # Сортировка по времени
        df = df.sort_values('ts_ns').reset_index(drop=True)
        
        # Группировка по времени с учетом синхронизации
        df['ts_group'] = (df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
        
        # Агрегация по группам времени и биржам
        synced_quotes = df.groupby(['ts_group', 'exchange', 'symbol']).agg({
            'ts_ns': 'first',
            'price': 'last'  # Берем последнюю цену в группе
        }).reset_index()
        
        # Удаление временной группировки
        synced_quotes = synced_quotes.drop('ts_group', axis=1)
        
        # Добавление типа рынка
        synced_quotes['market_type'] = market_type
        
        return synced_quotes
    
    def _calculate_basis(
        self,
        spot_df: pd.DataFrame,
        futures_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Вычисление базиса спот-фьючерс"""
        logger.info("Вычисление базиса...")
        
        if spot_df.empty or futures_df.empty:
            return pd.DataFrame()
        
        # Создание временных групп для синхронизации
        spot_df = spot_df.copy()
        futures_df = futures_df.copy()
        
        spot_df['ts_group'] = (spot_df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
        futures_df['ts_group'] = (futures_df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
        
        # Объединение по временным группам
        basis_records = []
        total_groups = min(spot_df['ts_group'].nunique(), futures_df['ts_group'].nunique())
        
        for i, (ts_group, spot_group) in enumerate(spot_df.groupby('ts_group')):
            if i % 1000 == 0:  # Логируем каждые 1000 групп
                logger.info(f"Прогресс базиса: {i+1}/{total_groups} временных групп")
            
            # Поиск соответствующих фьючерсных данных
            futures_group = futures_df[futures_df['ts_group'] == ts_group]
            
            if futures_group.empty:
                continue
            
            # Вычисление базиса для каждой пары спот-фьючерс
            for _, spot_row in spot_group.iterrows():
                for _, futures_row in futures_group.iterrows():
                    # Проверка совпадения символов (базовый актив)
                    if not self._symbols_match(spot_row['symbol'], futures_row['symbol']):
                        continue
                    
                    spot_price = spot_row['price']
                    futures_price = futures_row['price']
                    
                    # Вычисление базиса
                    basis_abs = futures_price - spot_price
                    basis_rel = (basis_abs / spot_price) * 100 if spot_price > 0 else 0
                    
                    # Валидация базиса
                    if not self._validate_basis(basis_abs, basis_rel):
                        continue
                    
                    basis_records.append({
                        'ts_ns': spot_row['ts_ns'],
                        'ts_group': ts_group,
                        'spot_exchange': spot_row['exchange'],
                        'futures_exchange': futures_row['exchange'],
                        'symbol': spot_row['symbol'],
                        'spot_price': spot_price,
                        'futures_price': futures_price,
                        'basis_abs': basis_abs,
                        'basis_rel': basis_rel,
                        'spot_market': 'spot',
                        'futures_market': 'futures'
                    })
        
        basis_df = pd.DataFrame(basis_records)
        
        if not basis_df.empty:
            # Удаление временной группировки
            basis_df = basis_df.drop('ts_group', axis=1)
            
            # Сортировка по времени
            basis_df = basis_df.sort_values('ts_ns').reset_index(drop=True)
            
            logger.info(f"Базис вычислен: {len(basis_df)} записей")
        else:
            logger.warning("Базис не вычислен - нет валидных данных")
        
        return basis_df
    
    def _symbols_match(self, spot_symbol: str, futures_symbol: str) -> bool:
        """Проверка соответствия символов спот и фьючерс"""
        # Упрощенная логика: убираем суффиксы фьючерсов
        spot_base = spot_symbol.replace('-USDT', '').replace('USDT', '')
        futures_base = futures_symbol.replace('-PERP', '').replace('-SWAP', '').replace('-USDT', '').replace('USDT', '')
        
        return spot_base == futures_base
    
    def _validate_basis(self, basis_abs: float, basis_rel: float) -> bool:
        """Валидация вычисленного базиса"""
        # Проверка на разумные пределы
        max_basis_rel = self.config.get('basis', {}).get('max_basis_rel', 50.0)  # 50%
        min_basis_rel = self.config.get('basis', {}).get('min_basis_rel', -50.0)  # -50%
        
        return min_basis_rel <= basis_rel <= max_basis_rel
    
    def _detect_anomalies(self, basis_df: pd.DataFrame) -> pd.DataFrame:
        """Обнаружение аномалий в базисе"""
        logger.info("Обнаружение аномалий...")
        
        if basis_df.empty:
            return basis_df
        
        df = basis_df.copy()
        
        # Скользящие средние для базиса
        for window in self.ma_windows:
            df[f'basis_rel_ma_{window}'] = df['basis_rel'].rolling(window=window).mean()
            df[f'basis_abs_ma_{window}'] = df['basis_abs'].rolling(window=window).mean()
        
        # Волатильность базиса
        df['basis_volatility'] = df['basis_rel'].rolling(window=self.volatility_window).std()
        
        # Обнаружение аномалий
        df = self._calculate_anomaly_scores(df)
        
        # Пометки аномалий
        df = self._flag_anomalies(df)
        
        logger.info(f"Аномалии обнаружены: {df['anomaly_flag'].sum()} записей")
        return df
    
    def _calculate_anomaly_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисление скоринга аномалий"""
        # Базовый скоринг на основе отклонения от скользящего среднего
        df['anomaly_score'] = 0.0
        
        for window in self.ma_windows:
            ma_col = f'basis_rel_ma_{window}'
            if ma_col in df.columns:
                # Отклонение от MA
                deviation = abs(df['basis_rel'] - df[ma_col])
                ma_score = deviation / (df[ma_col].abs() + 1e-8)  # Избегаем деления на 0
                
                # Накопление скоринга
                df['anomaly_score'] += ma_score
        
        # Нормализация скоринга
        if df['anomaly_score'].max() > 0:
            df['anomaly_score'] = df['anomaly_score'] / df['anomaly_score'].max()
        
        # Дополнительные факторы
        if 'basis_volatility' in df.columns:
            # Скоринг на основе волатильности
            volatility_score = df['basis_volatility'] / (df['basis_volatility'].mean() + 1e-8)
            df['anomaly_score'] += volatility_score * 0.3  # Вес 30%
        
        # Ограничение скоринга
        df['anomaly_score'] = df['anomaly_score'].clip(0, 1)
        
        return df
    
    def _flag_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Пометка аномалий"""
        # Порог для пометки аномалий
        threshold = self.anomaly_threshold
        
        # Аномалии на основе скоринга
        df['anomaly_flag'] = df['anomaly_score'] > threshold
        
        # Аномалии на основе волатильности
        if 'basis_volatility' in df.columns:
            volatility_threshold = df['basis_volatility'].quantile(0.95)  # 95-й процентиль
            volatility_anomalies = df['basis_volatility'] > volatility_threshold
            df['anomaly_flag'] = df['anomaly_flag'] | volatility_anomalies
        
        # Аномалии на основе резких изменений
        if 'basis_rel' in df.columns:
            basis_change = df['basis_rel'].diff().abs()
            change_threshold = basis_change.quantile(0.95)  # 95-й процентиль
            change_anomalies = basis_change > change_threshold
            df['anomaly_flag'] = df['anomaly_flag'] | change_anomalies
        
        return df
    
    def _calculate_additional_metrics(self, basis_df: pd.DataFrame) -> pd.DataFrame:
        """Вычисление дополнительных метрик"""
        logger.info("Вычисление дополнительных метрик...")
        
        if basis_df.empty:
            return basis_df
        
        df = basis_df.copy()
        
        # Изменения базиса
        df['basis_rel_change'] = df['basis_rel'].diff()
        df['basis_rel_change_pct'] = (df['basis_rel_change'] / df['basis_rel'].shift(1)) * 100
        
        # Тренды базиса
        df['basis_trend'] = df['basis_rel'].rolling(window=10).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
        )
        
        # Конвергенция/дивергенция
        df['convergence'] = df.apply(
            lambda row: 1.0 if abs(row['basis_rel']) < abs(row.get('basis_rel_ma_20', 0)) else 0.0, axis=1
        )
        
        # Z-score базиса
        if 'basis_rel_ma_20' in df.columns and 'basis_volatility' in df.columns:
            df['basis_zscore'] = (df['basis_rel'] - df['basis_rel_ma_20']) / (df['basis_volatility'] + 1e-8)
        
        # Категоризация базиса
        df['basis_category'] = df['basis_rel'].apply(self._categorize_basis)
        
        return df
    
    def _categorize_basis(self, basis_rel: float) -> str:
        """Категоризация базиса"""
        if pd.isna(basis_rel):
            return 'unknown'
        
        if basis_rel > 5.0:
            return 'strong_contango'
        elif basis_rel > 1.0:
            return 'contango'
        elif basis_rel > -1.0:
            return 'near_par'
        elif basis_rel > -5.0:
            return 'backwardation'
        else:
            return 'strong_backwardation'

def run_basis(
    spot_df: pd.DataFrame,
    futures_df: pd.DataFrame,
    config: Dict,
    output_dir: Path = Path("data/basis")
) -> pd.DataFrame:
    """Основная функция анализа базиса"""
    logger.info("Запуск анализа базиса спот-фьючерс...")
    
    # Создание анализатора
    analyzer = BasisAnalyzer(config)
    
    # Полный анализ
    basis_df = analyzer.run_basis_analysis(spot_df, futures_df)
    
    # Сохранение результатов
    if not basis_df.empty:
        output_path = output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Основной файл базиса
        basis_path = output_path / "basis_analyzed.parquet"
        basis_df.to_parquet(basis_path, engine="pyarrow", index=False)
        logger.info(f"Базис сохранен: {len(basis_df)} строк в {basis_path}")
        
        # Сохранение агрегированных данных
        save_aggregated_data(basis_df, output_path)
    
    logger.info("Анализ базиса завершен")
    return basis_df

def save_aggregated_data(basis_df: pd.DataFrame, output_dir: Path):
    """Сохранение агрегированных данных"""
    logger.info("Сохранение агрегированных данных...")
    
    # Агрегация по биржам
    if not basis_df.empty:
        exchange_summary = basis_df.groupby(['spot_exchange', 'futures_exchange']).agg({
            'basis_rel': ['mean', 'std', 'min', 'max'],
            'basis_abs': ['mean', 'std', 'min', 'max'],
            'anomaly_score': 'mean',
            'anomaly_flag': 'sum'
        }).round(6)
        
        exchange_summary.to_parquet(output_dir / "exchange_basis_summary.parquet", engine="pyarrow")
    
    # Агрегация по категориям базиса
    if 'basis_category' in basis_df.columns:
        category_summary = basis_df.groupby('basis_category').agg({
            'basis_rel': ['mean', 'std', 'count'],
            'anomaly_score': 'mean',
            'anomaly_flag': 'sum'
        }).round(6)
        
        category_summary.to_parquet(output_dir / "category_basis_summary.parquet", engine="pyarrow")
    
    # Временная агрегация
    if not basis_df.empty:
        # Добавление временных меток
        basis_df['timestamp'] = pd.to_datetime(basis_df['ts_ns'], unit='ns')
        basis_df['hour'] = basis_df['timestamp'].dt.hour
        basis_df['date'] = basis_df['timestamp'].dt.date
        
        # Почасовая агрегация
        hourly_summary = basis_df.groupby(['date', 'hour']).agg({
            'basis_rel': ['mean', 'std', 'min', 'max'],
            'basis_abs': ['mean', 'std'],
            'anomaly_score': 'mean',
            'anomaly_flag': 'sum'
        }).round(6)
        
        hourly_summary.to_parquet(output_dir / "hourly_basis_summary.parquet", engine="pyarrow")
    
    logger.info("Агрегированные данные сохранены")
