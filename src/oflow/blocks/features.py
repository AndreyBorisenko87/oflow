"""
Блок 10: Фичи паттернов
Подготовка признаков для 8 детекторов паттернов (D1-D8)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class DetectorFeatureExtractor:
    """Извлекатель признаков для детекторов паттернов"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tick_size_ms = config.get('time', {}).get('tick_size_ms', 10)
        self.top_levels = config.get('orderbook', {}).get('top_levels', 10)
        
        # Пороги для детекторов
        self.d1_vacuum_threshold = config.get('d1_vacuum', {}).get('threshold', 0.1)
        self.d1_empty_levels = config.get('d1_vacuum', {}).get('empty_levels', 3)
        
        self.d2_side_ratio = config.get('d2_absorption', {}).get('min_side_ratio', 0.7)
        self.d2_price_move = config.get('d2_absorption', {}).get('max_price_move_ticks', 2)
        
        self.d3_same_price_trades = config.get('d3_iceberg', {}).get('same_price_trades', 10)
        self.d3_time_window = config.get('d3_iceberg', {}).get('time_window_ms', 2000)
        
        self.d4_price_move = config.get('d4_stop_run', {}).get('min_price_move_percent', 0.3)
        self.d4_time_window = config.get('d4_stop_run', {}).get('time_window_ms', 2000)
        
        self.d5_reversal_threshold = config.get('d5_false_breakout', {}).get('reversal_threshold_ratio', 0.7)
        self.d5_reversal_window = config.get('d5_false_breakout', {}).get('reversal_window_ms', 5000)
        
        self.d6_bid_ask_ratio = config.get('d6_imbalance', {}).get('bid_ask_ratio', 3.0)
        self.d6_wall_multiplier = config.get('d6_imbalance', {}).get('level_wall_multiplier', 5.0)
        
        self.d7_large_order_multiplier = config.get('d7_spoofing', {}).get('large_order_multiplier', 5.0)
        self.d7_cancel_window = config.get('d7_spoofing', {}).get('cancel_time_window_ms', 1000)
        
        self.d8_acceleration_window = config.get('d8_momentum', {}).get('acceleration_window_ms', 3000)
        self.d8_volume_ratio = config.get('d8_momentum', {}).get('min_volume_ratio', 3.0)
        
    def run_feature_extraction(
        self,
        book_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        quotes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Полное извлечение признаков для детекторов"""
        start_time = time.time()
        logger.info("=== Начало извлечения признаков для детекторов ===")
        
        # Этап 1: Подготовка и валидация данных
        logger.info("Этап 1/6: Подготовка и валидация данных...")
        validated_data = self._validate_and_prepare_data(book_df, trades_df, quotes_df)
        
        # Этап 2: Базовые признаки Order Flow
        logger.info("Этап 2/6: Базовые признаки Order Flow...")
        features_df = self._extract_order_flow_features(validated_data)
        
        # Этап 3: Признаки для D1 (Вакуум ликвидности)
        logger.info("Этап 3/6: Признаки для D1 (Вакуум ликвидности)...")
        features_df = self._extract_d1_vacuum_features(features_df, validated_data)
        
        # Этап 4: Признаки для D2-D4 (Поглощение, Айсберг, Стоп-ран)
        logger.info("Этап 4/6: Признаки для D2-D4 (Поглощение, Айсберг, Стоп-ран)...")
        features_df = self._extract_d2_d4_features(features_df, validated_data)
        
        # Этап 5: Признаки для D5-D7 (Ложный вынос, Имбаланс, Спуфинг)
        logger.info("Этап 5/6: Признаки для D5-D7 (Ложный вынос, Имбаланс, Спуфинг)...")
        features_df = self._extract_d5_d7_features(features_df, validated_data)
        
        # Этап 6: Признаки для D8 (Импульс)
        logger.info("Этап 6/6: Признаки для D8 (Импульс)...")
        features_df = self._extract_d8_momentum_features(features_df, validated_data)
        
        # Завершение
        duration = time.time() - start_time
        logger.info(f"Извлечение завершено за {duration:.2f}с")
        logger.info(f"Признаки извлечены: {len(features_df)} записей")
        logger.info("=== Извлечение признаков для детекторов завершено ===")
        
        return features_df
    
    def _validate_and_prepare_data(
        self,
        book_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        quotes_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Валидация и подготовка данных"""
        logger.info("Валидация и подготовка данных...")
        
        validated_data = {}
        
        # Валидация book_df
        if not book_df.empty:
            required_book_columns = ['ts_ns', 'exchange', 'symbol', 'side', 'price', 'size', 'action']
            missing_book = [col for col in required_book_columns if col not in book_df.columns]
            
            if not missing_book:
                book_df = book_df.copy()
                book_df = book_df.sort_values('ts_ns').reset_index(drop=True)
                book_df['ts_group'] = (book_df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
                validated_data['book'] = book_df
                logger.info(f"Book данные валидированы: {len(book_df)} записей")
            else:
                logger.warning(f"Book данные не прошли валидацию: отсутствуют {missing_book}")
                validated_data['book'] = pd.DataFrame()
        else:
            validated_data['book'] = pd.DataFrame()
        
        # Валидация trades_df
        if not trades_df.empty:
            required_trades_columns = ['ts_ns', 'exchange', 'symbol', 'price', 'size', 'side']
            missing_trades = [col for col in required_trades_columns if col not in trades_df.columns]
            
            if not missing_trades:
                trades_df = trades_df.copy()
                trades_df = trades_df.sort_values('ts_ns').reset_index(drop=True)
                trades_df['ts_group'] = (trades_df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
                validated_data['trades'] = trades_df
                logger.info(f"Trades данные валидированы: {len(trades_df)} записей")
            else:
                logger.warning(f"Trades данные не прошли валидацию: отсутствуют {missing_trades}")
                validated_data['trades'] = pd.DataFrame()
        else:
            validated_data['trades'] = pd.DataFrame()
        
        # Валидация quotes_df
        if not quotes_df.empty:
            required_quotes_columns = ['ts_ns', 'exchange', 'symbol', 'bid', 'ask', 'bid_size', 'ask_size']
            missing_quotes = [col for col in required_quotes_columns if col not in quotes_df.columns]
            
            if not missing_quotes:
                quotes_df = quotes_df.copy()
                quotes_df = quotes_df.sort_values('ts_ns').reset_index(drop=True)
                quotes_df['ts_group'] = (quotes_df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
                validated_data['quotes'] = quotes_df
                logger.info(f"Quotes данные валидированы: {len(quotes_df)} записей")
            else:
                logger.warning(f"Quotes данные не прошли валидацию: отсутствуют {missing_quotes}")
                validated_data['quotes'] = pd.DataFrame()
        else:
            validated_data['quotes'] = pd.DataFrame()
        
        return validated_data
    
    def _extract_order_flow_features(self, validated_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Извлечение базовых Order Flow признаков"""
        logger.info("Извлечение базовых Order Flow признаков...")
        
        if validated_data['book'].empty:
            logger.warning("Book данные пусты, создаем пустой DataFrame")
            return pd.DataFrame(columns=['ts_ns', 'ts_group', 'exchange', 'symbol'])
        
        book_df = validated_data['book']
        
        # Агрегация по временным группам
        order_flow_features = []
        total_groups = book_df['ts_group'].nunique()
        
        for i, (ts_group, group_data) in enumerate(book_df.groupby('ts_group')):
            if i % 1000 == 0:  # Логируем каждые 1000 групп
                logger.info(f"Прогресс order flow: {i+1}/{total_groups} временных групп")
            
            # Агрегация по биржам и символам
            for (exchange, symbol), exchange_data in group_data.groupby(['exchange', 'symbol']):
                # Подсчет действий
                cancellations = len(exchange_data[exchange_data['action'] == 'delete'])
                additions = len(exchange_data[exchange_data['action'] == 'add'])
                updates = len(exchange_data[exchange_data['action'] == 'update'])
                total_actions = len(exchange_data)
                
                # Статистика по сторонам
                bid_cancellations = len(exchange_data[
                    (exchange_data['action'] == 'delete') & 
                    (exchange_data['side'] == 'bid')
                ])
                ask_cancellations = len(exchange_data[
                    (exchange_data['action'] == 'delete') & 
                    (exchange_data['side'] == 'ask')
                ])
                
                bid_additions = len(exchange_data[
                    (exchange_data['action'] == 'add') & 
                    (exchange_data['side'] == 'bid')
                ])
                ask_additions = len(exchange_data[
                    (exchange_data['action'] == 'add') & 
                    (exchange_data['side'] == 'ask')
                ])
                
                # Вычисление признаков
                cancellation_rate = cancellations / total_actions if total_actions > 0 else 0
                addition_rate = additions / total_actions if total_actions > 0 else 0
                update_rate = updates / total_actions if total_actions > 0 else 0
                
                bid_ask_imbalance = (bid_additions - ask_additions) / (bid_additions + ask_additions + 1e-8)
                cancellation_imbalance = (bid_cancellations - ask_cancellations) / (bid_cancellations + ask_cancellations + 1e-8)
                
                # Объемные признаки
                total_volume = exchange_data['size'].sum()
                avg_order_size = exchange_data['size'].mean()
                volume_std = exchange_data['size'].std()
                
                order_flow_features.append({
                    'ts_ns': exchange_data['ts_ns'].iloc[0],
                    'ts_group': ts_group,
                    'exchange': exchange,
                    'symbol': symbol,
                    'cancellation_rate': cancellation_rate,
                    'addition_rate': addition_rate,
                    'update_rate': update_rate,
                    'bid_ask_imbalance': bid_ask_imbalance,
                    'cancellation_imbalance': cancellation_imbalance,
                    'total_volume': total_volume,
                    'avg_order_size': avg_order_size,
                    'volume_std': volume_std,
                    'total_actions': total_actions,
                    'cancellations': cancellations,
                    'additions': additions,
                    'updates': updates
                })
        
        features_df = pd.DataFrame(order_flow_features)
        
        if not features_df.empty:
            logger.info(f"Order flow признаки извлечены: {len(features_df)} записей")
        else:
            logger.warning("Order flow признаки не извлечены")
        
        return features_df
    
    def _extract_d1_vacuum_features(
        self,
        features_df: pd.DataFrame,
        validated_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Признаки для D1: Вакуум ликвидности"""
        logger.info("Извлечение признаков для D1 (Вакуум ликвидности)...")
        
        if validated_data['quotes'].empty:
            logger.warning("Quotes данные пусты, пропускаем D1")
            features_df['d1_vacuum_score'] = 0.0
            features_df['d1_empty_levels'] = 0
            features_df['d1_liquidity_depth'] = 0.0
            return features_df
        
        quotes_df = validated_data['quotes']
        
        # Создание временных групп для features_df если их нет
        if 'ts_group' not in features_df.columns:
            features_df['ts_group'] = (features_df['ts_ns'] // (self.tick_size_ms * 1_000_000)).astype(int)
        
        # Анализ вакуума по временным группам
        vacuum_features = []
        
        for ts_group, group_data in features_df.groupby('ts_group'):
            # Поиск соответствующих quotes данных
            quotes_group = quotes_df[quotes_df['ts_group'] == ts_group]
            
            if quotes_group.empty:
                continue
            
            for _, feature_row in group_data.iterrows():
                exchange = feature_row['exchange']
                symbol = feature_row['symbol']
                
                # Фильтрация quotes по бирже и символу
                exchange_quotes = quotes_group[
                    (quotes_group['exchange'] == exchange) & 
                    (quotes_group['symbol'] == symbol)
                ]
                
                if exchange_quotes.empty:
                    vacuum_features.append({
                        'ts_group': ts_group,
                        'exchange': exchange,
                        'symbol': symbol,
                        'd1_vacuum_score': 0.0,
                        'd1_empty_levels': 0,
                        'd1_liquidity_depth': 0.0
                    })
                    continue
                
                # Анализ глубины ликвидности
                bid_depth = exchange_quotes['bid_size'].sum()
                ask_depth = exchange_quotes['ask_size'].sum()
                total_depth = bid_depth + ask_depth
                
                # Определение пустых уровней
                empty_bid_levels = len(exchange_quotes[exchange_quotes['bid_size'] < self.d1_vacuum_threshold])
                empty_ask_levels = len(exchange_quotes[exchange_quotes['ask_size'] < self.d1_vacuum_threshold])
                total_empty_levels = empty_bid_levels + empty_ask_levels
                
                # Скоринг вакуума для D1
                d1_vacuum_score = 0.0
                if total_empty_levels >= self.d1_empty_levels:
                    # Нормализованный скоринг на основе пустых уровней и глубины
                    empty_ratio = total_empty_levels / (len(exchange_quotes) * 2)  # bid + ask
                    depth_ratio = 1.0 / (1.0 + total_depth / 1000)  # Нормализация глубины
                    d1_vacuum_score = (empty_ratio * 0.7 + depth_ratio * 0.3)
                
                vacuum_features.append({
                    'ts_group': ts_group,
                    'exchange': exchange,
                    'symbol': symbol,
                    'd1_vacuum_score': d1_vacuum_score,
                    'd1_empty_levels': total_empty_levels,
                    'd1_liquidity_depth': total_depth
                })
        
        # Объединение с основными признаками
        vacuum_df = pd.DataFrame(vacuum_features)
        if not vacuum_df.empty:
            features_df = features_df.merge(
                vacuum_df[['ts_group', 'exchange', 'symbol', 'd1_vacuum_score', 'd1_empty_levels', 'd1_liquidity_depth']],
                on=['ts_group', 'exchange', 'symbol'],
                how='left'
            )
            
            # Заполнение пропущенных значений
            features_df['d1_vacuum_score'] = features_df['d1_vacuum_score'].fillna(0.0)
            features_df['d1_empty_levels'] = features_df['d1_empty_levels'].fillna(0)
            features_df['d1_liquidity_depth'] = features_df['d1_liquidity_depth'].fillna(0.0)
            
            logger.info("Признаки для D1 (Вакуум ликвидности) добавлены")
        else:
            # Если вакуум не извлечен, добавляем пустые колонки
            features_df['d1_vacuum_score'] = 0.0
            features_df['d1_empty_levels'] = 0
            features_df['d1_liquidity_depth'] = 0.0
        
        return features_df
    
    def _extract_d2_d4_features(
        self,
        features_df: pd.DataFrame,
        validated_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Признаки для D2-D4: Поглощение, Айсберг, Стоп-ран"""
        logger.info("Извлечение признаков для D2-D4...")
        
        if validated_data['trades'].empty:
            logger.warning("Trades данные пусты, пропускаем D2-D4")
            # Добавляем пустые колонки для D2-D4
            features_df['d2_side_ratio'] = 0.0
            features_df['d2_price_stable'] = False
            features_df['d3_same_price_count'] = 0
            features_df['d3_volume_concentration'] = 0.0
            features_df['d4_price_momentum'] = 0.0
            features_df['d4_volume_surge'] = 0.0
            return features_df
        
        trades_df = validated_data['trades']
        
        # Анализ по временным группам
        d2_d4_features = []
        
        for ts_group, group_data in features_df.groupby('ts_group'):
            # Поиск соответствующих trades данных
            trades_group = trades_df[trades_df['ts_group'] == ts_group]
            
            if trades_group.empty:
                continue
            
            for _, feature_row in group_data.iterrows():
                exchange = feature_row['exchange']
                symbol = feature_row['symbol']
                
                # Фильтрация trades по бирже и символу
                exchange_trades = trades_group[
                    (trades_group['exchange'] == exchange) & 
                    (trades_group['symbol'] == symbol)
                ]
                
                if exchange_trades.empty:
                    d2_d4_features.append({
                        'ts_group': ts_group,
                        'exchange': exchange,
                        'symbol': symbol,
                        'd2_side_ratio': 0.0,
                        'd2_price_stable': False,
                        'd3_same_price_count': 0,
                        'd3_volume_concentration': 0.0,
                        'd4_price_momentum': 0.0,
                        'd4_volume_surge': 0.0
                    })
                    continue
                
                # D2: Признаки поглощения
                buy_trades = len(exchange_trades[exchange_trades['side'] == 'buy'])
                sell_trades = len(exchange_trades[exchange_trades['side'] == 'sell'])
                total_trades = len(exchange_trades)
                
                d2_side_ratio = max(buy_trades, sell_trades) / total_trades if total_trades > 0 else 0
                d2_price_stable = len(exchange_trades['price'].unique()) <= self.d2_price_move
                
                # D3: Признаки айсберга
                price_counts = exchange_trades['price'].value_counts()
                d3_same_price_count = price_counts.max() if not price_counts.empty else 0
                d3_volume_concentration = d3_same_price_count / total_trades if total_trades > 0 else 0
                
                # D4: Признаки стоп-рана
                if len(exchange_trades) > 1:
                    price_changes = exchange_trades['price'].pct_change().abs()
                    d4_price_momentum = price_changes.mean()
                    
                    volume_surge = exchange_trades['size'].sum() / (self.d4_time_window / 1000) if self.d4_time_window > 0 else 0
                    d4_volume_surge = min(volume_surge / 1000, 1.0)  # Нормализация
                else:
                    d4_price_momentum = 0.0
                    d4_volume_surge = 0.0
                
                d2_d4_features.append({
                    'ts_group': ts_group,
                    'exchange': exchange,
                    'symbol': symbol,
                    'd2_side_ratio': d2_side_ratio,
                    'd2_price_stable': d2_price_stable,
                    'd3_same_price_count': d3_same_price_count,
                    'd3_volume_concentration': d3_volume_concentration,
                    'd4_price_momentum': d4_price_momentum,
                    'd4_volume_surge': d4_volume_surge
                })
        
        # Объединение с основными признаками
        d2_d4_df = pd.DataFrame(d2_d4_features)
        if not d2_d4_df.empty:
            features_df = features_df.merge(
                d2_d4_df[['ts_group', 'exchange', 'symbol', 'd2_side_ratio', 'd2_price_stable', 
                          'd3_same_price_count', 'd3_volume_concentration', 'd4_price_momentum', 'd4_volume_surge']],
                on=['ts_group', 'exchange', 'symbol'],
                how='left'
            )
            
            # Заполнение пропущенных значений
            for col in ['d2_side_ratio', 'd3_volume_concentration', 'd4_price_momentum', 'd4_volume_surge']:
                features_df[col] = features_df[col].fillna(0.0)
            features_df['d2_price_stable'] = features_df['d2_price_stable'].fillna(False)
            features_df['d3_same_price_count'] = features_df['d3_same_price_count'].fillna(0)
            
            logger.info("Признаки для D2-D4 добавлены")
        else:
            # Если признаки не извлечены, добавляем пустые колонки
            features_df['d2_side_ratio'] = 0.0
            features_df['d2_price_stable'] = False
            features_df['d3_same_price_count'] = 0
            features_df['d3_volume_concentration'] = 0.0
            features_df['d4_price_momentum'] = 0.0
            features_df['d4_volume_surge'] = 0.0
        
        return features_df
    
    def _extract_d5_d7_features(
        self,
        features_df: pd.DataFrame,
        validated_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Признаки для D5-D7: Ложный вынос, Имбаланс, Спуфинг"""
        logger.info("Извлечение признаков для D5-D7...")
        
        if validated_data['book'].empty:
            logger.warning("Book данные пусты, пропускаем D5-D7")
            # Добавляем пустые колонки для D5-D7
            features_df['d5_reversal_probability'] = 0.0
            features_df['d6_bid_ask_imbalance'] = 0.0
            features_df['d6_level_wall'] = 0.0
            features_df['d7_spoofing_score'] = 0.0
            features_df['d7_quick_cancel'] = False
            return features_df
        
        book_df = validated_data['book']
        
        # Анализ по временным группам
        d5_d7_features = []
        
        for ts_group, group_data in features_df.groupby('ts_group'):
            # Поиск соответствующих book данных
            book_group = book_df[book_df['ts_group'] == ts_group]
            
            if book_group.empty:
                continue
            
            for _, feature_row in group_data.iterrows():
                exchange = feature_row['exchange']
                symbol = feature_row['symbol']
                
                # Фильтрация book по бирже и символу
                exchange_book = book_group[
                    (book_group['exchange'] == exchange) & 
                    (book_group['symbol'] == symbol)
                ]
                
                if exchange_book.empty:
                    d5_d7_features.append({
                        'ts_group': ts_group,
                        'exchange': exchange,
                        'symbol': symbol,
                        'd5_reversal_probability': 0.0,
                        'd6_bid_ask_imbalance': 0.0,
                        'd6_level_wall': 0.0,
                        'd7_spoofing_score': 0.0,
                        'd7_quick_cancel': False
                    })
                    continue
                
                # D5: Признаки ложного выноса (упрощенно)
                d5_reversal_probability = 0.0
                if len(exchange_book) > 1:
                    # Анализ отмен vs добавлений
                    cancellations = len(exchange_book[exchange_book['action'] == 'delete'])
                    additions = len(exchange_book[exchange_book['action'] == 'add'])
                    total_actions = len(exchange_book)
                    
                    if total_actions > 0:
                        d5_reversal_probability = cancellations / total_actions
                
                # D6: Признаки имбаланса стакана
                bid_volume = exchange_book[exchange_book['side'] == 'bid']['size'].sum()
                ask_volume = exchange_book[exchange_book['side'] == 'ask']['size'].sum()
                
                d6_bid_ask_imbalance = 0.0
                if ask_volume > 0:
                    d6_bid_ask_imbalance = bid_volume / ask_volume
                
                # Анализ стен на уровнях
                level_volumes = exchange_book.groupby(['side', 'price'])['size'].sum()
                d6_level_wall = level_volumes.max() / (level_volumes.mean() + 1e-8) if not level_volumes.empty else 0
                
                # D7: Признаки спуфинга
                d7_spoofing_score = 0.0
                d7_quick_cancel = False
                
                if len(exchange_book) > 1:
                    # Анализ быстрых отмен
                    add_actions = exchange_book[exchange_book['action'] == 'add']
                    if not add_actions.empty:
                        for _, add_row in add_actions.iterrows():
                            # Поиск быстрых отмен
                            cancel_actions = exchange_book[
                                (exchange_book['action'] == 'delete') &
                                (exchange_book['side'] == add_row['side']) &
                                (exchange_book['price'] == add_row['price'])
                            ]
                            
                            if not cancel_actions.empty:
                                time_diff = cancel_actions['ts_ns'].iloc[0] - add_row['ts_ns']
                                if time_diff <= self.d7_cancel_window * 1_000_000:  # конвертация в наносекунды
                                    d7_quick_cancel = True
                                    d7_spoofing_score = 1.0
                                    break
                
                d5_d7_features.append({
                    'ts_group': ts_group,
                    'exchange': exchange,
                    'symbol': symbol,
                    'd5_reversal_probability': d5_reversal_probability,
                    'd6_bid_ask_imbalance': d6_bid_ask_imbalance,
                    'd6_level_wall': d6_level_wall,
                    'd7_spoofing_score': d7_spoofing_score,
                    'd7_quick_cancel': d7_quick_cancel
                })
        
        # Объединение с основными признаками
        d5_d7_df = pd.DataFrame(d5_d7_features)
        if not d5_d7_df.empty:
            features_df = features_df.merge(
                d5_d7_df[['ts_group', 'exchange', 'symbol', 'd5_reversal_probability', 'd6_bid_ask_imbalance', 
                          'd6_level_wall', 'd7_spoofing_score', 'd7_quick_cancel']],
                on=['ts_group', 'exchange', 'symbol'],
                how='left'
            )
            
            # Заполнение пропущенных значений
            features_df['d5_reversal_probability'] = features_df['d5_reversal_probability'].fillna(0.0)
            features_df['d6_bid_ask_imbalance'] = features_df['d6_bid_ask_imbalance'].fillna(0.0)
            features_df['d6_level_wall'] = features_df['d6_level_wall'].fillna(0.0)
            features_df['d7_spoofing_score'] = features_df['d7_spoofing_score'].fillna(0.0)
            features_df['d7_quick_cancel'] = features_df['d7_quick_cancel'].fillna(False)
            
            logger.info("Признаки для D5-D7 добавлены")
        else:
            # Если признаки не извлечены, добавляем пустые колонки
            features_df['d5_reversal_probability'] = 0.0
            features_df['d6_bid_ask_imbalance'] = 0.0
            features_df['d6_level_wall'] = 0.0
            features_df['d7_spoofing_score'] = 0.0
            features_df['d7_quick_cancel'] = False
        
        return features_df
    
    def _extract_d8_momentum_features(
        self,
        features_df: pd.DataFrame,
        validated_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Признаки для D8: Импульс"""
        logger.info("Извлечение признаков для D8 (Импульс)...")
        
        if validated_data['trades'].empty:
            logger.warning("Trades данные пусты, пропускаем D8")
            features_df['d8_momentum_score'] = 0.0
            features_df['d8_acceleration'] = 0.0
            features_df['d8_volume_surge'] = 0.0
            return features_df
        
        trades_df = validated_data['trades']
        
        # Анализ по временным группам
        momentum_features = []
        
        for ts_group, group_data in features_df.groupby('ts_group'):
            # Поиск соответствующих trades данных
            trades_group = trades_df[trades_df['ts_group'] == ts_group]
            
            if trades_group.empty:
                continue
            
            for _, feature_row in group_data.iterrows():
                exchange = feature_row['exchange']
                symbol = feature_row['symbol']
                
                # Фильтрация trades по бирже и символу
                exchange_trades = trades_group[
                    (trades_group['exchange'] == exchange) & 
                    (trades_group['symbol'] == symbol)
                ]
                
                if exchange_trades.empty:
                    momentum_features.append({
                        'ts_group': ts_group,
                        'exchange': exchange,
                        'symbol': symbol,
                        'd8_momentum_score': 0.0,
                        'd8_acceleration': 0.0,
                        'd8_volume_surge': 0.0
                    })
                    continue
                
                # D8: Признаки импульса
                d8_momentum_score = 0.0
                d8_acceleration = 0.0
                d8_volume_surge = 0.0
                
                if len(exchange_trades) > 1:
                    # Анализ ускорения цены
                    price_changes = exchange_trades['price'].pct_change()
                    if len(price_changes) > 1:
                        d8_acceleration = price_changes.diff().mean()
                    
                    # Анализ всплеска объема
                    volume_per_second = exchange_trades['size'].sum() / (self.d8_acceleration_window / 1000) if self.d8_acceleration_window > 0 else 0
                    d8_volume_surge = min(volume_per_second / 1000, 1.0)  # Нормализация
                    
                    # Общий скоринг импульса
                    d8_momentum_score = (abs(d8_acceleration) * 0.4 + d8_volume_surge * 0.6)
                
                momentum_features.append({
                    'ts_group': ts_group,
                    'exchange': exchange,
                    'symbol': symbol,
                    'd8_momentum_score': d8_momentum_score,
                    'd8_acceleration': d8_acceleration,
                    'd8_volume_surge': d8_volume_surge
                })
        
        # Объединение с основными признаками
        momentum_df = pd.DataFrame(momentum_features)
        if not momentum_df.empty:
            features_df = features_df.merge(
                momentum_df[['ts_group', 'exchange', 'symbol', 'd8_momentum_score', 'd8_acceleration', 'd8_volume_surge']],
                on=['ts_group', 'exchange', 'symbol'],
                how='left'
            )
            
            # Заполнение пропущенных значений
            features_df['d8_momentum_score'] = features_df['d8_momentum_score'].fillna(0.0)
            features_df['d8_acceleration'] = features_df['d8_acceleration'].fillna(0.0)
            features_df['d8_volume_surge'] = features_df['d8_volume_surge'].fillna(0.0)
            
            logger.info("Признаки для D8 (Импульс) добавлены")
        else:
            # Если признаки не извлечены, добавляем пустые колонки
            features_df['d8_momentum_score'] = 0.0
            features_df['d8_acceleration'] = 0.0
            features_df['d8_volume_surge'] = 0.0
        
        return features_df

def run_features(
    book_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    config: Dict,
    output_dir: Path = Path("data/features")
) -> pd.DataFrame:
    """Основная функция извлечения признаков для детекторов"""
    logger.info("Запуск извлечения признаков для детекторов...")
    
    # Создание извлекателя
    extractor = DetectorFeatureExtractor(config)
    
    # Полное извлечение признаков
    features_df = extractor.run_feature_extraction(book_df, trades_df, quotes_df)
    
    # Вычисление скорингов для детекторов
    if not features_df.empty:
        features_df = calculate_detector_scores(features_df, config)
        
        # Сохранение результатов
        output_path = output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Основной файл признаков
        features_path = output_path / "detector_features.parquet"
        features_df.to_parquet(features_path, engine="pyarrow", index=False)
        logger.info(f"Признаки для детекторов сохранены: {len(features_df)} строк в {features_path}")
        
        # Сохранение агрегированных данных
        save_detector_features_summary(features_df, output_path)
    
    logger.info("Извлечение признаков для детекторов завершено")
    return features_df

def calculate_detector_scores(features_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Вычисление скорингов для детекторов"""
    logger.info("Вычисление скорингов для детекторов...")
    
    if features_df.empty:
        return features_df
    
    df = features_df.copy()
    
    # Нормализация признаков
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col not in ['ts_ns', 'ts_group']:
            df[f'{col}_normalized'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    
    # Скоринг для каждого детектора
    df['d1_score'] = df['d1_vacuum_score']
    df['d2_score'] = df['d2_side_ratio'] * df['d2_price_stable'].astype(int)
    df['d3_score'] = df['d3_volume_concentration']
    df['d4_score'] = df['d4_price_momentum'] * df['d4_volume_surge']
    df['d5_score'] = df['d5_reversal_probability']
    df['d6_score'] = (df['d6_bid_ask_imbalance'] > config.get('d6_imbalance', {}).get('bid_ask_ratio', 3.0)).astype(int) * df['d6_level_wall']
    df['d7_score'] = df['d7_spoofing_score'] * df['d7_quick_cancel'].astype(int)
    df['d8_score'] = df['d8_momentum_score']
    
    # Общий скоринг всех детекторов
    df['overall_detector_score'] = (
        df['d1_score'] + df['d2_score'] + df['d3_score'] + df['d4_score'] +
        df['d5_score'] + df['d6_score'] + df['d7_score'] + df['d8_score']
    ) / 8.0
    
    # Классификация по детекторам
    df['primary_detector'] = df[['d1_score', 'd2_score', 'd3_score', 'd4_score', 
                                 'd5_score', 'd6_score', 'd7_score', 'd8_score']].idxmax(axis=1)
    
    # Очистка названий детекторов
    df['primary_detector'] = df['primary_detector'].str.replace('_score', '')
    
    logger.info(f"Скоринги для детекторов вычислены")
    return df

def save_detector_features_summary(features_df: pd.DataFrame, output_dir: Path):
    """Сохранение сводки признаков детекторов"""
    logger.info("Сохранение сводки признаков детекторов...")
    
    # Агрегация по биржам
    if not features_df.empty:
        exchange_summary = features_df.groupby('exchange').agg({
            'd1_score': 'mean',
            'd2_score': 'mean',
            'd3_score': 'mean',
            'd4_score': 'mean',
            'd5_score': 'mean',
            'd6_score': 'mean',
            'd7_score': 'mean',
            'd8_score': 'mean',
            'overall_detector_score': 'mean'
        }).round(6)
        
        exchange_summary.to_parquet(output_dir / "detector_exchange_summary.parquet", engine="pyarrow")
    
    # Агрегация по основным детекторам
    if 'primary_detector' in features_df.columns:
        detector_summary = features_df.groupby('primary_detector').agg({
            'overall_detector_score': ['mean', 'std', 'count'],
            'd1_score': 'mean',
            'd2_score': 'mean',
            'd3_score': 'mean',
            'd4_score': 'mean',
            'd5_score': 'mean',
            'd6_score': 'mean',
            'd7_score': 'mean',
            'd8_score': 'mean'
        }).round(6)
        
        detector_summary.to_parquet(output_dir / "detector_pattern_summary.parquet", engine="pyarrow")
    
    # Временная агрегация
    if not features_df.empty:
        # Добавление временных меток
        features_df['timestamp'] = pd.to_datetime(features_df['ts_ns'], unit='ns')
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['date'] = features_df['timestamp'].dt.date
        
        # Почасовая агрегация
        hourly_summary = features_df.groupby(['date', 'hour']).agg({
            'd1_score': 'mean',
            'd2_score': 'mean',
            'd3_score': 'mean',
            'd4_score': 'mean',
            'd5_score': 'mean',
            'd6_score': 'mean',
            'd7_score': 'mean',
            'd8_score': 'mean',
            'overall_detector_score': 'mean'
        }).round(6)
        
        hourly_summary.to_parquet(output_dir / "detector_hourly_summary.parquet", engine="pyarrow")
    
    logger.info("Сводка признаков детекторов сохранена")
