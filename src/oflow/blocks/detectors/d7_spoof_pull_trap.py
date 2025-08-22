"""
Детектор D7 — Спуфинг ловушка
Суть: крупная заявка размещается и быстро снимается для манипуляции.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_detector import BaseDetector
import logging

class D7SpoofPullTrap(BaseDetector):
    """Детектор спуфинг ловушки"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.large_order_multiplier = config.get('large_order_multiplier', 5.0)
        self.distance_ticks = config.get('distance_ticks', 2)
        self.cancel_time_window_ms = config.get('cancel_time_window_ms', 1000)
        self.no_trade_executed = config.get('no_trade_executed', True)
        
    def detect(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Детекция спуфинг ловушки"""
        self.logger.info("Детекция спуфинг ловушки...")
        
        if not self.validate_data(data):
            return pd.DataFrame()
            
        book_top = data['book_top']
        quotes = data['quotes']
        tape = data['tape']
        
        events = []
        
        # Анализ по временным окнам
        exchanges = book_top['exchange'].unique()
        total_exchanges = len(exchanges)
        self.logger.info(f"Обрабатываем {total_exchanges} бирж...")
        
        for i, exchange in enumerate(exchanges):
            if i % 5 == 0:  # Логируем каждые 5 бирж
                self.logger.info(f"Прогресс: {i+1}/{total_exchanges} бирж")
                
            exchange_book = book_top[book_top['exchange'] == exchange].copy()
            exchange_quotes = quotes[quotes['exchange'] == exchange].copy()
            exchange_tape = tape[tape['exchange'] == exchange].copy()
            
            if exchange_book.empty or exchange_quotes.empty:
                continue
                
            # Группировка по временным окнам (1 секунда)
            time_window_ms = 1000
            exchange_book['ts_group'] = exchange_book['ts_ns'] // (time_window_ms * 1_000_000)
            exchange_quotes['ts_group'] = exchange_quotes['ts_ns'] // (time_window_ms * 1_000_000)
            
            time_groups = exchange_book['ts_group'].unique()
            total_groups = len(time_groups)
            
            for j, ts_group in enumerate(time_groups):
                if j % 100 == 0 and j > 0:  # Логируем каждые 100 временных групп
                    self.logger.debug(f"Биржа {exchange}: {j+1}/{total_groups} временных групп")
                    
                # Анализ в окне
                window_book = exchange_book[exchange_book['ts_group'] == ts_group]
                window_quotes = exchange_quotes[exchange_quotes['ts_group'] == ts_group]
                
                if window_book.empty or window_quotes.empty:
                    continue
                    
                # Поиск крупных заявок
                large_orders = self._find_large_orders(window_book)
                
                for order in large_orders:
                    # Проверка расстояния от рынка
                    market_distance = self._check_market_distance(order, window_quotes)
                    
                    if market_distance['within_range']:
                        # Проверка быстрого снятия
                        cancellation_analysis = self._check_quick_cancellation(
                            exchange_book, 
                            order['ts_ns'], 
                            order['price'], 
                            order['side']
                        )
                        
                        if cancellation_analysis['quick_cancel']:
                            # Проверка отсутствия исполнения
                            execution_check = self._check_no_execution(
                                exchange_tape, 
                                order['ts_ns'], 
                                order['price'], 
                                order['side']
                            )
                            
                            if execution_check['no_execution'] or not self.no_trade_executed:
                                event = {
                                    'ts_ns': order['ts_ns'],
                                    'exchange': exchange,
                                    'pattern_type': 'spoof_pull_trap',
                                    'side': order['side'],
                                    'price': order['price'],
                                    'size': order['size'],
                                    'size_multiplier': order['size_multiplier'],
                                    'market_distance_ticks': market_distance['distance_ticks'],
                                    'cancel_time_ms': cancellation_analysis['cancel_time_ms'],
                                    'no_execution': execution_check['no_execution'],
                                    'confidence': self.get_confidence_score({
                                        'size_multiplier': order['size_multiplier'],
                                        'market_distance_ticks': market_distance['distance_ticks'],
                                        'cancel_time_ms': cancellation_analysis['cancel_time_ms']
                                    }),
                                    'metadata': {
                                        'order_details': order,
                                        'cancellation_details': cancellation_analysis,
                                        'execution_details': execution_check
                                    }
                                }
                                events.append(event)
        
        if not events:
            return pd.DataFrame()
            
        events_df = pd.DataFrame(events)
        events_df['detector'] = self.name
        events_df['ts_ns'] = pd.to_datetime(events_df['ts_ns'], unit='ns')
        
        return events_df
    
    def _find_large_orders(self, window_book: pd.DataFrame) -> List[Dict]:
        """Поиск крупных заявок"""
        if window_book.empty:
            return []
        
        large_orders = []
        
        # Вычисление средней заявки
        all_sizes = window_book['size'].values
        if len(all_sizes) == 0:
            return []
        
        avg_size = np.mean(all_sizes)
        
        # Поиск заявок значительно больше среднего
        for _, order in window_book.iterrows():
            size_multiplier = order['size'] / avg_size if avg_size > 0 else 0
            
            if size_multiplier >= self.large_order_multiplier:
                large_order = {
                    'ts_ns': order['ts_ns'],
                    'side': order['side'],
                    'price': order['price'],
                    'size': order['size'],
                    'size_multiplier': size_multiplier,
                    'level': order['level']
                }
                large_orders.append(large_order)
        
        return large_orders
    
    def _check_market_distance(self, order: Dict, window_quotes: pd.DataFrame) -> Dict:
        """Проверка расстояния от рынка"""
        if window_quotes.empty:
            return {'within_range': False, 'distance_ticks': 0}
        
        # Определение текущей рыночной цены
        current_mid = window_quotes['mid_price'].iloc[-1]
        
        # Вычисление расстояния в тиках
        price_diff = abs(order['price'] - current_mid)
        tick_size = 0.01  # Предполагаем тик = 0.01 для ETHUSDT
        distance_ticks = int(price_diff / tick_size)
        
        within_range = distance_ticks <= self.distance_ticks
        
        return {
            'within_range': within_range,
            'distance_ticks': distance_ticks,
            'market_price': current_mid
        }
    
    def _check_quick_cancellation(self, all_book: pd.DataFrame, order_ts: int, order_price: float, order_side: str) -> Dict:
        """Проверка быстрого снятия заявки"""
        # Фильтрация изменений после размещения заявки
        future_changes = all_book[all_book['ts_ns'] > order_ts].copy()
        
        if future_changes.empty:
            return {'quick_cancel': False, 'cancel_time_ms': 0}
        
        # Поиск изменений по той же цене и стороне
        price_side_changes = future_changes[
            (future_changes['price'] == order_price) & 
            (future_changes['side'] == order_side)
        ].copy()
        
        if price_side_changes.empty:
            return {'quick_cancel': False, 'cancel_time_ms': 0}
        
        # Сортировка по времени
        price_side_changes = price_side_changes.sort_values('ts_ns')
        
        # Анализ изменений размера
        for _, change in price_side_changes.iterrows():
            # Если размер уменьшился или стал 0 - это снятие
            if change['size'] < order_price or change['size'] == 0:
                cancel_time_ms = (change['ts_ns'] - order_ts) // 1_000_000
                
                quick_cancel = cancel_time_ms <= self.cancel_time_window_ms
                
                return {
                    'quick_cancel': quick_cancel,
                    'cancel_time_ms': cancel_time_ms,
                    'cancel_ts': change['ts_ns'],
                    'remaining_size': change['size']
                }
        
        return {'quick_cancel': False, 'cancel_time_ms': 0}
    
    def _check_no_execution(self, all_tape: pd.DataFrame, order_ts: int, order_price: float, order_side: str) -> Dict:
        """Проверка отсутствия исполнения заявки"""
        # Фильтрация сделок после размещения заявки
        future_trades = all_tape[all_tape['ts_ns'] > order_ts].copy()
        
        if future_trades.empty:
            return {'no_execution': True, 'executed_volume': 0}
        
        # Поиск сделок по той же цене и стороне
        relevant_trades = future_trades[
            (future_trades['price'] == order_price) & 
            (future_trades['side'] == order_side)
        ].copy()
        
        if relevant_trades.empty:
            return {'no_execution': True, 'executed_volume': 0}
        
        # Подсчет исполненного объема
        executed_volume = relevant_trades['size'].sum()
        
        # Если объем исполнения мал по сравнению с исходной заявкой
        no_execution = executed_volume < order_price * 0.1  # Менее 10% от исходной заявки
        
        return {
            'no_execution': no_execution,
            'executed_volume': executed_volume,
            'execution_ratio': executed_volume / order_price if order_price > 0 else 0
        }
    
    def get_confidence_score(self, pattern_data: Dict) -> float:
        """Скоринг уверенности для спуфинг ловушки"""
        base_score = 0.3
        
        # Чем больше заявка, тем выше уверенность
        if 'size_multiplier' in pattern_data:
            size_score = min(pattern_data['size_multiplier'] / self.large_order_multiplier, 1.0)
            base_score += size_score * 0.3
            
        # Чем ближе к рынку, тем выше уверенность
        if 'market_distance_ticks' in pattern_data:
            distance_score = 1.0 - min(pattern_data['market_distance_ticks'] / self.distance_ticks, 1.0)
            base_score += distance_score * 0.2
            
        # Чем быстрее снятие, тем выше уверенность
        if 'cancel_time_ms' in pattern_data:
            cancel_score = 1.0 - min(pattern_data['cancel_time_ms'] / self.cancel_time_window_ms, 1.0)
            base_score += cancel_score * 0.2
            
        return min(base_score, 1.0)
