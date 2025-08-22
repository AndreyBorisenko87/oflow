"""
Детектор D2 — Перелом на поглощении
Суть: одна сторона активно давит рынок, но цена не движется.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_detector import BaseDetector
import logging

class D2AbsorptionFlip(BaseDetector):
    """Детектор перелома на поглощении"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_side_ratio = config.get('min_side_ratio', 0.7)
        self.max_price_move_ticks = config.get('max_price_move_ticks', 2)
        self.min_total_volume = config.get('min_total_volume', 100)
        self.persistent_order_threshold = config.get('persistent_order_threshold', 50)
        
    def detect(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Детекция перелома на поглощении"""
        self.logger.info("Детекция перелома на поглощении...")
        
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
            
            if exchange_book.empty or exchange_quotes.empty or exchange_tape.empty:
                continue
                
            # Группировка по временным окнам (1 секунда)
            time_window_ms = 1000
            exchange_quotes['ts_group'] = exchange_quotes['ts_ns'] // (time_window_ms * 1_000_000)
            exchange_tape['ts_group'] = exchange_tape['ts_ns'] // (time_window_ms * 1_000_000)
            
            time_groups = exchange_quotes['ts_group'].unique()
            total_groups = len(time_groups)
            
            for j, ts_group in enumerate(time_groups):
                if j % 100 == 0 and j > 0:  # Логируем каждые 100 временных групп
                    self.logger.debug(f"Биржа {exchange}: {j+1}/{total_groups} временных групп")
                    
                # Анализ в окне
                window_quotes = exchange_quotes[exchange_quotes['ts_group'] == ts_group]
                window_tape = exchange_tape[exchange_tape['ts_group'] == ts_group]
                
                if window_quotes.empty or window_tape.empty:
                    continue
                    
                # Проверка общего объема
                total_volume = window_tape['size'].sum()
                if total_volume < self.min_total_volume:
                    continue
                    
                # Анализ соотношения сторон
                side_ratio = self._calculate_side_ratio(window_tape)
                
                if side_ratio >= self.min_side_ratio:
                    # Проверка движения цены
                    price_movement = self._check_price_movement(window_quotes)
                    
                    if price_movement['within_limit']:
                        # Поиск стойких заявок
                        persistent_orders = self._find_persistent_orders(window_book)
                        
                        if persistent_orders['found']:
                            event = {
                                'ts_ns': window_quotes['ts_ns'].iloc[0],
                                'exchange': exchange,
                                'pattern_type': 'absorption_flip',
                                'side_ratio': side_ratio,
                                'dominant_side': 'sell' if side_ratio >= 0.7 else 'buy',
                                'price_move_ticks': price_movement['move_ticks'],
                                'total_volume': total_volume,
                                'persistent_orders_count': persistent_orders['count'],
                                'confidence': self.get_confidence_score({
                                    'side_ratio': side_ratio,
                                    'price_move_ticks': price_movement['move_ticks'],
                                    'total_volume': total_volume,
                                    'persistent_orders_count': persistent_orders['count']
                                }),
                                'metadata': {
                                    'start_price': price_movement['start_price'],
                                    'end_price': price_movement['end_price'],
                                    'persistent_orders': persistent_orders['details']
                                }
                            }
                            events.append(event)
        
        if not events:
            return pd.DataFrame()
            
        events_df = pd.DataFrame(events)
        events_df['detector'] = self.name
        events_df['ts_ns'] = pd.to_datetime(events_df['ts_ns'], unit='ns')
        
        return events_df
    
    def _calculate_side_ratio(self, window_tape: pd.DataFrame) -> float:
        """Вычисление соотношения сторон в ленте"""
        if window_tape.empty:
            return 0.0
            
        # Подсчет объема по сторонам
        sell_volume = window_tape[window_tape['side'] == 'sell']['size'].sum()
        buy_volume = window_tape[window_tape['side'] == 'buy']['size'].sum()
        total_volume = sell_volume + buy_volume
        
        if total_volume == 0:
            return 0.0
            
        # Возвращаем долю большей стороны
        if sell_volume > buy_volume:
            return sell_volume / total_volume
        else:
            return buy_volume / total_volume
    
    def _check_price_movement(self, window_quotes: pd.DataFrame) -> Dict:
        """Проверка движения цены"""
        if window_quotes.empty:
            return {'within_limit': False, 'move_ticks': 0}
            
        # Сортировка по времени
        sorted_quotes = window_quotes.sort_values('ts_ns')
        
        if len(sorted_quotes) < 2:
            return {'within_limit': False, 'move_ticks': 0}
            
        # Анализ изменения цены
        start_price = sorted_quotes.iloc[0]['mid_price']
        end_price = sorted_quotes.iloc[-1]['mid_price']
        
        price_change = abs(end_price - start_price)
        
        # Вычисление тиков
        tick_size = 0.01  # Предполагаем тик = 0.01 для ETHUSDT
        move_ticks = int(price_change / tick_size)
        
        within_limit = move_ticks <= self.max_price_move_ticks
        
        return {
            'within_limit': within_limit,
            'move_ticks': move_ticks,
            'start_price': start_price,
            'end_price': end_price
        }
    
    def _find_persistent_orders(self, window_book: pd.DataFrame) -> Dict:
        """Поиск стойких заявок в стакане"""
        if window_book.empty:
            return {'found': False, 'count': 0, 'details': []}
            
        persistent_orders = []
        
        # Анализ заявок по уровням
        for level in range(5):  # Топ-5 уровней
            level_data = window_book[window_book['level'] == level]
            
            if level_data.empty:
                continue
                
            for _, order in level_data.iterrows():
                # Проверка на стойкость (упрощенная логика)
                if order['size'] >= self.persistent_order_threshold:
                    persistent_order = {
                        'level': level,
                        'side': order['side'],
                        'price': order['price'],
                        'size': order['size']
                    }
                    persistent_orders.append(persistent_order)
        
        return {
            'found': len(persistent_orders) > 0,
            'count': len(persistent_orders),
            'details': persistent_orders
        }
    
    def get_confidence_score(self, pattern_data: Dict) -> float:
        """Скоринг уверенности для перелома на поглощении"""
        base_score = 0.3
        
        # Чем больше соотношение сторон, тем выше уверенность
        if 'side_ratio' in pattern_data:
            side_score = min(pattern_data['side_ratio'] / self.min_side_ratio, 1.0)
            base_score += side_score * 0.3
            
        # Чем меньше движение цены, тем выше уверенность
        if 'price_move_ticks' in pattern_data:
            move_score = 1.0 - min(pattern_data['price_move_ticks'] / self.max_price_move_ticks, 1.0)
            base_score += move_score * 0.2
            
        # Чем больше объем, тем выше уверенность
        if 'total_volume' in pattern_data:
            volume_score = min(pattern_data['total_volume'] / self.min_total_volume, 1.0)
            base_score += volume_score * 0.1
            
        # Чем больше стойких заявок, тем выше уверенность
        if 'persistent_orders_count' in pattern_data:
            orders_score = min(pattern_data['persistent_orders_count'] / 3.0, 1.0)
            base_score += orders_score * 0.1
            
        return min(base_score, 1.0)
