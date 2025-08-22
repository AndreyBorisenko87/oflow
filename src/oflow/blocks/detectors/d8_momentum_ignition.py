"""
Детектор D8 — Разгон на импульсе
Суть: движение ускоряется за счёт толпы.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_detector import BaseDetector
import logging

class D8MomentumIgnition(BaseDetector):
    """Детектор разгона на импульсе"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.acceleration_window_ms = config.get('acceleration_window_ms', 3000)
        self.min_volume_ratio = config.get('min_volume_ratio', 3.0)
        self.one_side_ratio = config.get('one_side_ratio', 0.8)
        self.candle_body_ratio = config.get('candle_body_ratio', 0.7)
        
    def detect(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Детекция разгона на импульсе"""
        self.logger.info("Детекция разгона на импульсе...")
        
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
                
            # Группировка по временным окнам
            exchange_quotes['ts_group'] = exchange_quotes['ts_ns'] // (self.acceleration_window_ms * 1_000_000)
            exchange_tape['ts_group'] = exchange_tape['ts_ns'] // (self.acceleration_window_ms * 1_000_000)
            
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
                    
                # Поиск ускорения движения
                acceleration_events = self._detect_price_acceleration(window_quotes)
                
                for accel_event in acceleration_events:
                    # Проверка объема сделок
                    volume_analysis = self._analyze_volume_surge(window_tape, accel_event)
                    
                    if volume_analysis['volume_ratio'] >= self.min_volume_ratio:
                        # Проверка доминирования одной стороны
                        side_dominance = self._check_side_dominance(window_tape, accel_event['direction'])
                        
                        if side_dominance >= self.one_side_ratio:
                            # Проверка соотношения тела свечи
                            candle_analysis = self._analyze_candle_structure(window_quotes, accel_event)
                            
                            if candle_analysis['body_ratio'] >= self.candle_body_ratio:
                                event = {
                                    'ts_ns': accel_event['ts_ns'],
                                    'exchange': exchange,
                                    'pattern_type': 'momentum_ignition',
                                    'direction': accel_event['direction'],
                                    'acceleration_rate': accel_event['acceleration_rate'],
                                    'price_change_points': accel_event['price_change_points'],
                                    'volume_ratio': volume_analysis['volume_ratio'],
                                    'side_dominance': side_dominance,
                                    'candle_body_ratio': candle_analysis['body_ratio'],
                                    'confidence': self.get_confidence_score({
                                        'acceleration_rate': accel_event['acceleration_rate'],
                                        'volume_ratio': volume_analysis['volume_ratio'],
                                        'side_dominance': side_dominance,
                                        'candle_body_ratio': candle_analysis['body_ratio']
                                    }),
                                    'metadata': {
                                        'start_price': accel_event['start_price'],
                                        'end_price': accel_event['end_price'],
                                        'start_ts': accel_event['start_ts'],
                                        'end_ts': accel_event['end_ts'],
                                        'velocity_changes': accel_event['velocity_changes']
                                    }
                                }
                                events.append(event)
        
        if not events:
            return pd.DataFrame()
            
        events_df = pd.DataFrame(events)
        events_df['detector'] = self.name
        events_df['ts_ns'] = pd.to_datetime(events_df['ts_ns'], unit='ns')
        
        return events_df
    
    def _detect_price_acceleration(self, window_quotes: pd.DataFrame) -> List[Dict]:
        """Обнаружение ускорения движения цены"""
        acceleration_events = []
        
        if window_quotes.empty:
            return acceleration_events
            
        # Сортировка по времени
        sorted_quotes = window_quotes.sort_values('ts_ns')
        
        if len(sorted_quotes) < 3:
            return acceleration_events
            
        # Анализ изменения скорости движения
        velocity_changes = []
        
        for i in range(1, len(sorted_quotes)):
            prev_price = sorted_quotes.iloc[i-1]['mid_price']
            curr_price = sorted_quotes.iloc[i]['mid_price']
            prev_ts = sorted_quotes.iloc[i-1]['ts_ns']
            curr_ts = sorted_quotes.iloc[i]['ts_ns']
            
            # Вычисление скорости (изменение цены за единицу времени)
            time_diff = (curr_ts - prev_ts) / 1_000_000_000  # в секундах
            if time_diff > 0:
                velocity = (curr_price - prev_price) / time_diff
                velocity_changes.append({
                    'ts_ns': curr_ts,
                    'velocity': velocity,
                    'price': curr_price
                })
        
        if len(velocity_changes) < 2:
            return acceleration_events
            
        # Анализ ускорения
        for i in range(1, len(velocity_changes)):
            prev_velocity = velocity_changes[i-1]['velocity']
            curr_velocity = velocity_changes[i]['velocity']
            
            # Проверка на ускорение (увеличение скорости)
            if abs(curr_velocity) > abs(prev_velocity) and curr_velocity * prev_velocity > 0:
                # Вычисление ускорения
                time_diff = (velocity_changes[i]['ts_ns'] - velocity_changes[i-1]['ts_ns']) / 1_000_000_000
                if time_diff > 0:
                    acceleration_rate = (curr_velocity - prev_velocity) / time_diff
                    
                    # Проверка значимости ускорения
                    if abs(acceleration_rate) > 0.1:  # Минимальный порог ускорения
                        direction = 'up' if curr_velocity > 0 else 'down'
                        
                        accel_event = {
                            'ts_ns': velocity_changes[i]['ts_ns'],
                            'direction': direction,
                            'acceleration_rate': acceleration_rate,
                            'start_price': sorted_quotes.iloc[0]['mid_price'],
                            'end_price': velocity_changes[i]['price'],
                            'start_ts': sorted_quotes.iloc[0]['ts_ns'],
                            'end_ts': velocity_changes[i]['ts_ns'],
                            'price_change_points': abs(velocity_changes[i]['price'] - sorted_quotes.iloc[0]['mid_price']),
                            'velocity_changes': velocity_changes
                        }
                        acceleration_events.append(accel_event)
        
        return acceleration_events
    
    def _analyze_volume_surge(self, window_tape: pd.DataFrame, accel_event: Dict) -> Dict:
        """Анализ всплеска объема"""
        if window_tape.empty:
            return {'volume_ratio': 0.0, 'total_volume': 0.0}
            
        # Объем в текущем окне
        current_volume = window_tape['size'].sum()
        
        # Фильтрация сделок по направлению движения
        if accel_event['direction'] == 'up':
            relevant_trades = window_tape[window_tape['side'] == 'buy']
        else:
            relevant_trades = window_tape[window_tape['side'] == 'sell']
            
        if relevant_trades.empty:
            return {'volume_ratio': 0.0, 'total_volume': current_volume}
            
        # Объем в направлении движения
        direction_volume = relevant_trades['size'].sum()
        
        # Средний объем (упрощенная оценка)
        avg_volume = current_volume / len(window_tape) if len(window_tape) > 0 else 0
        
        if avg_volume == 0:
            return {'volume_ratio': 0.0, 'total_volume': current_volume}
            
        volume_ratio = direction_volume / avg_volume
        
        return {
            'volume_ratio': volume_ratio,
            'total_volume': current_volume,
            'direction_volume': direction_volume
        }
    
    def _check_side_dominance(self, window_tape: pd.DataFrame, direction: str) -> float:
        """Проверка доминирования одной стороны"""
        if window_tape.empty:
            return 0.0
            
        # Определение стороны для анализа
        if direction == 'up':
            dominant_side = 'buy'
        else:
            dominant_side = 'sell'
            
        # Подсчет объема по сторонам
        dominant_volume = window_tape[window_tape['side'] == dominant_side]['size'].sum()
        total_volume = window_tape['size'].sum()
        
        if total_volume == 0:
            return 0.0
            
        return dominant_volume / total_volume
    
    def _analyze_candle_structure(self, window_quotes: pd.DataFrame, accel_event: Dict) -> Dict:
        """Анализ структуры свечи"""
        if window_quotes.empty:
            return {'body_ratio': 0.0, 'high': 0.0, 'low': 0.0, 'open': 0.0, 'close': 0.0}
            
        # Определение OHLC
        open_price = window_quotes.iloc[0]['mid_price']
        close_price = window_quotes.iloc[-1]['mid_price']
        high_price = window_quotes['mid_price'].max()
        low_price = window_quotes['mid_price'].min()
        
        # Вычисление соотношения тела свечи
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        
        if total_range == 0:
            body_ratio = 0.0
        else:
            body_ratio = body_size / total_range
        
        return {
            'body_ratio': body_ratio,
            'high': high_price,
            'low': low_price,
            'open': open_price,
            'close': close_price
        }
    
    def get_confidence_score(self, pattern_data: Dict) -> float:
        """Скоринг уверенности для разгона на импульсе"""
        base_score = 0.3
        
        # Чем больше ускорение, тем выше уверенность
        if 'acceleration_rate' in pattern_data:
            accel_score = min(abs(pattern_data['acceleration_rate']) / 0.5, 1.0)
            base_score += accel_score * 0.2
            
        # Чем больше объем, тем выше уверенность
        if 'volume_ratio' in pattern_data:
            volume_score = min(pattern_data['volume_ratio'] / self.min_volume_ratio, 1.0)
            base_score += volume_score * 0.2
            
        # Чем больше доминирование стороны, тем выше уверенность
        if 'side_dominance' in pattern_data:
            side_score = min(pattern_data['side_dominance'] / self.one_side_ratio, 1.0)
            base_score += side_score * 0.2
            
        # Чем больше тело свечи, тем выше уверенность
        if 'candle_body_ratio' in pattern_data:
            candle_score = min(pattern_data['candle_body_ratio'] / self.candle_body_ratio, 1.0)
            base_score += candle_score * 0.1
            
        return min(base_score, 1.0)
