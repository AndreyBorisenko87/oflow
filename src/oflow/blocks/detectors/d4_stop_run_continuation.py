"""
Детектор D4 — Продолжение после выноса стопов
Суть: цена пробивает уровень стопов и продолжает движение.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_detector import BaseDetector
import logging

class D4StopRunContinuation(BaseDetector):
    """Детектор продолжения после выноса стопов"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_price_move_percent = config.get('min_price_move_percent', 0.3)
        self.time_window_ms = config.get('time_window_ms', 2000)
        self.no_pullback_window_ms = config.get('no_pullback_window_ms', 5000)
        self.min_volume_ratio = config.get('min_volume_ratio', 3.0)
        
    def detect(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Детекция продолжения после выноса стопов"""
        self.logger.info("Детекция продолжения после выноса стопов...")
        
        if not self.validate_data(data):
            return pd.DataFrame()
            
        book_top = data['book_top']
        quotes = data['quotes']
        tape = data['tape']
        
        events = []
        
        # Анализ по временным окнам
        exchanges = quotes['exchange'].unique()
        total_exchanges = len(exchanges)
        self.logger.info(f"Обрабатываем {total_exchanges} бирж...")
        
        for i, exchange in enumerate(exchanges):
            if i % 5 == 0:  # Логируем каждые 5 бирж
                self.logger.info(f"Прогресс: {i+1}/{total_exchanges} бирж")
                
            exchange_quotes = quotes[quotes['exchange'] == exchange].copy()
            exchange_tape = tape[tape['exchange'] == exchange].copy()
            
            if exchange_quotes.empty or exchange_tape.empty:
                continue
                
            # Группировка по временным окнам
            exchange_quotes['ts_group'] = exchange_quotes['ts_ns'] // (self.time_window_ms * 1_000_000)
            exchange_tape['ts_group'] = exchange_tape['ts_ns'] // (self.time_window_ms * 1_000_000)
            
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
                    
                # Поиск резких движений цены
                sharp_moves = self._detect_sharp_price_moves(window_quotes)
                
                for move in sharp_moves:
                    # Анализ объема в направлении движения
                    volume_analysis = self._analyze_volume_pattern(window_tape, move)
                    
                    if volume_analysis['volume_ratio'] >= self.min_volume_ratio:
                        # Проверка отсутствия возврата
                        no_pullback = self._check_no_pullback(
                            exchange_quotes, 
                            move['end_ts'], 
                            move['direction'],
                            move['end_price']
                        )
                        
                        if no_pullback:
                            event = {
                                'ts_ns': move['end_ts'],
                                'exchange': exchange,
                                'pattern_type': 'stop_run_continuation',
                                'direction': move['direction'],
                                'price_move_percent': move['price_move_percent'],
                                'price_change': move['price_change'],
                                'volume_ratio': volume_analysis['volume_ratio'],
                                'no_pullback_confirmed': True,
                                'confidence': self.get_confidence_score({
                                    'price_move_percent': move['price_move_percent'],
                                    'volume_ratio': volume_analysis['volume_ratio'],
                                    'no_pullback_confirmed': True
                                }),
                                'metadata': {
                                    'start_price': move['start_price'],
                                    'end_price': move['end_price'],
                                    'start_ts': move['start_ts'],
                                    'end_ts': move['end_ts'],
                                    'volume_details': volume_analysis
                                }
                            }
                            events.append(event)
        
        if not events:
            return pd.DataFrame()
            
        events_df = pd.DataFrame(events)
        events_df['detector'] = self.name
        events_df['ts_ns'] = pd.to_datetime(events_df['ts_ns'], unit='ns')
        
        return events_df
    
    def _detect_sharp_price_moves(self, window_quotes: pd.DataFrame) -> List[Dict]:
        """Обнаружение резких движений цены"""
        if window_quotes.empty:
            return []
            
        moves = []
        
        # Сортировка по времени
        sorted_quotes = window_quotes.sort_values('ts_ns')
        
        if len(sorted_quotes) < 2:
            return moves
            
        # Анализ изменения цены
        start_price = sorted_quotes.iloc[0]['mid_price']
        end_price = sorted_quotes.iloc[-1]['mid_price']
        
        price_change = end_price - start_price
        price_change_abs = abs(price_change)
        
        # Вычисление процентного изменения
        price_move_percent = (price_change_abs / start_price) * 100
        
        if price_move_percent >= self.min_price_move_percent:
            direction = 'up' if price_change > 0 else 'down'
            
            move = {
                'start_price': start_price,
                'end_price': end_price,
                'price_change': price_change,
                'price_move_percent': price_move_percent,
                'direction': direction,
                'start_ts': sorted_quotes.iloc[0]['ts_ns'],
                'end_ts': sorted_quotes.iloc[-1]['ts_ns']
            }
            moves.append(move)
        
        return moves
    
    def _analyze_volume_pattern(self, window_tape: pd.DataFrame, move: Dict) -> Dict:
        """Анализ паттерна объема"""
        if window_tape.empty:
            return {'volume_ratio': 0.0, 'direction_volume': 0.0}
            
        # Фильтрация сделок по направлению движения
        if move['direction'] == 'up':
            relevant_trades = window_tape[window_tape['side'] == 'buy']
        else:
            relevant_trades = window_tape[window_tape['side'] == 'sell']
            
        if relevant_trades.empty:
            return {'volume_ratio': 0.0, 'direction_volume': 0.0}
            
        # Объем в направлении движения
        direction_volume = relevant_trades['size'].sum()
        
        # Общий объем
        total_volume = window_tape['size'].sum()
        
        # Соотношение объема
        volume_ratio = direction_volume / total_volume if total_volume > 0 else 0
        
        return {
            'volume_ratio': volume_ratio,
            'direction_volume': direction_volume,
            'total_volume': total_volume
        }
    
    def _check_no_pullback(self, exchange_quotes: pd.DataFrame, start_ts: int, direction: str, start_price: float) -> bool:
        """Проверка отсутствия возврата"""
        # Фильтрация котировок после движения
        future_quotes = exchange_quotes[exchange_quotes['ts_ns'] > start_ts].copy()
        
        if future_quotes.empty:
            return True
            
        # Ограничение временным окном
        end_ts = start_ts + (self.no_pullback_window_ms * 1_000_000)
        future_quotes = future_quotes[future_quotes['ts_ns'] <= end_ts]
        
        if future_quotes.empty:
            return True
            
        # Проверка на возврат
        if direction == 'up':
            # Для роста проверяем, не упала ли цена ниже начальной
            min_price = future_quotes['mid_price'].min()
            no_pullback = min_price >= start_price
        else:
            # Для падения проверяем, не выросла ли цена выше начальной
            max_price = future_quotes['mid_price'].max()
            no_pullback = max_price <= start_price
        
        return no_pullback
    
    def get_confidence_score(self, pattern_data: Dict) -> float:
        """Скоринг уверенности для продолжения после выноса стопов"""
        base_score = 0.3
        
        # Чем больше движение цены, тем выше уверенность
        if 'price_move_percent' in pattern_data:
            move_score = min(pattern_data['price_move_percent'] / self.min_price_move_percent, 1.0)
            base_score += move_score * 0.3
            
        # Чем больше объем, тем выше уверенность
        if 'volume_ratio' in pattern_data:
            volume_score = min(pattern_data['volume_ratio'] / self.min_volume_ratio, 1.0)
            base_score += volume_score * 0.2
            
        # Подтверждение отсутствия возврата
        if 'no_pullback_confirmed' in pattern_data and pattern_data['no_pullback_confirmed']:
            base_score += 0.2
            
        return min(base_score, 1.0)
