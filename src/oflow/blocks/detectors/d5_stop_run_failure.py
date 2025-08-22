"""
Детектор D5 — Ложный вынос стопов
Суть: цена выбивает стопы, но сразу возвращается.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_detector import BaseDetector
import logging

class D5StopRunFailure(BaseDetector):
    """Детектор ложного выноса стопов"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.break_extreme_confirm_ticks = config.get('break_extreme_confirm_ticks', 2)
        self.reversal_window_ms = config.get('reversal_window_ms', 5000)
        self.reversal_threshold_ratio = config.get('reversal_threshold_ratio', 0.7)
        self.opposite_large_trade = config.get('opposite_large_trade', True)
        
    def detect(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Детекция ложного выноса стопов"""
        self.logger.info("Детекция ложного выноса стопов...")
        
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
                    
                # Поиск пробитий экстремумов
                extreme_breaks = self._detect_extreme_breaks(window_quotes)
                
                for break_event in extreme_breaks:
                    # Проверка возврата цены
                    reversal_analysis = self._check_price_reversal(
                        exchange_quotes, 
                        break_event['break_ts'], 
                        break_event['direction'],
                        break_event['break_price'],
                        break_event['extreme_price']
                    )
                    
                    if reversal_analysis['reversal_detected']:
                        # Поиск крупных сделок в обратную сторону
                        opposite_trades = self._find_opposite_large_trades(
                            exchange_tape, 
                            break_event['break_ts'], 
                            break_event['direction']
                        )
                        
                        if opposite_trades['found'] or not self.opposite_large_trade:
                            event = {
                                'ts_ns': break_event['break_ts'],
                                'exchange': exchange,
                                'pattern_type': 'stop_run_failure',
                                'direction': break_event['direction'],
                                'break_price': break_event['break_price'],
                                'extreme_price': break_event['extreme_price'],
                                'break_ticks': break_event['break_ticks'],
                                'reversal_ratio': reversal_analysis['reversal_ratio'],
                                'opposite_large_trades': opposite_trades['count'],
                                'confidence': self.get_confidence_score({
                                    'break_ticks': break_event['break_ticks'],
                                    'reversal_ratio': reversal_analysis['reversal_ratio'],
                                    'opposite_large_trades': opposite_trades['count']
                                }),
                                'metadata': {
                                    'break_ts': break_event['break_ts'],
                                    'reversal_ts': reversal_analysis['reversal_ts'],
                                    'opposite_trades': opposite_trades['details']
                                }
                            }
                            events.append(event)
        
        if not events:
            return pd.DataFrame()
            
        events_df = pd.DataFrame(events)
        events_df['detector'] = self.name
        events_df['ts_ns'] = pd.to_datetime(events_df['ts_ns'], unit='ns')
        
        return events_df
    
    def _detect_extreme_breaks(self, window_quotes: pd.DataFrame) -> List[Dict]:
        """Обнаружение пробитий экстремумов"""
        if window_quotes.empty:
            return []
            
        breaks = []
        
        # Сортировка по времени
        sorted_quotes = window_quotes.sort_values('ts_ns')
        
        if len(sorted_quotes) < 3:
            return breaks
            
        # Определение локального экстремума
        for i in range(1, len(sorted_quotes) - 1):
            prev_price = sorted_quotes.iloc[i-1]['mid_price']
            curr_price = sorted_quotes.iloc[i]['mid_price']
            next_price = sorted_quotes.iloc[i+1]['mid_price']
            
            # Проверка на максимум
            if curr_price > prev_price and curr_price > next_price:
                extreme_price = curr_price
                direction = 'up'
            # Проверка на минимум
            elif curr_price < prev_price and curr_price < next_price:
                extreme_price = curr_price
                direction = 'down'
            else:
                continue
            
            # Проверка пробития экстремума
            if direction == 'up':
                # При росте проверяем, пробил ли следующий тик максимум
                if next_price > extreme_price:
                    break_price = next_price
                    break_ts = sorted_quotes.iloc[i+1]['ts_ns']
                    
                    # Подтверждение пробития
                    if i + 2 < len(sorted_quotes):
                        confirm_price = sorted_quotes.iloc[i+2]['mid_price']
                        if confirm_price > extreme_price:
                            break_ticks = int((break_price - extreme_price) / 0.01)  # Предполагаем тик = 0.01
                            
                            if break_ticks >= self.break_extreme_confirm_ticks:
                                break_event = {
                                    'direction': direction,
                                    'extreme_price': extreme_price,
                                    'break_price': break_price,
                                    'break_ts': break_ts,
                                    'break_ticks': break_ticks
                                }
                                breaks.append(break_event)
            
            elif direction == 'down':
                # При падении проверяем, пробил ли следующий тик минимум
                if next_price < extreme_price:
                    break_price = next_price
                    break_ts = sorted_quotes.iloc[i+1]['ts_ns']
                    
                    # Подтверждение пробития
                    if i + 2 < len(sorted_quotes):
                        confirm_price = sorted_quotes.iloc[i+2]['mid_price']
                        if confirm_price < extreme_price:
                            break_ticks = int((extreme_price - break_price) / 0.01)
                            
                            if break_ticks >= self.break_extreme_confirm_ticks:
                                break_event = {
                                    'direction': direction,
                                    'extreme_price': extreme_price,
                                    'break_price': break_price,
                                    'break_ts': break_ts,
                                    'break_ticks': break_ticks
                                }
                                breaks.append(break_event)
        
        return breaks
    
    def _check_price_reversal(self, exchange_quotes: pd.DataFrame, break_ts: int, direction: str, break_price: float, extreme_price: float) -> Dict:
        """Проверка возврата цены"""
        # Фильтрация котировок после пробития
        future_quotes = exchange_quotes[exchange_quotes['ts_ns'] > break_ts].copy()
        
        if future_quotes.empty:
            return {'reversal_detected': False, 'reversal_ratio': 0.0}
            
        # Ограничение временным окном
        end_ts = break_ts + (self.reversal_window_ms * 1_000_000)
        future_quotes = future_quotes[future_quotes['ts_ns'] <= end_ts]
        
        if future_quotes.empty:
            return {'reversal_detected': False, 'reversal_ratio': 0.0}
            
        # Вычисление возврата
        if direction == 'up':
            # При пробитии максимума ищем возврат к уровню экстремума
            min_price = future_quotes['mid_price'].min()
            price_range = break_price - extreme_price
            reversal_range = break_price - min_price
            
            if price_range > 0:
                reversal_ratio = reversal_range / price_range
            else:
                reversal_ratio = 0.0
                
        else:  # direction == 'down'
            # При пробитии минимума ищем возврат к уровню экстремума
            max_price = future_quotes['mid_price'].max()
            price_range = extreme_price - break_price
            reversal_range = max_price - break_price
            
            if price_range > 0:
                reversal_ratio = reversal_range / price_range
            else:
                reversal_ratio = 0.0
        
        reversal_detected = reversal_ratio >= self.reversal_threshold_ratio
        
        return {
            'reversal_detected': reversal_detected,
            'reversal_ratio': reversal_ratio,
            'reversal_ts': future_quotes['ts_ns'].iloc[-1]
        }
    
    def _find_opposite_large_trades(self, exchange_tape: pd.DataFrame, break_ts: int, direction: str) -> Dict:
        """Поиск крупных сделок в обратную сторону"""
        # Фильтрация сделок после пробития
        future_trades = exchange_tape[exchange_tape['ts_ns'] > break_ts].copy()
        
        if future_trades.empty:
            return {'found': False, 'count': 0, 'details': []}
            
        # Определение противоположной стороны
        if direction == 'up':
            opposite_side = 'sell'
        else:
            opposite_side = 'buy'
            
        # Фильтрация по стороне
        opposite_trades = future_trades[future_trades['side'] == opposite_side].copy()
        
        if opposite_trades.empty:
            return {'found': False, 'count': 0, 'details': []}
            
        # Поиск крупных сделок (больше среднего)
        avg_size = opposite_trades['size'].mean()
        large_trades = opposite_trades[opposite_trades['size'] > avg_size * 2].copy()
        
        return {
            'found': len(large_trades) > 0,
            'count': len(large_trades),
            'details': large_trades[['ts_ns', 'price', 'size']].to_dict('records')
        }
    
    def get_confidence_score(self, pattern_data: Dict) -> float:
        """Скоринг уверенности для ложного выноса стопов"""
        base_score = 0.3
        
        # Чем больше пробитие, тем выше уверенность
        if 'break_ticks' in pattern_data:
            break_score = min(pattern_data['break_ticks'] / self.break_extreme_confirm_ticks, 1.0)
            base_score += break_score * 0.3
            
        # Чем больше возврат, тем выше уверенность
        if 'reversal_ratio' in pattern_data:
            reversal_score = min(pattern_data['reversal_ratio'] / self.reversal_threshold_ratio, 1.0)
            base_score += reversal_score * 0.3
            
        # Чем больше крупных сделок в обратную сторону, тем выше уверенность
        if 'opposite_large_trades' in pattern_data:
            trades_score = min(pattern_data['opposite_large_trades'] / 3.0, 1.0)
            base_score += trades_score * 0.1
            
        return min(base_score, 1.0)
