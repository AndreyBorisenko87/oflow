"""
Детектор D1 — Вакуум ликвидности
Суть: исчезновение заявок в стакане, цена прыгает через пустое место.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_detector import BaseDetector
import logging

class D1LiquidityVacuumBreak(BaseDetector):
    """Детектор вакуума ликвидности"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.empty_levels = config.get('empty_levels', 3)
        self.min_jump_ticks = config.get('min_jump_ticks', 5)
        self.time_window_ms = config.get('time_window_ms', 1000)
        self.min_volume_ratio = config.get('min_volume_ratio', 2.0)
        
    def detect(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Детекция вакуума ликвидности"""
        self.logger.info("Детекция вакуума ликвидности...")
        
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
            exchange_quotes['ts_group'] = exchange_quotes['ts_ns'] // (self.time_window_ms * 1_000_000)
            exchange_tape['ts_group'] = exchange_tape['ts_ns'] // (self.time_window_ms * 1_000_000)
            exchange_book['ts_group'] = exchange_book['ts_ns'] // (self.time_window_ms * 1_000_000)
            
            time_groups = exchange_quotes['ts_group'].unique()
            total_groups = len(time_groups)
            
            for j, ts_group in enumerate(time_groups):
                if j % 100 == 0 and j > 0:  # Логируем каждые 100 временных групп
                    self.logger.debug(f"Биржа {exchange}: {j+1}/{total_groups} временных групп")
                    
                # Анализ в окне
                window_quotes = exchange_quotes[exchange_quotes['ts_group'] == ts_group]
                window_tape = exchange_tape[exchange_tape['ts_group'] == ts_group]
                window_book = exchange_book[exchange_book['ts_group'] == ts_group]
                
                if window_quotes.empty or window_tape.empty or window_book.empty:
                    continue
                    
                # Поиск пустых уровней
                empty_levels_count = self._count_empty_levels(window_book)
                
                if empty_levels_count >= self.empty_levels:
                    # Детекция скачка цены
                    price_jump = self._detect_price_jump(window_quotes)
                    
                    if price_jump['jump_detected']:
                        # Проверка объема сделок
                        volume_ratio = self._check_volume_ratio(window_tape, price_jump)
                        
                        if volume_ratio >= self.min_volume_ratio:
                            event = {
                                'ts_ns': price_jump['ts_ns'],
                                'exchange': exchange,
                                'pattern_type': 'liquidity_vacuum_break',
                                'empty_levels': empty_levels_count,
                                'jump_ticks': price_jump['jump_ticks'],
                                'price_change': price_jump['price_change'],
                                'volume_ratio': volume_ratio,
                                'confidence': self.get_confidence_score({
                                    'empty_levels': empty_levels_count,
                                    'jump_ticks': price_jump['jump_ticks'],
                                    'volume_ratio': volume_ratio
                                }),
                                'metadata': {
                                    'start_price': price_jump['start_price'],
                                    'end_price': price_jump['end_price'],
                                    'start_ts': price_jump['start_ts'],
                                    'end_ts': price_jump['end_ts']
                                }
                            }
                            events.append(event)
        
        if not events:
            return pd.DataFrame()
            
        events_df = pd.DataFrame(events)
        events_df['detector'] = self.name
        events_df['ts_ns'] = pd.to_datetime(events_df['ts_ns'], unit='ns')
        
        return events_df
    
    def _count_empty_levels(self, window_book: pd.DataFrame) -> int:
        """Подсчет пустых уровней в стакане"""
        if window_book.empty:
            return 0
            
        # Анализ по уровням
        empty_count = 0
        for level in range(self.empty_levels):
            level_data = window_book[window_book['level'] == level]
            if level_data.empty or level_data['size'].sum() == 0:
                empty_count += 1
            else:
                break  # Прерываем при первом непустом уровне
                
        return empty_count
    
    def _detect_price_jump(self, window_quotes: pd.DataFrame) -> Dict:
        """Детекция скачка цены"""
        if window_quotes.empty:
            return {'jump_detected': False}
            
        # Сортировка по времени
        sorted_quotes = window_quotes.sort_values('ts_ns')
        
        if len(sorted_quotes) < 2:
            return {'jump_detected': False}
            
        # Анализ изменения цены
        start_price = sorted_quotes.iloc[0]['mid_price']
        end_price = sorted_quotes.iloc[-1]['mid_price']
        start_ts = sorted_quotes.iloc[0]['ts_ns']
        end_ts = sorted_quotes.iloc[-1]['ts_ns']
        
        price_change = end_price - start_price
        price_change_abs = abs(price_change)
        
        # Проверка на скачок в тиках
        tick_size = 0.01  # Предполагаем тик = 0.01 для ETHUSDT
        jump_ticks = int(price_change_abs / tick_size)
        
        jump_detected = jump_ticks >= self.min_jump_ticks
        
        return {
            'jump_detected': jump_detected,
            'jump_ticks': jump_ticks,
            'price_change': price_change,
            'start_price': start_price,
            'end_price': end_price,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'ts_ns': end_ts
        }
    
    def _check_volume_ratio(self, window_tape: pd.DataFrame, price_jump: Dict) -> float:
        """Проверка соотношения объема"""
        if window_tape.empty:
            return 0.0
            
        # Объем в текущем окне
        current_volume = window_tape['size'].sum()
        
        # Средний объем (упрощенная оценка)
        avg_volume = current_volume / len(window_tape) if len(window_tape) > 0 else 0
        
        if avg_volume == 0:
            return 0.0
            
        volume_ratio = current_volume / avg_volume
        
        return volume_ratio
    
    def get_confidence_score(self, pattern_data: Dict) -> float:
        """Скоринг уверенности для вакуума ликвидности"""
        base_score = 0.3
        
        # Чем больше пустых уровней, тем выше уверенность
        if 'empty_levels' in pattern_data:
            levels_score = min(pattern_data['empty_levels'] / self.empty_levels, 1.0)
            base_score += levels_score * 0.3
            
        # Чем больше скачок, тем выше уверенность
        if 'jump_ticks' in pattern_data:
            jump_score = min(pattern_data['jump_ticks'] / self.min_jump_ticks, 1.0)
            base_score += jump_score * 0.2
            
        # Чем больше объем, тем выше уверенность
        if 'volume_ratio' in pattern_data:
            volume_score = min(pattern_data['volume_ratio'] / self.min_volume_ratio, 1.0)
            base_score += volume_score * 0.2
            
        return min(base_score, 1.0)
