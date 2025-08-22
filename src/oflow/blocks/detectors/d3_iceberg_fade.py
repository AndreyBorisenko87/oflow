"""
Детектор D3 — Тающий айсберг
Суть: скрытая крупная заявка, разбитая на маленькие куски.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_detector import BaseDetector
import logging

class D3IcebergFade(BaseDetector):
    """Детектор тающего айсберга"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.same_price_trades = config.get('same_price_trades', 10)
        self.time_window_ms = config.get('time_window_ms', 2000)
        self.min_total_volume = config.get('min_total_volume', 200)
        self.refresh_detect = config.get('refresh_detect', True)
        
    def detect(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Детекция тающего айсберга"""
        self.logger.info("Детекция тающего айсберга...")
        
        if not self.validate_data(data):
            return pd.DataFrame()
            
        book_top = data['book_top']
        quotes = data['quotes']
        tape = data['tape']
        
        events = []
        
        # Анализ по временным окнам
        exchanges = tape['exchange'].unique()
        total_exchanges = len(exchanges)
        self.logger.info(f"Обрабатываем {total_exchanges} бирж...")
        
        for i, exchange in enumerate(exchanges):
            if i % 5 == 0:  # Логируем каждые 5 бирж
                self.logger.info(f"Прогресс: {i+1}/{total_exchanges} бирж")
                
            exchange_tape = tape[tape['exchange'] == exchange].copy()
            exchange_book = book_top[book_top['exchange'] == exchange].copy()
            exchange_quotes = quotes[quotes['exchange'] == exchange].copy()
            
            if exchange_tape.empty:
                continue
                
            # Группировка по временным окнам
            exchange_tape['ts_group'] = exchange_tape['ts_ns'] // (self.time_window_ms * 1_000_000)
            
            time_groups = exchange_tape['ts_group'].unique()
            total_groups = len(time_groups)
            
            for j, ts_group in enumerate(time_groups):
                if j % 100 == 0 and j > 0:  # Логируем каждые 100 временных групп
                    self.logger.debug(f"Биржа {exchange}: {j+1}/{total_groups} временных групп")
                    
                # Анализ в окне
                window_tape = exchange_tape[exchange_tape['ts_group'] == ts_group]
                
                if window_tape.empty:
                    continue
                    
                # Поиск паттернов айсберга
                iceberg_patterns = self._detect_iceberg_patterns(window_tape)
                
                for pattern in iceberg_patterns:
                    # Проверка общего объема
                    if pattern['total_volume'] >= self.min_total_volume:
                        # Проверка обновлений в стакане
                        book_refreshes = 0
                        if self.refresh_detect:
                            book_refreshes = self._count_book_refreshes(
                                exchange_book, 
                                pattern['price'], 
                                pattern['side'],
                                pattern['start_ts'],
                                pattern['end_ts']
                            )
                        
                        event = {
                            'ts_ns': pattern['end_ts'],
                            'exchange': exchange,
                            'pattern_type': 'iceberg_fade',
                            'price': pattern['price'],
                            'side': pattern['side'],
                            'trades_count': pattern['trades_count'],
                            'total_volume': pattern['total_volume'],
                            'book_refreshes': book_refreshes,
                            'confidence': self.get_confidence_score({
                                'trades_count': pattern['trades_count'],
                                'total_volume': pattern['total_volume'],
                                'book_refreshes': book_refreshes
                            }),
                            'metadata': {
                                'start_ts': pattern['start_ts'],
                                'end_ts': pattern['end_ts'],
                                'trades_details': pattern['trades_details']
                            }
                        }
                        events.append(event)
        
        if not events:
            return pd.DataFrame()
            
        events_df = pd.DataFrame(events)
        events_df['detector'] = self.name
        events_df['ts_ns'] = pd.to_datetime(events_df['ts_ns'], unit='ns')
        
        return events_df
    
    def _detect_iceberg_patterns(self, window_tape: pd.DataFrame) -> List[Dict]:
        """Обнаружение паттернов айсберга"""
        if window_tape.empty:
            return []
            
        patterns = []
        
        # Группировка по цене и стороне агрессии
        for (price, side), group in window_tape.groupby(['price', 'aggression_side']):
            if len(group) >= self.same_price_trades:
                # Сортировка по времени
                sorted_group = group.sort_values('ts_ns')
                
                # Анализ временного распределения
                start_ts = sorted_group.iloc[0]['ts_ns']
                end_ts = sorted_group.iloc[-1]['ts_ns']
                
                # Проверка временного окна
                time_diff_ms = (end_ts - start_ts) // 1_000_000
                if time_diff_ms <= self.time_window_ms:
                    total_volume = group['size'].sum()
                    
                    pattern = {
                        'price': price,
                        'side': side,
                        'trades_count': len(group),
                        'total_volume': total_volume,
                        'start_ts': start_ts,
                        'end_ts': end_ts,
                        'trades_details': group[['ts_ns', 'size']].to_dict('records')
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _count_book_refreshes(self, exchange_book: pd.DataFrame, price: float, side: str, start_ts: int, end_ts: int) -> int:
        """Подсчет обновлений в стакане"""
        if exchange_book.empty:
            return 0
            
        # Фильтрация изменений в стакане в заданном временном окне
        book_changes = exchange_book[
            (exchange_book['ts_ns'] >= start_ts) & 
            (exchange_book['ts_ns'] <= end_ts) &
            (exchange_book['price'] == price) &
            (exchange_book['side'] == side)
        ].copy()
        
        if book_changes.empty:
            return 0
            
        # Подсчет уникальных временных меток (каждое обновление)
        unique_timestamps = book_changes['ts_ns'].nunique()
        
        return unique_timestamps
    
    def get_confidence_score(self, pattern_data: Dict) -> float:
        """Скоринг уверенности для тающего айсберга"""
        base_score = 0.3
        
        # Чем больше сделок, тем выше уверенность
        if 'trades_count' in pattern_data:
            trades_score = min(pattern_data['trades_count'] / self.same_price_trades, 1.0)
            base_score += trades_score * 0.3
            
        # Чем больше объем, тем выше уверенность
        if 'total_volume' in pattern_data:
            volume_score = min(pattern_data['total_volume'] / self.min_total_volume, 1.0)
            base_score += volume_score * 0.2
            
        # Чем больше обновлений в стакане, тем выше уверенность
        if 'book_refreshes' in pattern_data and self.refresh_detect:
            refresh_score = min(pattern_data['book_refreshes'] / 5.0, 1.0)
            base_score += refresh_score * 0.2
            
        return min(base_score, 1.0)
