"""
Детектор Scalping Strategy (LVB)
Обнаруживает паттерны вакуума ликвидности и refill
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_detector import BaseDetector
import logging

class LVBDetector(BaseDetector):
    """Детектор Scalping Strategy"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_vacuum_size = config.get('lvb_min_vacuum_size', 1000)  # USDT
        self.vacuum_threshold = config.get('lvb_vacuum_threshold', 0.7)  # 70% исчезновения
        self.refill_threshold = config.get('lvb_refill_threshold', 0.5)  # 50% восстановления
        self.time_window = config.get('lvb_time_window', 1000)  # мс
        
    def detect(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Детекция LVB паттернов"""
        self.logger.info("Детекция LVB паттернов...")
        
        if not self.validate_data(data):
            return pd.DataFrame()
            
        book_top = data['book_top']
        quotes = data['quotes']
        
        events = []
        
        # Анализ по уровням книги
        for level in range(5):  # Топ-5 уровней
            level_data = book_top[book_top['level'] == level].copy()
            
            if level_data.empty:
                continue
                
            # Поиск вакуумов ликвидности
            vacuum_events = self._detect_vacuum(level_data, level)
            events.extend(vacuum_events)
            
            # Поиск refill паттернов
            refill_events = self._detect_refill(level_data, level)
            events.extend(refill_events)
        
        if not events:
            return pd.DataFrame()
            
        events_df = pd.DataFrame(events)
        events_df['detector'] = self.name
        events_df['ts_ns'] = pd.to_datetime(events_df['ts_ns'], unit='ns')
        
        return events_df
    
    def _detect_vacuum(self, level_data: pd.DataFrame, level: int) -> List[Dict]:
        """Обнаружение вакуума ликвидности"""
        events = []
        
        # Группировка по времени
        level_data['ts_group'] = level_data['ts_ns'] // (self.time_window * 1_000_000)
        
        for ts_group, group in level_data.groupby('ts_group'):
            if group.empty:
                continue
                
            # Анализ изменения размера
            size_changes = group['size'].diff().fillna(0)
            total_size = group['size'].sum()
            
            # Вакуум: резкое уменьшение размера
            if size_changes.min() < -self.min_vacuum_size:
                vacuum_ratio = abs(size_changes.min()) / total_size
                
                if vacuum_ratio > self.vacuum_threshold:
                    event = {
                        'ts_ns': group['ts_ns'].iloc[0],
                        'exchange': group['exchange'].iloc[0],
                        'pattern_type': 'vacuum',
                        'level': level,
                        'vacuum_size': abs(size_changes.min()),
                        'vacuum_ratio': vacuum_ratio,
                        'confidence': self.get_confidence_score({
                            'vacuum_ratio': vacuum_ratio,
                            'level': level
                        }),
                        'metadata': {
                            'side': group['side'].iloc[0],
                            'price': group['price'].iloc[0]
                        }
                    }
                    events.append(event)
        
        return events
    
    def _detect_refill(self, level_data: pd.DataFrame, level: int) -> List[Dict]:
        """Обнаружение refill паттернов"""
        events = []
        
        # Группировка по времени
        level_data['ts_group'] = level_data['ts_ns'] // (self.time_window * 1_000_000)
        
        for ts_group, group in level_data.groupby('ts_group'):
            if group.empty:
                continue
                
            # Анализ восстановления ликвидности
            size_changes = group['size'].diff().fillna(0)
            total_size = group['size'].sum()
            
            # Refill: увеличение размера после вакуума
            if size_changes.max() > 0:
                refill_ratio = size_changes.max() / total_size
                
                if refill_ratio > self.refill_threshold:
                    event = {
                        'ts_ns': group['ts_ns'].iloc[0],
                        'exchange': group['exchange'].iloc[0],
                        'pattern_type': 'refill',
                        'level': level,
                        'refill_size': size_changes.max(),
                        'refill_ratio': refill_ratio,
                        'confidence': self.get_confidence_score({
                            'refill_ratio': refill_ratio,
                            'level': level
                        }),
                        'metadata': {
                            'side': group['side'].iloc[0],
                            'price': group['price'].iloc[0]
                        }
                    }
                    events.append(event)
        
        return events
    
    def get_confidence_score(self, pattern_data: Dict) -> float:
        """Скоринг уверенности для LVB паттернов"""
        base_score = 0.5
        
        if 'vacuum_ratio' in pattern_data:
            # Чем больше вакуум, тем выше уверенность
            vacuum_score = min(pattern_data['vacuum_ratio'] / self.vacuum_threshold, 1.0)
            base_score += vacuum_score * 0.3
            
        if 'refill_ratio' in pattern_data:
            # Чем больше refill, тем выше уверенность
            refill_score = min(pattern_data['refill_ratio'] / self.refill_threshold, 1.0)
            base_score += refill_score * 0.2
            
        # Уровень влияет на уверенность (ближе к рынку = выше)
        if 'level' in pattern_data:
            level_score = (5 - pattern_data['level']) / 5.0
            base_score += level_score * 0.2
            
        return min(base_score, 1.0)
