"""
Детектор D6 — Сильный перекос стакана
Суть: одна сторона стакана многократно перевешивает другую.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_detector import BaseDetector
import logging

class D6BookImbalance(BaseDetector):
    """Детектор сильного перекоса стакана"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.bid_ask_ratio = config.get('bid_ask_ratio', 3.0)  # Во сколько раз bid > ask
        self.level_wall_multiplier = config.get('level_wall_multiplier', 5.0)  # Во сколько раз уровень > средней стены
        self.levels_depth = config.get('levels_depth', 5)  # Количество уровней для анализа
        
    def detect(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Детекция сильного перекоса стакана"""
        self.logger.info("Детекция сильного перекоса стакана...")
        
        if not self.validate_data(data):
            return pd.DataFrame()
            
        book_top = data['book_top']
        
        events = []
        
        # Анализ по временным окнам
        exchanges = book_top['exchange'].unique()
        total_exchanges = len(exchanges)
        self.logger.info(f"Обрабатываем {total_exchanges} бирж...")
        
        for i, exchange in enumerate(exchanges):
            if i % 5 == 0:  # Логируем каждые 5 бирж
                self.logger.info(f"Прогресс: {i+1}/{total_exchanges} бирж")
                
            exchange_book = book_top[book_top['exchange'] == exchange].copy()
            
            if exchange_book.empty:
                continue
                
            # Группировка по временным окнам (1 секунда)
            time_window_ms = 1000
            exchange_book['ts_group'] = exchange_book['ts_ns'] // (time_window_ms * 1_000_000)
            
            time_groups = exchange_book['ts_group'].unique()
            total_groups = len(time_groups)
            
            for j, ts_group in enumerate(time_groups):
                if j % 100 == 0 and j > 0:  # Логируем каждые 100 временных групп
                    self.logger.debug(f"Биржа {exchange}: {j+1}/{total_groups} временных групп")
                    
                # Анализ в окне
                window_book = exchange_book[exchange_book['ts_group'] == ts_group]
                
                if window_book.empty:
                    continue
                    
                # Анализ перекоса bid/ask
                imbalance_analysis = self._analyze_bid_ask_imbalance(window_book)
                
                if imbalance_analysis['imbalance_ratio'] >= self.bid_ask_ratio:
                    # Поиск стен в стакане
                    wall_analysis = self._find_level_walls(window_book)
                    
                    if wall_analysis['walls_found']:
                        event = {
                            'ts_ns': window_book['ts_ns'].iloc[0],
                            'exchange': exchange,
                            'pattern_type': 'book_imbalance',
                            'imbalance_ratio': imbalance_analysis['imbalance_ratio'],
                            'bid_volume': imbalance_analysis['bid_volume'],
                            'ask_volume': imbalance_analysis['ask_volume'],
                            'dominant_side': imbalance_analysis['dominant_side'],
                            'walls_count': wall_analysis['walls_count'],
                            'max_wall_multiplier': wall_analysis['max_wall_multiplier'],
                            'confidence': self.get_confidence_score({
                                'imbalance_ratio': imbalance_analysis['imbalance_ratio'],
                                'walls_count': wall_analysis['walls_count'],
                                'max_wall_multiplier': wall_analysis['max_wall_multiplier']
                            }),
                            'metadata': {
                                'walls': wall_analysis['walls_details'],
                                'levels_analysis': imbalance_analysis['levels_breakdown']
                            }
                        }
                        events.append(event)
        
        if not events:
            return pd.DataFrame()
            
        events_df = pd.DataFrame(events)
        events_df['detector'] = self.name
        events_df['ts_ns'] = pd.to_datetime(events_df['ts_ns'], unit='ns')
        
        return events_df
    
    def _analyze_bid_ask_imbalance(self, window_book: pd.DataFrame) -> Dict:
        """Анализ перекоса bid/ask"""
        if window_book.empty:
            return {
                'imbalance_ratio': 0.0,
                'bid_volume': 0.0,
                'ask_volume': 0.0,
                'dominant_side': 'none',
                'levels_breakdown': []
            }
        
        # Анализ по уровням
        levels_breakdown = []
        total_bid_volume = 0
        total_ask_volume = 0
        
        for level in range(self.levels_depth):
            level_data = window_book[window_book['level'] == level]
            
            if level_data.empty:
                levels_breakdown.append({
                    'level': level,
                    'bid_volume': 0,
                    'ask_volume': 0,
                    'ratio': 0.0
                })
                continue
            
            # Подсчет объема по сторонам
            bid_data = level_data[level_data['side'] == 'bid']
            ask_data = level_data[level_data['side'] == 'ask']
            
            bid_volume = bid_data['size'].sum() if not bid_data.empty else 0
            ask_volume = ask_data['size'].sum() if not ask_data.empty else 0
            
            total_bid_volume += bid_volume
            total_ask_volume += ask_volume
            
            # Соотношение для уровня
            level_ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
            
            levels_breakdown.append({
                'level': level,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'ratio': level_ratio
            })
        
        # Общий перекос
        if total_ask_volume > 0:
            imbalance_ratio = total_bid_volume / total_ask_volume
        else:
            imbalance_ratio = float('inf')
        
        # Определение доминирующей стороны
        if total_bid_volume > total_ask_volume * self.bid_ask_ratio:
            dominant_side = 'bid'
        elif total_ask_volume > total_bid_volume * self.bid_ask_ratio:
            dominant_side = 'ask'
        else:
            dominant_side = 'balanced'
        
        return {
            'imbalance_ratio': imbalance_ratio,
            'bid_volume': total_bid_volume,
            'ask_volume': total_ask_volume,
            'dominant_side': dominant_side,
            'levels_breakdown': levels_breakdown
        }
    
    def _find_level_walls(self, window_book: pd.DataFrame) -> Dict:
        """Поиск стен в стакане"""
        if window_book.empty:
            return {
                'walls_found': False,
                'walls_count': 0,
                'max_wall_multiplier': 0.0,
                'walls_details': []
            }
        
        walls = []
        max_multiplier = 0.0
        
        # Вычисление средней заявки
        all_sizes = window_book['size'].values
        if len(all_sizes) == 0:
            return {
                'walls_found': False,
                'walls_count': 0,
                'max_wall_multiplier': 0.0,
                'walls_details': []
            }
        
        avg_size = np.mean(all_sizes)
        
        # Поиск стен (заявок значительно больше среднего)
        for level in range(self.levels_depth):
            level_data = window_book[window_book['level'] == level]
            
            if level_data.empty:
                continue
            
            for _, order in level_data.iterrows():
                size_multiplier = order['size'] / avg_size if avg_size > 0 else 0
                
                if size_multiplier >= self.level_wall_multiplier:
                    wall = {
                        'level': level,
                        'side': order['side'],
                        'price': order['price'],
                        'size': order['size'],
                        'multiplier': size_multiplier
                    }
                    walls.append(wall)
                    
                    if size_multiplier > max_multiplier:
                        max_multiplier = size_multiplier
        
        return {
            'walls_found': len(walls) > 0,
            'walls_count': len(walls),
            'max_wall_multiplier': max_multiplier,
            'walls_details': walls
        }
    
    def get_confidence_score(self, pattern_data: Dict) -> float:
        """Скоринг уверенности для перекоса стакана"""
        base_score = 0.3
        
        # Чем больше перекос, тем выше уверенность
        if 'imbalance_ratio' in pattern_data:
            imbalance_score = min(pattern_data['imbalance_ratio'] / self.bid_ask_ratio, 1.0)
            base_score += imbalance_score * 0.4
            
        # Чем больше стен, тем выше уверенность
        if 'walls_count' in pattern_data:
            walls_score = min(pattern_data['walls_count'] / 3.0, 1.0)
            base_score += walls_score * 0.2
            
        # Чем больше размер стены, тем выше уверенность
        if 'max_wall_multiplier' in pattern_data:
            wall_score = min(pattern_data['max_wall_multiplier'] / self.level_wall_multiplier, 1.0)
            base_score += wall_score * 0.1
            
        return min(base_score, 1.0)
