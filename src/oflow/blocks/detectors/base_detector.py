"""
Базовый класс для всех детекторов
"""

from abc import ABC, abstractmethod
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseDetector(ABC):
    """Базовый абстрактный класс для детекторов"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = self.__class__.__name__.replace('Detector', '')
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
    @abstractmethod
    def detect(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Основной метод детекции"""
        pass
    
    def detect_with_progress(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Детекция с отслеживанием прогресса"""
        start_time = time.time()
        self.logger.info(f"=== Начало детекции {self.name} ===")
        
        # Валидация данных
        self.logger.info("Этап 1/3: Валидация данных...")
        if not self.validate_data(data):
            self.logger.warning("Данные не прошли валидацию")
            return pd.DataFrame()
        
        # Детекция
        self.logger.info("Этап 2/3: Анализ паттернов...")
        result = self.detect(data)
        
        # Завершение
        duration = time.time() - start_time
        events_count = len(result) if not result.empty else 0
        self.logger.info(f"Этап 3/3: Завершено за {duration:.2f}с, найдено {events_count} событий")
        self.logger.info(f"=== Детекция {self.name} завершена ===")
        
        return result
    
    def get_name(self) -> str:
        """Название детектора"""
        return self.name
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Валидация входных данных"""
        required_keys = ['book_top', 'quotes', 'tape']
        
        for key in required_keys:
            if key not in data:
                self.logger.error(f"Отсутствует ключ: {key}")
                return False
            
            if not isinstance(data[key], pd.DataFrame):
                self.logger.error(f"Ключ {key} должен быть DataFrame")
                return False
            
            if data[key].empty:
                self.logger.warning(f"DataFrame {key} пустой")
                return False
        
        self.logger.info(f"Валидация данных пройдена успешно")
        return True
    
    def get_confidence_score(self, pattern_data: Dict) -> float:
        """Базовый скоринг уверенности"""
        return 0.5
    
    def save_events(self, events: pd.DataFrame, output_dir: str = "data/events") -> str:
        """Сохранение событий в файл"""
        if events.empty:
            self.logger.warning("Нет событий для сохранения")
            return ""
        
        output_path = Path(output_dir) / f"events_{self.name.lower()}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        events.to_parquet(output_path, engine="pyarrow", index=False)
        self.logger.info(f"События сохранены в {output_path}")
        
        return str(output_path)
