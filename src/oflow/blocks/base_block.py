"""
Базовый класс для всех блоков обработки данных
"""

import logging
import time
from typing import Dict, Any
from abc import ABC, abstractmethod

class BaseBlock(ABC):
    """Базовый абстрактный класс для всех блоков"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(self.name)
        
        # Настройка логирования
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def run(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Основной метод выполнения блока - должен быть реализован в наследниках"""
        pass
    
    def run_with_progress(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Запуск блока с прогресс-логированием"""
        start_time = time.time()
        
        self.logger.info(f"🚀 Запуск блока {self.name}")
        
        try:
            # Выполнение основного метода
            result = self.run(data, config)
            
            # Логирование результата
            execution_time = time.time() - start_time
            self.logger.info(f"✅ Блок {self.name} завершен за {execution_time:.2f} сек")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Ошибка в блоке {self.name} после {execution_time:.2f} сек: {e}")
            raise
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Валидация входных данных"""
        if not isinstance(data, dict):
            self.logger.error("Входные данные должны быть словарем")
            return False
        return True
    
    def validate_output(self, result: Dict[str, Any]) -> bool:
        """Валидация выходных данных"""
        if not isinstance(result, dict):
            self.logger.error("Выходные данные должны быть словарем")
            return False
        return True
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Получение значения из конфигурации с fallback"""
        return self.config.get(key, default)
    
    def log_progress(self, current: int, total: int, stage: str = ""):
        """Логирование прогресса выполнения"""
        if total > 0:
            percentage = (current / total) * 100
            self.logger.info(f"📊 {stage}: {current}/{total} ({percentage:.1f}%)")
        else:
            self.logger.info(f"📊 {stage}: {current}")
