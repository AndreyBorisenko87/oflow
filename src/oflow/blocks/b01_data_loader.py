"""
Блок 01: Загрузчик данных (DataLoader)

Отвечает за загрузку конфигурации, определение путей к данным
и подготовку входных параметров для всех последующих блоков.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any
try:
    from .base_block import BaseBlock
except ImportError:
    # Для прямого запуска
    from base_block import BaseBlock

logger = logging.getLogger(__name__)

class DataLoaderBlock(BaseBlock):
    """Блок загрузки данных и конфигурации"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "DataLoaderBlock"
    
    def run(self, data: Dict[str, Any], config: Dict) -> Dict[str, Any]:
        """Основной метод выполнения блока"""
        logger.info("Запуск блока загрузки данных...")
        
        # Загрузка конфигурации
        config_data = self._load_config(config)
        
        # Определение путей к данным
        data_paths = self._resolve_data_paths(config_data)
        
        # Валидация данных
        validated_data = self._validate_data_paths(data_paths)
        
        # Подготовка результата
        result = {
            'config': config_data,
            'data_paths': validated_data,
            'start_time': data.get('start_time'),
            'end_time': data.get('end_time')
        }
        
        logger.info(f"Загружено {len(validated_data)} путей к данным")
        return result
    
    def _load_config(self, config: Dict) -> Dict:
        """Загрузка конфигурации"""
        config_dir = config.get('config_dir', 'configs')
        config_path = Path(config_dir) / 'config.yaml'
        sources_path = Path(config_dir) / 'sources.yaml'
        
        # Загружаем основную конфигурацию
        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
        
        # Загружаем источники данных
        with open(sources_path, 'r', encoding='utf-8') as f:
            sources = yaml.safe_load(f)
        
        # Объединяем конфигурации
        main_config['sources'] = sources
        return main_config
    
    def _resolve_data_paths(self, config: Dict) -> List[str]:
        """Разрешение путей к данным"""
        sources = config.get('sources', {})
        data_paths = []
        
        logger.info(f"Обрабатываем источники: {list(sources.keys())}")
        
        # Обрабатываем все источники
        for exchange, exchange_data in sources.items():
            if not isinstance(exchange_data, dict):
                logger.warning(f"Пропускаем {exchange}: не словарь")
                continue
                
            logger.info(f"Обрабатываем {exchange}: {list(exchange_data.keys())}")
            
            for market, market_data in exchange_data.items():
                if not isinstance(market_data, dict):
                    logger.warning(f"Пропускаем {exchange}.{market}: не словарь")
                    continue
                    
                logger.info(f"Обрабатываем {exchange}.{market}: {list(market_data.keys())}")
                
                for data_type, file_path in market_data.items():
                    if isinstance(file_path, str):
                        data_paths.append(file_path)
                        logger.info(f"Добавлен путь: {exchange}.{market}.{data_type} -> {file_path}")
                    else:
                        logger.warning(f"Пропускаем {exchange}.{market}.{data_type}: не строка")
        
        logger.info(f"Всего найдено путей: {len(data_paths)}")
        return data_paths
    
    def _validate_data_paths(self, data_paths: List[str]) -> List[str]:
        """Валидация путей к данным"""
        validated_paths = []
        
        for path in data_paths:
            if Path(path).exists():
                validated_paths.append(path)
            else:
                logger.warning(f"Путь не найден: {path}")
        
        return validated_paths

def run_data_loader(data: Dict[str, Any], config: Dict) -> Dict[str, Any]:
    """Функция для запуска блока загрузки данных"""
    block = DataLoaderBlock(config)
    return block.run_with_progress(data, config)


if __name__ == "__main__":
    """Запуск блока 01 для тестирования"""
    import sys
    import os
    
    # Добавляем путь для импорта base_block
    sys.path.append(os.path.dirname(__file__))
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    print("🧪 ТЕСТ БЛОКА 01: DataLoader")
    print("=" * 50)
    
    try:
        # Создаем блок
        config = {'config_dir': '../../../configs'}
        block = DataLoaderBlock(config)
        
        # Тестовые данные
        test_data = {}
        
        # Запускаем блок
        print("🚀 Запускаем блок 01...")
        result = block.run(test_data, config)
        
        # Анализируем результат
        data_paths = result.get('data_paths', [])
        print(f"✅ Блок 01 выполнен!")
        print(f"📊 Найдено путей к данным: {len(data_paths)}")
        
        if data_paths:
            print("📁 Пути к данным:")
            for i, path in enumerate(data_paths[:5], 1):  # Показываем первые 5
                print(f"   {i}. {path}")
            if len(data_paths) > 5:
                print(f"   ... и еще {len(data_paths) - 5} путей")
        
        print("✅ Блок 01 успешно протестирован!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
