"""
Блок 03: Нормализация данных (Normalization)

Отвечает за нормализацию trades, depth и quotes данных
в единый формат для дальнейшей обработки.
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Tuple
try:
    from .base_block import BaseBlock
except ImportError:
    # Для прямого запуска
    from base_block import BaseBlock

logger = logging.getLogger(__name__)

class NormalizationBlock(BaseBlock):
    """Блок нормализации данных"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "NormalizationBlock"
    
    def run(self, data: Dict[str, Any], config: Dict) -> Dict[str, pd.DataFrame]:
        """Основной метод выполнения блока"""
        logger.info("Запуск блока нормализации данных...")
        
        # Получаем данные из предыдущего блока
        trades_df = data.get('trades_df')
        depth_df = data.get('depth_df')
        
        if trades_df is not None:
            trades_df = self.normalize_trades(trades_df, config)
        
        if depth_df is not None:
            depth_df = self.normalize_depth(depth_df, config)
        
        result = {
            'trades_df': trades_df,
            'depth_df': depth_df
        }
        
        logger.info("Нормализация данных завершена")
        return result
    
    def normalize_trades(self, trades_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Нормализация trades данных"""
        if trades_df.empty:
            return trades_df
        
        # Приводим к единому формату
        normalized = trades_df.copy()
        
        # Убеждаемся что есть обязательные колонки (адаптировано под реальные данные блока 02)
        required_columns = ['ts_ns', 'exchange', 'market', 'instrument', 'price', 'size']
        for col in required_columns:
            if col not in normalized.columns:
                logger.warning(f"Отсутствует колонка {col} в trades данных")
                return pd.DataFrame()
        
        # Сортируем по времени
        normalized = normalized.sort_values('ts_ns').reset_index(drop=True)
        
        return normalized
    
    def normalize_depth(self, depth_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Нормализация depth данных"""
        if depth_df.empty:
            return depth_df
        
        # Приводим к единому формату
        normalized = depth_df.copy()
        
        # Убеждаемся что есть обязательные колонки (адаптировано под реальные данные блока 02)
        required_columns = ['ts_ns', 'exchange', 'market', 'symbol', 'side', 'price', 'size']
        for col in required_columns:
            if col not in normalized.columns:
                logger.warning(f"Отсутствует колонка {col} в depth данных")
                return pd.DataFrame()
        
        # Сортируем по времени
        normalized = normalized.sort_values('ts_ns').reset_index(drop=True)
        
        return normalized
    
    def normalize_quotes(self, quotes_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Нормализация quotes данных"""
        if quotes_df.empty:
            return quotes_df
        
        # Приводим к единому формату
        normalized = quotes_df.copy()
        
        # Убеждаемся что есть обязательные колонки
        required_columns = ['ts_ns', 'exchange', 'symbol', 'best_bid', 'best_ask']
        for col in required_columns:
            if col not in normalized.columns:
                logger.warning(f"Отсутствует колонка {col} в quotes данных")
                return pd.DataFrame()
        
        # Сортируем по времени
        normalized = normalized.sort_values('ts_ns').reset_index(drop=True)
        
        return normalized

def normalize_trades(trades_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Функция для нормализации trades"""
    block = NormalizationBlock(config)
    return block.normalize_trades(trades_df, config)

def normalize_depth(depth_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Функция для нормализации depth"""
    block = NormalizationBlock(config)
    return block.normalize_depth(depth_df, config)

def normalize_quotes(quotes_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Функция для нормализации quotes"""
    block = NormalizationBlock(config)
    return block.normalize_quotes(quotes_df, config)


if __name__ == "__main__":
    """Запуск блока 03 для тестирования"""
    import pandas as pd
    import logging
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 ТЕСТ БЛОКА 03: Normalization")
    print("=" * 50)
    
    try:
        # Загружаем реальные данные от блока 02 (все 4 биржи)
        trades_df = pd.read_parquet("../../../data/normalized/trades.parquet")
        depth_df = pd.read_parquet("../../../data/normalized/book_top.parquet")
        
        print(f"📊 Анализ данных от всех бирж:")
        if not trades_df.empty:
            exchanges = trades_df['exchange'].unique()
            print(f"   - Биржи в trades: {exchanges}")
            print(f"   - Количество бирж: {len(exchanges)}")
        
        if not depth_df.empty:
            exchanges_depth = depth_df['exchange'].unique()
            print(f"   - Биржи в depth: {exchanges_depth}")
            print(f"   - Количество бирж: {len(exchanges_depth)}")
        
        print(f"✅ Загружены данные:")
        print(f"   - Trades: {len(trades_df)} строк")
        print(f"   - Depth: {len(depth_df)} строк")
        
        # Создаем блок
        config = {'test_mode': True}
        block = NormalizationBlock(config)
        
        # Тестовые данные
        test_data = {
            'trades_df': trades_df,
            'depth_df': depth_df
        }
        
        # Запускаем блок
        print("🚀 Запускаем блок 03...")
        result = block.run(test_data, config)
        
        # Анализируем результат
        norm_trades = result.get('trades_df')
        norm_depth = result.get('depth_df')
        
        print(f"✅ Блок 03 выполнен успешно!")
        print(f"📊 Результат:")
        print(f"   - Trades: {len(norm_trades) if norm_trades is not None else 0} строк")
        print(f"   - Depth: {len(norm_depth) if norm_depth is not None else 0} строк")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
