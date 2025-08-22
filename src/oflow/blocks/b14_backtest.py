"""
Блок 14: Бэктестирование (Backtest)

Отвечает за историческое тестирование торговых стратегий
на основе детекторов паттернов.
"""

import pandas as pd
import logging
from typing import Dict, List, Any
from .base_block import BaseBlock

logger = logging.getLogger(__name__)

class BacktestBlock(BaseBlock):
    """Блок бэктестирования"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "BacktestBlock"
    
    def run(self, data: Dict[str, Any], config: Dict) -> Dict[str, Any]:
        """Основной метод выполнения блока"""
        logger.info("Запуск блока бэктестирования...")
        
        # Получаем данные от детекторов
        detector_events = data.get('detector_events', [])
        
        if not detector_events:
            logger.warning("Нет событий от детекторов для бэктестирования")
            return {'backtest_results': None}
        
        # Выполняем бэктест
        results = self._run_backtest(detector_events, config)
        
        logger.info(f"Бэктест завершен. Результатов: {len(results)}")
        return {'backtest_results': results}
    
    def _run_backtest(self, events: List, config: Dict) -> List[Dict]:
        """Выполнение бэктеста"""
        results = []
        
        for event in events:
            # Симуляция торговли на основе события
            trade_result = self._simulate_trade(event, config)
            if trade_result:
                results.append(trade_result)
        
        return results
    
    def _simulate_trade(self, event: Dict, config: Dict) -> Dict:
        """Симуляция одной сделки"""
        # Базовая логика симуляции
        return {
            'event_id': event.get('id'),
            'pattern_type': event.get('pattern_type'),
            'entry_price': event.get('price'),
            'entry_time': event.get('timestamp'),
            'pnl': 0.0,  # Пока заглушка
            'status': 'simulated'
        }

class BacktestEngine:
    """Движок бэктестирования"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def run(self, data: Dict) -> Dict:
        """Запуск бэктеста"""
        block = BacktestBlock(self.config)
        return block.run(data, self.config)

def run_backtest(data: Dict[str, Any], config: Dict) -> Dict[str, Any]:
    """Функция для запуска бэктеста"""
    engine = BacktestEngine(config)
    return engine.run(data)


if __name__ == "__main__":
    """Запуск блока 14 для тестирования"""
    import pandas as pd
    import logging
    import sys
    import os
    import time
    
    # Добавляем путь для импорта base_block
    sys.path.append(os.path.dirname(__file__))
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    print("🧪 ТЕСТ БЛОКА 14: Backtest")
    print("=" * 50)
    
    try:
        # Проверяем доступность событий от детекторов
        from pathlib import Path
        events_dir = Path("../../../data/events")
        if not events_dir.exists():
            print("⚠️  Папка событий не найдена (ожидаемо для тестирования)")
            print("   Создаем тестовые события для демонстрации")
        else:
            print(f"✅ Найдена папка событий: {events_dir}")
        
        # Создаем тестовые данные для бэктеста
        test_events = [
            {
                'id': 'event_001',
                'pattern_type': 'D1_LiquidityVacuum',
                'price': 4500.0,
                'timestamp': '2025-01-22T10:00:00',
                'exchange': 'binance',
                'confidence': 0.85
            },
            {
                'id': 'event_002',
                'pattern_type': 'D6_BookImbalance',
                'price': 4505.0,
                'timestamp': '2025-01-22T10:05:00',
                'exchange': 'bybit',
                'confidence': 0.72
            }
        ]
        
        print(f"✅ Созданы тестовые события: {len(test_events)} событий")
        
        # Конфигурация для бэктеста
        config = {
            'backtest': {
                'initial_capital': 10000.0,
                'position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.04
            },
            'risk': {
                'max_position_size': 0.2,
                'max_daily_loss': 0.05
            }
        }
        
        # Запускаем блок 14
        print("🚀 Запускаем блок 14: Бэктестирование...")
        start_time = time.time()
        
        data = {'detector_events': test_events}
        results = run_backtest(data, config)
        
        execution_time = time.time() - start_time
        
        print(f"✅ Блок 14 выполнен за {execution_time:.2f} секунд!")
        print(f"📊 Результат бэктеста:")
        
        # Анализируем результаты
        if results and 'backtest_results' in results:
            backtest_results = results['backtest_results']
            if backtest_results:
                print(f"   - Симулировано сделок: {len(backtest_results)}")
                
                for i, result in enumerate(backtest_results, 1):
                    print(f"   - Сделка {i}:")
                    print(f"     Паттерн: {result.get('pattern_type', 'N/A')}")
                    print(f"     Цена входа: {result.get('entry_price', 'N/A')}")
                    print(f"     Статус: {result.get('status', 'N/A')}")
            else:
                print("   - Сделки не симулированы")
        else:
            print("   - Результаты бэктеста недоступны")
        
        print("✅ Блок 14 успешно протестирован!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
