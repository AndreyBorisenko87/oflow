# Исходный код (Source Code)

## 📋 Описание

Папка содержит весь исходный код Scalping Strategy.

## 📁 Структура папок

### **oflow/** - Основной модуль
- **pipeline.py** - Главный pipeline выполнения
- **__init__.py** - Экспорты модуля
- **blocks/** - Все 15 блоков обработки
- **detectors/** - Детекторы паттернов

### **blocks/** - Блоки обработки
- **01_data_loader.py** - Загрузчик данных
- **02_import_raw.py** - Импорт сырых данных
- **03_normalization.py** - Нормализация
- **04_sync_time.py** - Синхронизация времени
- **05_best_prices.py** - Лучшие цены
- **06_book_top.py** - Топ книги заявок
- **07_trade_tape.py** - Лента сделок
- **08_cross_exchange.py** - Кросс-биржевой анализ
- **09_basis.py** - Базис
- **10_features.py** - Генерация признаков
- **11_canonical.py** - Канонические данные
- **12_detectors.py** - Детекторы
- **13_scanner.py** - Сканер
- **14_backtest.py** - Бэктестирование
- **15_export.py** - Экспорт

### **detectors/** - Детекторы паттернов
- **base_detector.py** - Базовый класс
- **d1_liquidity_vacuum_break.py** - D1 детектор
- **d2_absorption_flip.py** - D2 детектор
- **d3_iceberg_fade.py** - D3 детектор
- **d4_stop_run_continuation.py** - D4 детектор
- **d5_stop_run_failure.py** - D5 детектор
- **d6_book_imbalance.py** - D6 детектор
- **d7_spoof_pull_trap.py** - D7 детектор
- **d8_momentum_ignition.py** - D8 детектор

## 🏗️ Архитектура

### **Принципы:**
- Модульная структура
- Наследование от BaseBlock
- Единый интерфейс выполнения
- Прогресс-логирование

### **Паттерны:**
- Strategy Pattern для детекторов
- Template Method для блоков
- Factory для создания объектов
- Observer для логирования

## 🔧 Разработка

### **Создание нового блока:**
```python
from .base_block import BaseBlock

class NewBlock(BaseBlock):
    def __init__(self, config):
        super().__init__(config)
        self.name = "NewBlock"
    
    def run(self, data, config):
        # Логика блока
        return result
```

### **Создание детектора:**
```python
from .base_detector import BaseDetector

class NewDetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        self.name = "NewDetector"
    
    def detect(self, data):
        # Логика детектора
        return events
```

## 📊 Тестирование

```bash
# Запуск всех тестов
python -m pytest tests/

# Тест конкретного блока
python -m pytest tests/test_block_02.py

# Тест с покрытием
python -m pytest --cov=src tests/
```

## 📝 Примечания

- Все блоки наследуют от BaseBlock
- Детекторы наследуют от BaseDetector
- Единый формат логирования
- Автоматическое создание логов
