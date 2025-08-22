# Блок 02: Импорт сырых данных (ImportRaw)

## 📋 Описание

Блок отвечает за импорт и первичную нормализацию сырых рыночных данных с различных бирж. Обрабатывает trades, depth (книга заявок) и quotes.

## 🔧 Функциональность

### Основные возможности:
- **Импорт trades** - исторические сделки по инструментам
- **Импорт depth** - снапшоты книги заявок (bid/ask)
- **Импорт quotes** - котировки лучших цен
- **Нормализация данных** - приведение к единому формату
- **Фильтрация по времени** - обрезка данных по временным интервалам

### Поддерживаемые форматы:
- **Входной формат**: Parquet файлы с сырыми данными
- **Выходной формат**: Нормализованные DataFrame с унифицированными колонками

## 📊 Структура данных

### Trades (сделки):
```
ts_ns: int64           # Timestamp в наносекундах
exchange: string       # Название биржи
market: string         # Тип рынка (spot/futures)
instrument: string     # Инструмент (ETH-USDT)
base_symbol: string    # Базовый символ (ETHUSDT)
aggressor: string      # Сторона инициатора (buy/sell)
price: float64         # Цена сделки
size: float64          # Объем сделки
trade_id: string       # ID сделки
_src_file: string      # Исходный файл
```

### Depth (книга заявок):
```
ts_ns: int64           # Timestamp в наносекундах
exchange: string       # Название биржи
market: string         # Тип рынка
symbol: string         # Символ
side: string           # Сторона (bid/ask)
price: float64         # Цена уровня
size: float64          # Объем на уровне
action: string         # Действие (update/del)
_src_file: string      # Исходный файл
```

### Quotes (котировки):
```
ts_ns: int64           # Timestamp в наносекундах
exchange: string       # Название биржи
market: string         # Тип рынка
symbol: string         # Символ
best_bid: float64      # Лучший bid
best_ask: float64      # Лучший ask
mid: float64           # Средняя цена
spread: float64        # Спред
_src_file: string      # Исходный файл
```

## ⚙️ Конфигурация

```yaml
input_dir: "data/raw"              # Директория с сырыми данными
output_dir: "data/normalized"      # Директория для результатов
symbols: ["ETHUSDT"]               # Список символов для обработки
exchanges: ["binance", "bybit", "okx"]  # Список бирж
```

## 🚀 Использование

### В составе пайплайна:
```python
from oflow.pipeline import run_full_pipeline

# Запуск полного пайплайна
results = run_full_pipeline(
    config_dir='configs',
    test_mode=False
)
```

### Отдельный запуск блока:
```python
from oflow.blocks.import_raw import import_raw_data

# Конфигурация
config = {
    'input_dir': 'data/raw',
    'output_dir': 'data/normalized',
    'symbols': ['ETHUSDT'],
    'exchanges': ['binance']
}

# Данные для обработки
data = {
    'data_paths': [
        'path/to/trades.parquet',
        'path/to/depth.parquet',
        'path/to/quotes.parquet'
    ],
    'start_time': None,  # Опционально
    'end_time': None     # Опционально
}

# Выполнение
result = import_raw_data(data, config)
trades_df, depth_df = result
```

## 📈 Производительность

- **Trades**: ~43K записей за 0.6 сек
- **Depth**: ~1.3M записей за 70 сек  
- **Memory**: Эффективная обработка больших файлов
- **Progress**: Прогресс-логирование каждые 10K записей

## 🔍 Валидация

Блок автоматически проверяет:
- ✅ Наличие обязательных колонок
- ✅ Корректность типов данных
- ✅ Существование файлов данных
- ✅ Валидность timestamp

## 🐛 Обработка ошибок

- **Отсутствующие файлы**: Логирование предупреждений
- **Некорректные данные**: Пропуск с логированием
- **Ошибки парсинга**: Graceful fallback
- **Memory overflow**: Батчинг больших файлов

## 📝 Логирование

```
🚀 === НАЧАЛО ВЫПОЛНЕНИЯ ImportRawBlock ===
📋 Этап 1/3: Валидация данных...
✅ Валидация данных пройдена
⚙️  Этап 2/3: Обработка данных...
🔍 Нормализация depth: 24040 записей...
📊 Обработано 10000/24040 (41.6%)
✅ Нормализация depth завершена: 1319515 строк
✅ Обработка данных завершена успешно
⏱️  Этап 3/3: Завершено за 70.69с
🎉 === БЛОК ImportRawBlock ЗАВЕРШЕН ===
```

## 🔗 Связи с другими блоками

**Входные данные**: Сырые Parquet файлы  
**Выходные данные**: → Блок 04 (SyncTime)

## 📚 Примеры

### Обработка только trades:
```python
data = {
    'data_paths': ['data/binance-trades.parquet']
}
result = import_raw_data(data, config)
```

### Обработка с фильтрацией по времени:
```python
from datetime import datetime

data = {
    'data_paths': ['data/all-data.parquet'],
    'start_time': datetime(2025, 1, 1),
    'end_time': datetime(2025, 1, 2)
}
result = import_raw_data(data, config)
```
