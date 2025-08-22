# Блок 11: Канонический слой

## Обзор

Канонический слой предназначен для эффективного пакетного сохранения обработанных данных по дням. Это финальный этап конвейера, который организует все типы данных (quotes, book_top, tape, nbbo, basis, features, events) в структурированном формате для быстрого доступа и анализа.

## Ключевые возможности

### 1. Пакетное сохранение по дням
- Автоматическая группировка данных по датам
- Настраиваемый размер пакетов (1, 7, 30 дней)
- Оптимизация размеров файлов для эффективного хранения

### 2. Индексация и метаданные
- Автоматическое создание индексов файлов
- Метаданные с статистикой и схемой данных
- Быстрый поиск файлов по дате и типу данных

### 3. Поддержка всех типов данных
- **quotes**: Котировки лучших цен
- **book_top**: Топ-зона стакана заявок
- **tape**: Лента сделок с анализом агрессии
- **nbbo**: Лучшие цены по биржам
- **basis**: Базис спот-фьючерс
- **features**: Признаки паттернов для детекторов
- **events**: События паттернов
- **trades**: Сырые сделки
- **depth**: Сырые данные стакана

### 4. Оптимизация производительности
- Сжатие Snappy для быстрого чтения/записи
- Партиционирование по биржам и символам
- Контроль размеров файлов и памяти

## Структура данных

### Входные данные

Канонический слой принимает словарь DataFrame'ов:

```python
data_dict = {
    "quotes": quotes_df,           # Котировки
    "book_top": book_top_df,       # Топ-зона стакана
    "tape": tape_df,               # Лента сделок
    "nbbo": nbbo_df,               # NBBO
    "basis": basis_df,             # Базис
    "features": features_df,       # Признаки детекторов
    "events": events_df,           # События
    "trades": trades_df,           # Сырые сделки
    "depth": depth_df              # Сырые данные стакана
}
```

### Структура каталогов

```
data/canon/
├── quotes/
│   ├── quotes_20241201_20241201.parquet
│   ├── quotes_20241202_20241202.parquet
│   └── ...
├── book_top/
│   ├── book_top_20241201_20241201.parquet
│   └── ...
├── tape/
├── nbbo/
├── basis/
├── features/
├── events/
├── trades/
├── depth/
├── file_index.json              # Индекс всех файлов
└── metadata.json                # Метаданные и статистика
```

### Именование файлов

Формат: `{type}_{start_date}_{end_date}.parquet`

Примеры:
- `quotes_20241201_20241201.parquet` - данные за 1 день
- `tape_20241201_20241207.parquet` - данные за 7 дней
- `features_20241201_20241230.parquet` - данные за месяц

## Использование

### Базовое использование

```python
from oflow.blocks.canonical import run_canonical
import yaml

# Загрузка конфигурации
with open('configs/canonical.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Подготовка данных
data_dict = {
    "quotes": quotes_df,
    "features": features_df,
    # ... другие типы данных
}

# Сохранение в канонический слой
saved_files = run_canonical(data_dict, config, output_dir=Path("data/canon"))
```

### Продвинутое использование

```python
from oflow.blocks.canonical import CanonicalStorage

# Создание хранилища с настройками
storage = CanonicalStorage(config)

# Полное сохранение с прогресс-логированием
saved_files = storage.run_canonical_storage(data_dict, output_dir)

# Результат: словарь путей к сохраненным файлам
print(f"Сохранено {sum(len(files) for files in saved_files.values())} файлов")
```

### Legacy функции

```python
from oflow.blocks.canonical import batch_by_days, save_canonical

# Разбивка данных по дням
batched_data = batch_by_days(data_dict, batch_size_days=7)

# Простое сохранение без индексов
save_canonical(data_dict, output_dir, config)
```

## Конфигурация

### Основные параметры

```yaml
# Настройки пакетирования
batch:
  days: 1                         # Размер пакета в днях
  chunk_size: 10000               # Размер чанка для обработки
  
# Настройки сжатия
compression:
  type: "snappy"                  # Тип сжатия
  
# Настройки индексирования
indexing:
  enabled: true                   # Включить создание индексов
  create_file_index: true         # Создать индекс файлов
  
# Настройки метаданных
metadata:
  enabled: true                   # Включить создание метаданных
  include_statistics: true        # Включить статистику
```

### Настройки типов данных

Каждый тип данных имеет свои требования:

```yaml
data_types:
  quotes:
    required_columns: ["ts_ns", "exchange", "symbol", "bid", "ask"]
    partition_by: ["exchange", "symbol"]
  
  features:
    required_columns: ["ts_ns", "exchange", "symbol"]
    partition_by: ["exchange", "symbol"]
```

## Алгоритмы

### 1. Валидация данных
```python
def _validate_data(self, data_dict):
    # Проверка поддерживаемых типов данных
    # Проверка обязательных колонок
    # Проверка временной колонки ts_ns
    # Сортировка по времени
    # Добавление колонки date для группировки
```

### 2. Разбивка по дням
```python
def _batch_by_days(self, data_dict):
    # Группировка данных по датам
    # Создание пакетов по размеру batch_size_days
    # Вычисление ключей пакетов (YYYYMMDD_YYYYMMDD)
    # Объединение данных в пакеты
```

### 3. Сохранение пачками
```python
def _save_batches(self, batched_data, output_dir):
    # Создание структуры директорий
    # Сохранение каждого пакета в отдельный файл
    # Применение сжатия
    # Создание списка сохраненных файлов
```

### 4. Создание индексов и метаданных
```python
def _create_indexes_and_metadata(self, saved_files, output_dir):
    # Создание file_index.json с информацией о всех файлах
    # Создание metadata.json со статистикой и конфигурацией
    # Извлечение временных границ из имен файлов
    # Вычисление размеров файлов и общей статистики
```

## Экспорт данных

### Индекс файлов (file_index.json)

```json
{
  "created_at": "2024-12-01T12:00:00",
  "total_files": 45,
  "data_types": ["quotes", "features", "tape"],
  "files": {
    "quotes": [
      {
        "path": "quotes/quotes_20241201_20241201.parquet",
        "filename": "quotes_20241201_20241201.parquet",
        "size_bytes": 1048576,
        "date_range": {
          "start": "20241201",
          "end": "20241201"
        }
      }
    ]
  }
}
```

### Метаданные (metadata.json)

```json
{
  "version": "1.0",
  "created_at": "2024-12-01T12:00:00",
  "statistics": {
    "quotes": {
      "file_count": 7,
      "total_size_mb": 125.5
    }
  },
  "summary": {
    "total_files": 45,
    "total_size_mb": 890.2,
    "compression": "snappy",
    "batch_size_days": 1
  }
}
```

## Интеграция с другими блоками

### Входные данные от предыдущих блоков

```python
# Данные от блока 05: Лучшие цены
quotes_df = run_best_prices(...)

# Данные от блока 07: Лента сделок  
tape_df = run_trade_tape(...)

# Данные от блока 08: Кросс-биржевая агрегация
nbbo_df = run_cross_exchange(...)

# Данные от блока 09: Базис спот-фьючерс
basis_df = run_basis(...)

# Данные от блока 10: Фичи паттернов
features_df = run_features(...)

# Объединение всех данных
data_dict = {
    "quotes": quotes_df,
    "tape": tape_df,
    "nbbo": nbbo_df,
    "basis": basis_df,
    "features": features_df
}

# Сохранение в канонический слой
saved_files = run_canonical(data_dict, config)
```

### Использование сохраненных данных

```python
import pandas as pd
import json
from pathlib import Path

# Загрузка индекса файлов
with open("data/canon/file_index.json", 'r') as f:
    file_index = json.load(f)

# Поиск файлов по типу данных
quotes_files = file_index["files"]["quotes"]

# Загрузка данных за определенную дату
quotes_df = pd.read_parquet("data/canon/quotes/quotes_20241201_20241201.parquet")

# Загрузка всех файлов определенного типа
all_quotes = []
for file_info in quotes_files:
    df = pd.read_parquet(f"data/canon/{file_info['path']}")
    all_quotes.append(df)

combined_quotes = pd.concat(all_quotes, ignore_index=True)
```

## Производительность

### Оптимизации
- **Сжатие Snappy**: быстрое чтение/запись с разумным коэффициентом сжатия
- **Партиционирование**: группировка данных по биржам и символам
- **Пакетная обработка**: объединение мелких файлов в большие пакеты
- **Временная сортировка**: данные всегда отсортированы по времени

### Рекомендации по размерам пакетов
- **1 день**: для активного анализа и частых обновлений
- **7 дней**: для еженедельного анализа и умеренного объема данных
- **30 дней**: для архивного хранения и больших объемов данных

### Примерные размеры файлов
При сжатии Snappy:
- **quotes**: ~10-50 MB на день
- **features**: ~5-20 MB на день  
- **tape**: ~20-100 MB на день
- **book_top**: ~50-200 MB на день

## Логирование

### Уровни логирования
- **INFO**: основные этапы сохранения и статистика
- **DEBUG**: детальная информация о файлах
- **WARNING**: проблемы с данными или конфигурацией
- **ERROR**: ошибки сохранения

### Прогресс-логирование

```
=== Начало сохранения в канонический слой ===
Этап 1/4: Валидация входных данных...
Валидирован quotes: 142857 записей
Валидирован features: 12345 записей

Этап 2/4: Разбивка данных по дням...
Обработка quotes: 7 дней данных
Создано 7 пакетов для quotes

Этап 3/4: Сохранение данных пачками...
Сохранение quotes: 7 пакетов
Завершено сохранение quotes: 7 файлов

Этап 4/4: Создание индексов и метаданных...
Индекс файлов сохранен: data/canon/file_index.json
Метаданные сохранены: data/canon/metadata.json

Сохранение завершено за 12.34с
Сохранено 45 файлов по 5 типам данных
=== Сохранение в канонический слой завершено ===
```

## Мониторинг и отладка

### Проверка целостности данных

```python
# Загрузка и проверка метаданных
with open("data/canon/metadata.json", 'r') as f:
    metadata = json.load(f)

print(f"Всего файлов: {metadata['summary']['total_files']}")
print(f"Общий размер: {metadata['summary']['total_size_mb']} MB")

# Проверка наличия всех файлов
with open("data/canon/file_index.json", 'r') as f:
    file_index = json.load(f)

missing_files = []
for data_type, files in file_index["files"].items():
    for file_info in files:
        file_path = Path("data/canon") / file_info["path"]
        if not file_path.exists():
            missing_files.append(str(file_path))

if missing_files:
    print(f"Отсутствующие файлы: {missing_files}")
```

### Анализ производительности

```python
# Анализ размеров файлов по типам данных
for data_type, stats in metadata["statistics"].items():
    avg_size = stats["total_size_mb"] / stats["file_count"]
    print(f"{data_type}: {stats['file_count']} файлов, "
          f"средний размер {avg_size:.1f} MB")
```

## Troubleshooting

### Частые проблемы

1. **Отсутствуют обязательные колонки**
   ```
   WARNING: Отсутствуют обязательные колонки для quotes: ['bid', 'ask']
   ```
   **Решение**: Проверить схему входных данных

2. **Отсутствует временная колонка**
   ```
   WARNING: Отсутствует временная колонка ts_ns для quotes
   ```
   **Решение**: Добавить колонку ts_ns с наносекундами

3. **Пустые DataFrame**
   ```
   WARNING: Пустой DataFrame для типа: features
   ```
   **Решение**: Проверить результаты предыдущих блоков

4. **Ошибки сжатия**
   ```
   ERROR: Failed to save with snappy compression
   ```
   **Решение**: Установить pyarrow или изменить тип сжатия

### Восстановление данных

```python
# Поиск поврежденных файлов
import pandas as pd

corrupted_files = []
for data_type, files in file_index["files"].items():
    for file_info in files:
        try:
            file_path = f"data/canon/{file_info['path']}"
            df = pd.read_parquet(file_path)
            if df.empty:
                corrupted_files.append(file_path)
        except Exception as e:
            corrupted_files.append(file_path)
            print(f"Поврежден файл {file_path}: {e}")

print(f"Найдено поврежденных файлов: {len(corrupted_files)}")
```

## Будущие улучшения

1. **Параллельная обработка**: использование Dask для больших объемов данных
2. **Автоматическая очистка**: удаление старых временных файлов
3. **Инкрементальные обновления**: добавление новых данных без пересохранения
4. **Контрольные суммы**: проверка целостности файлов
5. **Сжатие архивов**: дополнительное сжатие старых данных

## Примеры использования

### Пример 1: Ежедневное сохранение

```python
# Конфигурация для ежедневного сохранения
config = {
    'batch': {'days': 1},
    'compression': {'type': 'snappy'},
    'indexing': {'enabled': True},
    'metadata': {'enabled': True}
}

# Сохранение данных за день
daily_data = {
    "quotes": daily_quotes_df,
    "features": daily_features_df
}

saved_files = run_canonical(daily_data, config, Path("data/canon/daily"))
```

### Пример 2: Архивное сохранение

```python
# Конфигурация для архивного сохранения
config = {
    'batch': {'days': 30},
    'compression': {'type': 'gzip'},
    'indexing': {'enabled': True},
    'metadata': {'enabled': True}
}

# Сохранение данных за месяц
monthly_data = load_monthly_data()
saved_files = run_canonical(monthly_data, config, Path("data/canon/archive"))
```

### Пример 3: Экспорт для анализа

```python
# Загрузка данных за период
start_date = "20241201"
end_date = "20241207"

# Поиск подходящих файлов
matching_files = find_files_by_date_range(file_index, start_date, end_date)

# Загрузка и объединение
combined_data = {}
for data_type in ["quotes", "features", "tape"]:
    dfs = []
    for file_path in matching_files[data_type]:
        df = pd.read_parquet(file_path)
        dfs.append(df)
    combined_data[data_type] = pd.concat(dfs, ignore_index=True)

# Анализ загруженных данных
analyze_patterns(combined_data)
```
