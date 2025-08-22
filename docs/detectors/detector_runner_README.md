# Запуск детекторов (DetectorRunner)

## 📋 Описание

Модуль для запуска всех детекторов паттернов с прогресс-логированием и конфигурацией.

## 🔧 Основные функции

### `run_detectors(detectors, data, use_progress=True)`
Запускает все детекторы на данных с отслеживанием прогресса.

### `create_detectors_from_config(config)`
Создает список детекторов из конфигурационного файла.

## 📊 Параметры

- `detectors`: список экземпляров детекторов
- `data`: данные для анализа
- `use_progress`: включить прогресс-логирование
- `config`: конфигурация детекторов

## 🚀 Использование

```python
from oflow.blocks.detectors import run_detectors, create_detectors_from_config

# Создание детекторов из конфига
detectors = create_detectors_from_config(config)

# Запуск всех детекторов
results = run_detectors(detectors, data, use_progress=True)
```

## 📝 Логирование

Автоматическое логирование прогресса по биржам и временным группам.
