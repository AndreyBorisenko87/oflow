# Детектор D7: Спуфинг ловушка (SpoofPullTrap)

## 📋 Описание

Детектор обнаруживает манипуляции рынком через размещение и быстрое снятие крупных заявок.

## 🔧 Принцип работы

### Признаки паттерна:
- Крупная заявка размещается близко к рынку
- Быстро снимается (≤1 секунда)
- Не исполняется или исполняется минимально

### Параметры:
- `large_order_multiplier`: множитель крупной заявки (по умолчанию: 5.0)
- `distance_ticks`: расстояние от рынка в тиках (по умолчанию: 2)
- `cancel_time_window_ms`: окно снятия в мс (по умолчанию: 1000)
- `no_trade_executed`: требовать отсутствия исполнения (по умолчанию: true)

## 📊 Выходные данные

DataFrame с событиями:
- `ts_ns`: временная метка
- `exchange`: биржа
- `pattern_type`: "spoof_pull_trap"
- `confidence`: уверенность (0-1)
- `metadata`: дополнительные данные

## 🚀 Использование

```python
from oflow.blocks.detectors import D7SpoofPullTrap

detector = D7SpoofPullTrap(config)
events = detector.detect_with_progress(data)
```
