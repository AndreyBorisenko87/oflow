# Детектор D5: Ложный вынос стопов (StopRunFailure)

## 📋 Описание

Детектор обнаруживает ситуации, когда цена выбивает стопы, но сразу возвращается обратно.

## 🔧 Принцип работы

### Признаки паттерна:
- Цена пробивает локальный экстремум
- В течение ≤5 секунд возврат на ≥70% движения
- В ленте фиксируются крупные сделки в обратную сторону

### Параметры:
- `break_extreme_confirm_ticks`: подтверждение пробития в тиках (по умолчанию: 2)
- `reversal_window_ms`: окно возврата в мс (по умолчанию: 5000)
- `reversal_threshold_ratio`: порог возврата (по умолчанию: 0.7)
- `opposite_large_trade`: требовать крупные сделки в обратную сторону (по умолчанию: true)

## 📊 Выходные данные

DataFrame с событиями:
- `ts_ns`: временная метка
- `exchange`: биржа
- `pattern_type`: "stop_run_failure"
- `confidence`: уверенность (0-1)
- `metadata`: дополнительные данные

## 🚀 Использование

```python
from oflow.blocks.detectors import D5StopRunFailure

detector = D5StopRunFailure(config)
events = detector.detect_with_progress(data)
```
