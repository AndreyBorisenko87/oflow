# Детектор D6: Сильный перекос стакана (BookImbalance)

## 📋 Описание

Детектор обнаруживает ситуации, когда одна сторона стакана многократно перевешивает другую.

## 🔧 Принцип работы

### Признаки паттерна:
- Сумма объёмов топ-5 bid >> топ-5 ask (или наоборот) в X раз
- Объем на одном уровне > средней стены в Y раз
- Нет мгновенного снятия этой заявки

### Параметры:
- `bid_ask_ratio`: соотношение bid/ask (по умолчанию: 3.0)
- `level_wall_multiplier`: множитель стены (по умолчанию: 5.0)
- `levels_depth`: глубина анализа уровней (по умолчанию: 5)

## 📊 Выходные данные

DataFrame с событиями:
- `ts_ns`: временная метка
- `exchange`: биржа
- `pattern_type`: "book_imbalance"
- `confidence`: уверенность (0-1)
- `metadata`: дополнительные данные

## 🚀 Использование

```python
from oflow.blocks.detectors import D6BookImbalance

detector = D6BookImbalance(config)
events = detector.detect_with_progress(data)
```
