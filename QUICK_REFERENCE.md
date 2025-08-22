


# 🚀 QUICK REFERENCE - Scalping Strategy

## 🎯 **БЫСТРЫЙ СТАРТ**
```bash
# Запуск полного пайплайна (15 блоков)
cd src
python -c "from oflow.pipeline import run_full_pipeline; run_full_pipeline(config_dir='../configs', test_mode=False)"

# Запуск тестового пайплайна
python -c "from oflow.pipeline import run_test_pipeline; run_test_pipeline('../configs')"
```

## 🏗️ **КЛЮЧЕВЫЕ ФАЙЛЫ**
- **Главный исполнительный**: `src/oflow/pipeline.py`
- **Конфигурации**: `configs/` (15 YAML файлов)
- **Блоки**: `src/oflow/blocks/` (15 блоков)
- **Контекст**: `docs/context.md`

## 📚 **ОБЯЗАТЕЛЬНО ПРОЧИТАТЬ ПЕРЕД РАБОТОЙ**
1. `docs/context.md` - полный контекст проекта
2. `src/oflow/pipeline.py` - архитектура пайплайна
3. `.cursorrules` - правила работы

## ❌ **НЕ ДЕЛАТЬ**
- Создавать новые скрипты
- Игнорировать существующую архитектуру
- Действовать без изучения контекста

## ✅ **ДЕЛАТЬ**
- Использовать `run_full_pipeline()`
- Читать контекст перед любым действием
- Следовать существующей архитектуре
