#!/usr/bin/env python
"""
Скрипт запуска полного пайплайна Order Flow
"""

import sys
import os
from pathlib import Path

# Добавляем src в sys.path для корректных импортов
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Теперь можем импортировать модули
from oflow.pipeline import run_full_pipeline, load_config, setup_logging

def main():
    """Главная функция запуска пайплайна"""
    print("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ORDER FLOW")
    print("=" * 60)
    
    try:
        # Загружаем конфигурацию
        config = load_config()
        
        # Настраиваем логирование
        setup_logging(config)
        
        # Запускаем полный пайплайн
        print("🔄 Запуск полного пайплайна...")
        results = run_full_pipeline()
        
        print("✅ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
        print("=" * 60)
        
        # Выводим результаты
        if results:
            print("📊 РЕЗУЛЬТАТЫ ПАЙПЛАЙНА:")
            for step, result in results.items():
                if hasattr(result, '__len__'):
                    print(f"   - {step}: {len(result)} записей")
                else:
                    print(f"   - {step}: выполнен")
        
        return True
        
    except KeyboardInterrupt:
        print("\n❌ Пайплайн прерван пользователем")
        return False
    except Exception as e:
        print(f"❌ ОШИБКА В ПАЙПЛАЙНЕ: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
