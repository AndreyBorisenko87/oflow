"""
Блок 15: Экспорт разметки
Выгрузка событий/сделок в CSV для TradingView/ATAS
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import logging
from .base_block import BaseBlock

logger = logging.getLogger(__name__)

def export_to_csv(
    data_df: pd.DataFrame,
    output_path: Path,
    format_type: str = "tradingview",
    config: Dict = None
) -> None:
    """Экспорт в CSV формат с настройками"""
    logger.info(f"📤 Экспорт в CSV ({format_type})...")
    
    if data_df.empty:
        logger.warning("⚠️ DataFrame пустой, экспорт пропущен")
        return
    
    try:
        # Получаем настройки экспорта
        export_config = config.get('export', {}) if config else {}
        encoding = export_config.get('encoding', 'utf-8')
        separator = export_config.get('separator', ',')
        decimal_separator = export_config.get('decimal_separator', '.')
        date_format = export_config.get('date_format', '%Y-%m-%d %H:%M:%S')
        
        # Создаем директорию если не существует
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Форматируем данные для экспорта
        export_df = data_df.copy()
        
        # Обрабатываем числовые колонки (заменяем точку на запятую если нужно)
        if decimal_separator != '.':
            numeric_columns = export_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                export_df[col] = export_df[col].astype(str).str.replace('.', decimal_separator)
        
        # Экспортируем в CSV
        export_df.to_csv(
            output_path,
            index=False,
            sep=separator,
            encoding=encoding,
            date_format=date_format,
            float_format='%.6f'
        )
        
        # Логируем информацию о файле
        file_size_kb = output_path.stat().st_size / 1024
        logger.info(f"✅ CSV экспорт завершен: {output_path}")
        logger.info(f"📊 Размер файла: {file_size_kb:.1f} KB")
        logger.info(f"📈 Количество строк: {len(export_df)}")
        
        # Статистика по колонкам
        logger.info(f"🔤 Колонки: {', '.join(export_df.columns)}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при экспорте CSV: {e}")
        raise

def format_for_tradingview(events_df: pd.DataFrame) -> pd.DataFrame:
    """Форматирование для TradingView"""
    logger.info("Форматирование для TradingView...")
    
    if events_df.empty:
        return events_df
    
    # TODO: Реализовать форматирование для TradingView
    # 1. Структура колонок
    # 2. Формат времени
    # 3. Ценовые уровни
    
    # Заглушка
    tv_df = events_df.copy()
    tv_df['time'] = pd.to_datetime(tv_df['ts_ns'], unit='ns')
    
    return tv_df

def format_for_atas(events_df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Форматирование данных для ATAS"""
    logger.info("📊 Форматирование для ATAS...")
    
    if events_df.empty:
        logger.warning("⚠️ DataFrame пустой, форматирование пропущено")
        return events_df
    
    try:
        # Получаем настройки ATAS
        atas_config = config.get('atas', {}) if config else {}
        
        # Создаем копию для форматирования
        atas_df = events_df.copy()
        
        # Форматируем время для ATAS
        atas_df['time'] = pd.to_datetime(atas_df['ts_ns'], unit='ns')
        
        # Добавляем колонки специфичные для ATAS
        if 'price' in atas_df.columns:
            atas_df['open'] = atas_df['price']
            atas_df['high'] = atas_df['price']
            atas_df['low'] = atas_df['price']
            atas_df['close'] = atas_df['price']
        
        if 'size' in atas_df.columns:
            atas_df['volume'] = atas_df['size']
        
        # Выбираем нужные колонки для ATAS
        atas_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in atas_columns if col in atas_df.columns]
        
        atas_df = atas_df[available_columns]
        
        logger.info(f"✅ ATAS форматирование завершено: {len(atas_df)} строк")
        return atas_df
        
    except Exception as e:
        logger.error(f"❌ Ошибка при форматировании для ATAS: {e}")
        raise

# ---------- основной класс блока ----------
class ExportBlock(BaseBlock):
    """Блок экспорта данных в различные форматы"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.output_dir = config.get('output_dir', 'data/export')
        self.formats = config.get('formats', ['csv', 'tradingview', 'atas'])
        self.encoding = config.get('encoding', 'utf-8')
        self.separator = config.get('separator', ',')
        
    def validate_data(self, data: Dict[str, Any], config: Dict) -> bool:
        """Расширенная валидация для блока экспорта"""
        if not super().validate_data(data, config):
            return False
        
        # Проверяем наличие данных для экспорта
        if not data:
            self.logger.error("Нет данных для экспорта")
            return False
        
        # Проверяем, что есть хотя бы один DataFrame
        has_dataframes = any(
            isinstance(value, pd.DataFrame) and not value.empty 
            for value in data.values()
        )
        
        if not has_dataframes:
            self.logger.error("Нет непустых DataFrame для экспорта")
            return False
        
        self.logger.info("Валидация данных для экспорта пройдена")
        return True
    
    def run(self, data: Dict[str, Any], config: Dict) -> Dict[str, str]:
        """Основная логика экспорта"""
        self.logger.info(f"Экспорт данных в форматы: {', '.join(self.formats)}")
        
        # Создаем выходную директорию
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        export_results = {}
        
        # Экспортируем каждый тип данных
        for data_key, data_value in data.items():
            if not isinstance(data_value, pd.DataFrame) or data_value.empty:
                continue
            
            self.logger.info(f"Экспорт {data_key}: {len(data_value)} строк")
            
            # Экспорт в CSV
            if 'csv' in self.formats:
                csv_path = output_path / f"{data_key}.csv"
                try:
                    export_to_csv(data_value, csv_path, "standard", config)
                    export_results[f"{data_key}_csv"] = str(csv_path)
                    self.logger.info(f"CSV экспорт {data_key} завершен")
                except Exception as e:
                    self.logger.error(f"Ошибка CSV экспорта {data_key}: {e}")
            
            # Экспорт для TradingView
            if 'tradingview' in self.formats:
                tv_df = format_for_tradingview(data_value)
                if not tv_df.empty:
                    tv_path = output_path / f"{data_key}_tradingview.csv"
                    try:
                        export_to_csv(tv_df, tv_path, "tradingview", config)
                        export_results[f"{data_key}_tradingview"] = str(tv_path)
                        self.logger.info(f"TradingView экспорт {data_key} завершен")
                    except Exception as e:
                        self.logger.error(f"Ошибка TradingView экспорта {data_key}: {e}")
            
            # Экспорт для ATAS
            if 'atas' in self.formats:
                atas_df = format_for_atas(data_value, config)
                if not atas_df.empty:
                    atas_path = output_path / f"{data_key}_atas.csv"
                    try:
                        export_to_csv(atas_df, atas_path, "atas", config)
                        export_results[f"{data_key}_atas"] = str(atas_path)
                        self.logger.info(f"ATAS экспорт {data_key} завершен")
                    except Exception as e:
                        self.logger.error(f"Ошибка ATAS экспорта {data_key}: {e}")
        
        self.logger.info(f"Экспорт завершен. Создано файлов: {len(export_results)}")
        return export_results

# ---------- функции для обратной совместимости ----------
def export_data(data: Dict[str, pd.DataFrame], config: Dict) -> Dict[str, str]:
    """Функция для обратной совместимости"""
    block = ExportBlock(config)
    return block.run_with_progress(data, config)


if __name__ == "__main__":
    """Запуск блока 15 для тестирования"""
    import pandas as pd
    import logging
    import sys
    import os
    import time
    
    # Добавляем путь для импорта base_block
    sys.path.append(os.path.dirname(__file__))
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    print("🧪 ТЕСТ БЛОКА 15: Export")
    print("=" * 50)
    
    try:
        # Создаем тестовые данные для экспорта
        test_data = {
            'events': pd.DataFrame({
                'ts_ns': [1755774000000000000, 1755774001000000000, 1755774002000000000],
                'exchange': ['binance', 'bybit', 'okx'],
                'pattern_type': ['D1_LiquidityVacuum', 'D6_BookImbalance', 'D2_AbsorptionFlip'],
                'price': [4500.0, 4505.0, 4510.0],
                'confidence': [0.85, 0.72, 0.91]
            }),
            'trades': pd.DataFrame({
                'ts_ns': [1755774000000000000, 1755774001000000000],
                'exchange': ['binance', 'bybit'],
                'side': ['buy', 'sell'],
                'price': [4500.0, 4505.0],
                'size': [1.5, 2.0]
            })
        }
        
        print(f"✅ Созданы тестовые данные для экспорта:")
        for key, df in test_data.items():
            print(f"   - {key}: {len(df)} строк")
        
        # Конфигурация для экспорта
        config = {
            'output_dir': 'data/export',
            'formats': ['csv', 'tradingview', 'atas'],
            'encoding': 'utf-8',
            'separator': ',',
            'export': {
                'decimal_separator': '.',
                'date_format': '%Y-%m-%d %H:%M:%S'
            }
        }
        
        # Запускаем блок 15
        print("🚀 Запускаем блок 15: Экспорт данных...")
        start_time = time.time()
        
        results = export_data(test_data, config)
        
        execution_time = time.time() - start_time
        
        print(f"✅ Блок 15 выполнен за {execution_time:.2f} секунд!")
        print(f"📊 Результат экспорта:")
        
        # Анализируем результаты
        if results:
            print(f"   - Создано файлов: {len(results)}")
            
            for export_key, file_path in results.items():
                print(f"   - {export_key}: {file_path}")
        else:
            print("   - Файлы не созданы")
        
        print("✅ Блок 15 успешно протестирован!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()