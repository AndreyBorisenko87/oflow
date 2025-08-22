"""
Блок 13: Сканер
Обход всех файлов/дат/бирж и запуск выбранных детекторов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import logging
import time
import json
from datetime import datetime, timedelta
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from .detectors.detector_runner import run_detectors, create_detectors_from_config

logger = logging.getLogger(__name__)

class DataScanner:
    """Сканер данных для автоматического запуска детекторов"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.supported_data_types = config.get('data_types', ['quotes', 'book_top', 'tape', 'nbbo', 'basis', 'features'])
        self.batch_size = config.get('processing', {}).get('batch_size', 10)
        self.max_workers = config.get('processing', {}).get('max_workers', 4)
        self.memory_limit_gb = config.get('processing', {}).get('memory_limit_gb', 8)
        self.date_range = config.get('filters', {}).get('date_range', {})
        self.exchanges = config.get('filters', {}).get('exchanges', [])
        self.symbols = config.get('filters', {}).get('symbols', [])
        
    def scan_canonical_files(self, data_dir: Path) -> Dict[str, List[Path]]:
        """Сканирование канонических файлов данных"""
        start_time = time.time()
        logger.info("=== Начало сканирования файлов ===")
        logger.info(f"Сканирование директории: {data_dir}")
        
        if not data_dir.exists():
            logger.error(f"Директория не найдена: {data_dir}")
            return {}
        
        # Загружаем индекс файлов, если он существует
        index_path = data_dir / "file_index.json"
        file_index = None
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    file_index = json.load(f)
                logger.info("✓ Загружен индекс файлов")
            except Exception as e:
                logger.warning(f"Не удалось загрузить индекс файлов: {e}")
        
        # Структура найденных файлов
        found_files = {data_type: [] for data_type in self.supported_data_types}
        
        # Прогресс сканирования
        if file_index and 'files' in file_index:
            # Используем индекс файлов
            logger.info("Использование индекса файлов для быстрого поиска...")
            
            for data_type in self.supported_data_types:
                if data_type in file_index['files']:
                    for file_info in file_index['files'][data_type]:
                        file_path = data_dir / file_info['path']
                        if file_path.exists() and self._passes_filters(file_info, file_path):
                            found_files[data_type].append(file_path)
        else:
            # Ручное сканирование
            logger.info("Ручное сканирование файлов...")
            
            for data_type in self.supported_data_types:
                type_dir = data_dir / data_type
                if type_dir.exists():
                    parquet_files = list(type_dir.glob("*.parquet"))
                    logger.info(f"Найдено {len(parquet_files)} файлов {data_type}")
                    
                    for file_path in parquet_files:
                        if self._passes_filters({}, file_path):
                            found_files[data_type].append(file_path)
        
        # Статистика
        total_files = sum(len(files) for files in found_files.values())
        duration = time.time() - start_time
        
        logger.info(f"=== Сканирование завершено за {duration:.2f}с ===")
        logger.info(f"Найдено файлов по типам:")
        for data_type, files in found_files.items():
            if files:
                logger.info(f"  - {data_type}: {len(files)} файлов")
        logger.info(f"Всего файлов: {total_files}")
        
        return found_files
    
    def _passes_filters(self, file_info: Dict, file_path: Path) -> bool:
        """Проверка файла на соответствие фильтрам"""
        # Фильтр по дате
        if self.date_range:
            if not self._check_date_filter(file_info, file_path):
                return False
        
        # Фильтр по биржам
        if self.exchanges:
            if not self._check_exchange_filter(file_path):
                return False
        
        # Фильтр по символам
        if self.symbols:
            if not self._check_symbol_filter(file_path):
                return False
        
        return True
    
    def _check_date_filter(self, file_info: Dict, file_path: Path) -> bool:
        """Проверка фильтра по дате"""
        start_date = self.date_range.get('start')
        end_date = self.date_range.get('end')
        
        if not start_date and not end_date:
            return True
        
        # Извлекаем дату из имени файла или метаданных
        file_date = None
        
        if 'date_range' in file_info:
            file_date = file_info['date_range']['start']
        else:
            # Парсим дату из имени файла (формат: type_YYYYMMDD_YYYYMMDD.parquet)
            filename = file_path.name
            date_match = re.search(r'_(\d{8})_\d{8}\.parquet$', filename)
            if date_match:
                file_date = date_match.group(1)
        
        if file_date:
            if start_date and file_date < start_date:
                return False
            if end_date and file_date > end_date:
                return False
        
        return True
    
    def _check_exchange_filter(self, file_path: Path) -> bool:
        """Проверка фильтра по биржам"""
        # Простая проверка: если название биржи есть в пути файла
        file_str = str(file_path).lower()
        return any(exchange.lower() in file_str for exchange in self.exchanges)
    
    def _check_symbol_filter(self, file_path: Path) -> bool:
        """Проверка фильтра по символам"""
        # Простая проверка: если название символа есть в пути файла
        file_str = str(file_path).lower()
        return any(symbol.lower() in file_str for symbol in self.symbols)
    
    def load_batch_data(self, file_batch: Dict[str, List[Path]]) -> Dict[str, pd.DataFrame]:
        """Загрузка пакета данных из файлов"""
        logger.info(f"Загрузка пакета данных...")
        
        batch_data = {}
        
        for data_type, files in file_batch.items():
            if not files:
                continue
                
            logger.info(f"Загрузка {data_type}: {len(files)} файлов")
            
            # Загрузка всех файлов типа и объединение
            dfs = []
            for i, file_path in enumerate(files):
                if i % 5 == 0:  # Логируем каждые 5 файлов
                    logger.info(f"  Прогресс {data_type}: {i+1}/{len(files)} файлов")
                
                try:
                    df = pd.read_parquet(file_path)
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"Ошибка чтения {file_path}: {e}")
            
            # Объединение данных
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df = combined_df.sort_values('ts_ns').reset_index(drop=True)
                batch_data[data_type] = combined_df
                logger.info(f"✓ {data_type}: {len(combined_df)} записей загружено")
            else:
                logger.warning(f"○ {data_type}: нет данных для загрузки")
        
        return batch_data
    
    def create_batches(self, found_files: Dict[str, List[Path]]) -> List[Dict[str, List[Path]]]:
        """Создание пакетов файлов для обработки"""
        logger.info(f"Создание пакетов для обработки (размер пакета: {self.batch_size})...")
        
        # Определяем общее количество файлов
        all_files = []
        for data_type, files in found_files.items():
            for file_path in files:
                all_files.append((data_type, file_path))
        
        # Создание пакетов
        batches = []
        for i in range(0, len(all_files), self.batch_size):
            batch_files = all_files[i:i + self.batch_size]
            
            # Группировка по типам данных
            batch = {data_type: [] for data_type in self.supported_data_types}
            for data_type, file_path in batch_files:
                batch[data_type].append(file_path)
            
            # Добавляем только непустые пакеты
            if any(files for files in batch.values()):
                batches.append(batch)
        
        logger.info(f"Создано {len(batches)} пакетов для обработки")
        return batches
    
    def process_batch(self, batch: Dict[str, List[Path]], detectors: List, output_dir: Path, batch_num: int) -> Dict[str, pd.DataFrame]:
        """Обработка одного пакета данных"""
        logger.info(f"=== Обработка пакета {batch_num} ===")
        
        try:
            # Загрузка данных пакета
            batch_data = self.load_batch_data(batch)
            
            if not batch_data:
                logger.warning(f"Пакет {batch_num}: нет данных для обработки")
                return {}
            
            # Валидация данных для детекторов
            required_types = ['book_top', 'quotes', 'tape']
            missing_types = [t for t in required_types if t not in batch_data or batch_data[t].empty]
            
            if missing_types:
                logger.warning(f"Пакет {batch_num}: отсутствуют данные {missing_types}, пропускаем детекцию")
                return {}
            
            # Запуск детекторов
            logger.info(f"Пакет {batch_num}: запуск {len(detectors)} детекторов...")
            batch_output_dir = output_dir / f"batch_{batch_num:03d}"
            results = run_detectors(detectors, batch_data, batch_output_dir, use_progress=True)
            
            logger.info(f"✓ Пакет {batch_num}: обработан")
            return results
            
        except Exception as e:
            logger.error(f"✗ Ошибка обработки пакета {batch_num}: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_full_scan(self, data_dir: Path, output_dir: Path, detectors_config: Dict) -> Dict[str, pd.DataFrame]:
        """Запуск полного сканирования и детекции"""
        start_time = time.time()
        logger.info("🚀 Запуск полного сканирования и детекции")
        logger.info("=" * 80)
        
        # Этап 1: Сканирование файлов
        logger.info("Этап 1/4: Сканирование канонических файлов...")
        found_files = self.scan_canonical_files(data_dir)
        
        if not any(files for files in found_files.values()):
            logger.warning("Нет файлов для обработки")
            return {}
        
        # Этап 2: Создание детекторов
        logger.info("Этап 2/4: Создание детекторов...")
        detectors = create_detectors_from_config(detectors_config)
        
        if not detectors:
            logger.warning("Нет детекторов для запуска")
            return {}
        
        # Этап 3: Создание пакетов
        logger.info("Этап 3/4: Создание пакетов данных...")
        batches = self.create_batches(found_files)
        
        # Этап 4: Обработка пакетов
        logger.info("Этап 4/4: Обработка пакетов...")
        all_results = {}
        
        if self.max_workers > 1:
            # Параллельная обработка
            logger.info(f"Параллельная обработка {len(batches)} пакетов ({self.max_workers} воркеров)...")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_batch, batch, detectors, output_dir, i+1): i+1 
                    for i, batch in enumerate(batches)
                }
                
                for future in as_completed(futures):
                    batch_num = futures[future]
                    try:
                        batch_results = future.result()
                        
                        # Объединение результатов
                        for detector_name, events in batch_results.items():
                            if detector_name not in all_results:
                                all_results[detector_name] = []
                            if not events.empty:
                                all_results[detector_name].append(events)
                        
                        logger.info(f"✓ Пакет {batch_num} завершен")
                        
                    except Exception as e:
                        logger.error(f"✗ Ошибка в пакете {batch_num}: {e}")
        else:
            # Последовательная обработка
            logger.info(f"Последовательная обработка {len(batches)} пакетов...")
            
            for i, batch in enumerate(batches, 1):
                batch_results = self.process_batch(batch, detectors, output_dir, i)
                
                # Объединение результатов
                for detector_name, events in batch_results.items():
                    if detector_name not in all_results:
                        all_results[detector_name] = []
                    if not events.empty:
                        all_results[detector_name].append(events)
        
        # Финализация результатов
        final_results = {}
        for detector_name, events_list in all_results.items():
            if events_list:
                combined_events = pd.concat(events_list, ignore_index=True)
                combined_events = combined_events.sort_values('ts_ns').reset_index(drop=True)
                final_results[detector_name] = combined_events
            else:
                final_results[detector_name] = pd.DataFrame()
        
        # Сохранение финальных результатов
        self._save_final_results(final_results, output_dir)
        
        # Итоговая статистика
        duration = time.time() - start_time
        total_events = sum(len(df) for df in final_results.values())
        successful_detectors = sum(1 for df in final_results.values() if not df.empty)
        
        logger.info("=" * 80)
        logger.info("📊 ИТОГОВАЯ СТАТИСТИКА СКАНИРОВАНИЯ:")
        logger.info(f"⏱️  Время выполнения: {duration:.2f}с")
        logger.info(f"📁 Обработано пакетов: {len(batches)}")
        logger.info(f"🔍 Успешных детекторов: {successful_detectors}/{len(detectors)}")
        logger.info(f"🎯 Всего событий найдено: {total_events}")
        
        for detector_name, events_df in final_results.items():
            if not events_df.empty:
                logger.info(f"  - {detector_name}: {len(events_df)} событий")
        
        logger.info("🎉 Полное сканирование завершено!")
        
        return final_results
    
    def _save_final_results(self, results: Dict[str, pd.DataFrame], output_dir: Path) -> None:
        """Сохранение финальных результатов"""
        logger.info("Сохранение финальных результатов...")
        
        summary_dir = output_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        for detector_name, events_df in results.items():
            if not events_df.empty:
                summary_path = summary_dir / f"final_{detector_name.lower()}.parquet"
                events_df.to_parquet(summary_path, engine="pyarrow", index=False)
                logger.info(f"✓ Сохранен финальный результат {detector_name}: {summary_path}")
        
        # Создание общего отчета
        summary_report = {
            "scan_completed_at": datetime.now().isoformat(),
            "total_detectors": len(results),
            "detectors_with_events": sum(1 for df in results.values() if not df.empty),
            "total_events": sum(len(df) for df in results.values()),
            "detector_summary": {
                name: {"event_count": len(df), "has_events": not df.empty}
                for name, df in results.items()
            }
        }
        
        report_path = summary_dir / "scan_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"✓ Отчет сканирования сохранен: {report_path}")

def run_scanner(
    config: Dict,
    detectors_config: Dict,
    data_dir: Path = Path("data/canon"),
    output_dir: Path = Path("data/events")
) -> Dict[str, pd.DataFrame]:
    """Основная функция сканера"""
    logger.info("Запуск автоматического сканера данных...")
    
    # Создание сканера
    scanner = DataScanner(config)
    
    # Запуск полного сканирования
    results = scanner.run_full_scan(data_dir, output_dir, detectors_config)
    
    logger.info("Сканер данных завершен")
    return results


if __name__ == "__main__":
    """Запуск блока 13 для тестирования"""
    import pandas as pd
    import logging
    import sys
    import os
    import time
    
    # Добавляем путь для импорта base_block
    sys.path.append(os.path.dirname(__file__))
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    print("🧪 ТЕСТ БЛОКА 13: Scanner")
    print("=" * 50)
    
    try:
        # Проверяем доступность канонических данных
        canon_dir = Path("../../../data/canon")
        if not canon_dir.exists():
            print("❌ Папка канонических данных не найдена")
            print("   Сначала запустите блок 11 (Canonical)")
            sys.exit(1)
        
        print(f"✅ Найдена папка канонических данных: {canon_dir}")
        
        # Конфигурация для сканера
        config = {
            'data_types': ['quotes', 'book_top', 'features'],
            'processing': {
                'batch_size': 5,
                'max_workers': 2,
                'memory_limit_gb': 4
            },
            'filters': {
                'date_range': {},
                'exchanges': [],
                'symbols': []
            }
        }
        
        # Конфигурация для детекторов
        detectors_config = {
            'D1_LiquidityVacuumBreak': {
                'threshold': 0.1,
                'empty_levels': 3
            },
            'D2_AbsorptionFlip': {
                'min_side_ratio': 0.7,
                'max_price_move_ticks': 2
            },
            'D6_BookImbalance': {
                'bid_ask_ratio': 3.0,
                'level_wall_multiplier': 5.0
            }
        }
        
        # Запускаем блок 13
        print("🚀 Запускаем блок 13: Автоматический сканер данных...")
        start_time = time.time()
        
        results = run_scanner(config, detectors_config)
        
        execution_time = time.time() - start_time
        
        print(f"✅ Блок 13 выполнен за {execution_time:.2f} секунд!")
        print(f"📊 Результат сканирования:")
        
        # Анализируем результаты
        total_events = sum(len(df) for df in results.values())
        successful_detectors = sum(1 for df in results.values() if not df.empty)
        
        print(f"   - Успешных детекторов: {successful_detectors}/{len(results)}")
        print(f"   - Всего событий: {total_events}")
        
        for detector_name, events_df in results.items():
            if not events_df.empty:
                print(f"   - {detector_name}: {len(events_df)} событий")
            else:
                print(f"   - {detector_name}: событий не найдено")
        
        print("✅ Блок 13 успешно протестирован!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()