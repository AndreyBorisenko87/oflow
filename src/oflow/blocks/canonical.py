"""
Блок 11: Канонический слой
Сохранить quotes/book_top/tape/nbbo/basis/features пачками по дням
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import time
import json
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class CanonicalStorage:
    """Канонический слой для сохранения данных пачками по дням"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.batch_size_days = config.get('batch', {}).get('days', 1)
        self.compression = config.get('compression', {}).get('type', 'snappy')
        self.index_enabled = config.get('indexing', {}).get('enabled', True)
        self.metadata_enabled = config.get('metadata', {}).get('enabled', True)
        self.chunk_size = config.get('batch', {}).get('chunk_size', 10000)
        
        # Поддерживаемые типы данных
        self.supported_types = [
            'quotes', 'book_top', 'tape', 'nbbo', 
            'basis', 'features', 'events', 'trades', 'depth'
        ]
        
    def run_canonical_storage(
        self,
        data_dict: Dict[str, pd.DataFrame],
        output_dir: Path
    ) -> Dict[str, List[Path]]:
        """Полное сохранение данных в канонический слой"""
        start_time = time.time()
        logger.info("=== Начало сохранения в канонический слой ===")
        
        # Этап 1: Валидация данных
        logger.info("Этап 1/4: Валидация входных данных...")
        validated_data = self._validate_data(data_dict)
        
        # Этап 2: Разбивка по дням
        logger.info("Этап 2/4: Разбивка данных по дням...")
        batched_data = self._batch_by_days(validated_data)
        
        # Этап 3: Сохранение пачками
        logger.info("Этап 3/4: Сохранение данных пачками...")
        saved_files = self._save_batches(batched_data, output_dir)
        
        # Этап 4: Создание индексов и метаданных
        logger.info("Этап 4/4: Создание индексов и метаданных...")
        self._create_indexes_and_metadata(saved_files, output_dir)
        
        # Завершение
        duration = time.time() - start_time
        total_files = sum(len(files) for files in saved_files.values())
        logger.info(f"Сохранение завершено за {duration:.2f}с")
        logger.info(f"Сохранено {total_files} файлов по {len(saved_files)} типам данных")
        logger.info("=== Сохранение в канонический слой завершено ===")
        
        return saved_files
    
    def _validate_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Валидация входных данных"""
        logger.info("Валидация входных данных...")
        
        validated_data = {}
        
        for data_type, df in data_dict.items():
            if data_type not in self.supported_types:
                logger.warning(f"Неподдерживаемый тип данных: {data_type}")
                continue
            
            if df.empty:
                logger.warning(f"Пустой DataFrame для типа: {data_type}")
                continue
            
            # Проверка обязательных колонок
            required_columns = self._get_required_columns(data_type)
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Отсутствуют обязательные колонки для {data_type}: {missing_columns}")
                continue
            
            # Проверка временной колонки
            if 'ts_ns' in df.columns:
                df = df.sort_values('ts_ns').reset_index(drop=True)
                df['date'] = pd.to_datetime(df['ts_ns'], unit='ns').dt.date
                validated_data[data_type] = df
                logger.info(f"Валидирован {data_type}: {len(df)} записей")
            else:
                logger.warning(f"Отсутствует временная колонка ts_ns для {data_type}")
        
        return validated_data
    
    def _get_required_columns(self, data_type: str) -> List[str]:
        """Получение обязательных колонок для типа данных"""
        required_columns_map = {
            'quotes': ['ts_ns', 'exchange', 'symbol', 'bid', 'ask'],
            'book_top': ['ts_ns', 'exchange', 'symbol', 'side', 'price', 'size'],
            'tape': ['ts_ns', 'exchange', 'symbol', 'aggression_side'],
            'nbbo': ['ts_ns', 'best_bid', 'best_ask'],
            'basis': ['ts_ns', 'spot_symbol', 'futures_symbol', 'basis_abs'],
            'features': ['ts_ns', 'exchange', 'symbol'],
            'events': ['ts_ns', 'exchange', 'symbol', 'pattern_type'],
            'trades': ['ts_ns', 'exchange', 'symbol', 'price', 'size'],
            'depth': ['ts_ns', 'exchange', 'symbol', 'side', 'price', 'size']
        }
        return required_columns_map.get(data_type, ['ts_ns'])
    
    def _batch_by_days(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, List[pd.DataFrame]]]:
        """Разбивка данных по дням"""
        logger.info(f"Разбивка данных по дням (размер пакета: {self.batch_size_days} дней)...")
        
        batched_data = {}
        
        for data_type, df in data_dict.items():
            if df.empty:
                continue
            
            # Группировка по датам
            date_groups = df.groupby('date')
            logger.info(f"Обработка {data_type}: {len(date_groups)} дней данных")
            
            # Создание пакетов
            batches = {}
            dates = sorted(df['date'].unique())
            
            # Прогресс-логирование
            total_dates = len(dates)
            for i, date in enumerate(dates):
                if i % 10 == 0:  # Логируем каждые 10 дней
                    logger.info(f"Прогресс {data_type}: {i+1}/{total_dates} дней")
                
                # Определение пакета
                batch_key = self._get_batch_key(date, self.batch_size_days)
                
                if batch_key not in batches:
                    batches[batch_key] = []
                
                # Добавление данных дня в пакет
                day_data = date_groups.get_group(date).copy()
                day_data = day_data.drop(columns=['date'])  # Убираем служебную колонку
                batches[batch_key].append(day_data)
            
            # Объединение пакетов
            final_batches = {}
            for batch_key, day_dfs in batches.items():
                if day_dfs:
                    combined_df = pd.concat(day_dfs, ignore_index=True)
                    combined_df = combined_df.sort_values('ts_ns').reset_index(drop=True)
                    final_batches[batch_key] = [combined_df]  # Один объединенный DataFrame на пакет
            
            batched_data[data_type] = final_batches
            logger.info(f"Создано {len(final_batches)} пакетов для {data_type}")
        
        return batched_data
    
    def _get_batch_key(self, date, batch_size_days: int) -> str:
        """Получение ключа пакета для даты"""
        # Конвертируем в datetime для арифметики
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        
        # Определяем начало эпохи (например, 2020-01-01)
        epoch_start = datetime(2020, 1, 1).date()
        
        # Вычисляем количество дней от начала эпохи
        days_since_epoch = (date - epoch_start).days
        
        # Определяем номер пакета
        batch_number = days_since_epoch // batch_size_days
        
        # Вычисляем границы пакета
        batch_start_days = batch_number * batch_size_days
        batch_start_date = epoch_start + timedelta(days=batch_start_days)
        batch_end_date = batch_start_date + timedelta(days=batch_size_days - 1)
        
        return f"{batch_start_date.strftime('%Y%m%d')}_{batch_end_date.strftime('%Y%m%d')}"
    
    def _save_batches(
        self,
        batched_data: Dict[str, Dict[str, List[pd.DataFrame]]],
        output_dir: Path
    ) -> Dict[str, List[Path]]:
        """Сохранение пакетов данных"""
        logger.info("Сохранение пакетов данных...")
        
        saved_files = {}
        
        for data_type, batches in batched_data.items():
            saved_files[data_type] = []
            
            logger.info(f"Сохранение {data_type}: {len(batches)} пакетов")
            
            for i, (batch_key, batch_dfs) in enumerate(batches.items()):
                if i % 5 == 0:  # Логируем каждые 5 пакетов
                    logger.info(f"Прогресс сохранения {data_type}: {i+1}/{len(batches)} пакетов")
                
                for j, df in enumerate(batch_dfs):
                    if df.empty:
                        continue
                    
                    # Определение пути файла
                    filename = f"{data_type}_{batch_key}"
                    if len(batch_dfs) > 1:
                        filename += f"_part{j+1:03d}"
                    filename += ".parquet"
                    
                    file_path = output_dir / data_type / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Сохранение с настройками сжатия
                    df.to_parquet(
                        file_path,
                        engine="pyarrow",
                        compression=self.compression,
                        index=False
                    )
                    
                    saved_files[data_type].append(file_path)
                    logger.debug(f"Сохранен {file_path}: {len(df)} записей")
            
            logger.info(f"Завершено сохранение {data_type}: {len(saved_files[data_type])} файлов")
        
        return saved_files
    
    def _create_indexes_and_metadata(
        self,
        saved_files: Dict[str, List[Path]],
        output_dir: Path
    ) -> None:
        """Создание индексов и метаданных"""
        logger.info("Создание индексов и метаданных...")
        
        if not self.index_enabled and not self.metadata_enabled:
            logger.info("Индексация и метаданные отключены")
            return
        
        # Создание общего индекса
        if self.index_enabled:
            index_data = self._create_file_index(saved_files)
            index_path = output_dir / "file_index.json"
            
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Индекс файлов сохранен: {index_path}")
        
        # Создание метаданных
        if self.metadata_enabled:
            metadata = self._create_metadata(saved_files)
            metadata_path = output_dir / "metadata.json"
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Метаданные сохранены: {metadata_path}")
    
    def _create_file_index(self, saved_files: Dict[str, List[Path]]) -> Dict:
        """Создание индекса файлов"""
        index_data = {
            "created_at": datetime.now().isoformat(),
            "total_files": sum(len(files) for files in saved_files.values()),
            "data_types": list(saved_files.keys()),
            "files": {}
        }
        
        for data_type, files in saved_files.items():
            index_data["files"][data_type] = []
            
            for file_path in files:
                # Извлечение информации из имени файла
                filename = file_path.name
                file_info = {
                    "path": str(file_path.relative_to(file_path.parent.parent)),
                    "filename": filename,
                    "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
                    "created_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
                }
                
                # Извлечение временных границ из имени файла
                if "_" in filename:
                    try:
                        parts = filename.replace(".parquet", "").split("_")
                        if len(parts) >= 3:
                            start_date = parts[1]
                            end_date = parts[2]
                            file_info["date_range"] = {
                                "start": start_date,
                                "end": end_date
                            }
                    except Exception as e:
                        logger.debug(f"Не удалось извлечь временные границы из {filename}: {e}")
                
                index_data["files"][data_type].append(file_info)
        
        return index_data
    
    def _create_metadata(self, saved_files: Dict[str, List[Path]]) -> Dict:
        """Создание метаданных"""
        metadata = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "config": self.config,
            "statistics": {},
            "schema": {}
        }
        
        # Статистика по типам данных
        for data_type, files in saved_files.items():
            total_size = sum(
                file_path.stat().st_size 
                for file_path in files 
                if file_path.exists()
            )
            
            metadata["statistics"][data_type] = {
                "file_count": len(files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 2)
            }
        
        # Общая статистика
        total_files = sum(len(files) for files in saved_files.values())
        total_size = sum(
            stat["total_size_bytes"] 
            for stat in metadata["statistics"].values()
        )
        
        metadata["summary"] = {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "compression": self.compression,
            "batch_size_days": self.batch_size_days
        }
        
        return metadata

def batch_by_days(
    data_dict: Dict[str, pd.DataFrame],
    batch_size_days: int = 1
) -> Dict[str, List[pd.DataFrame]]:
    """Упрощенная функция разбивки данных по дням (legacy)"""
    logger.info(f"Разбивка данных по дням (размер пакета: {batch_size_days})...")
    
    config = {
        'batch': {'days': batch_size_days},
        'compression': {'type': 'snappy'},
        'indexing': {'enabled': False},
        'metadata': {'enabled': False}
    }
    
    storage = CanonicalStorage(config)
    validated_data = storage._validate_data(data_dict)
    batched_data = storage._batch_by_days(validated_data)
    
    # Преобразование в legacy формат
    legacy_batches = {}
    for data_type, batches in batched_data.items():
        legacy_batches[data_type] = []
        for batch_key, batch_dfs in batches.items():
            legacy_batches[data_type].extend(batch_dfs)
    
    return legacy_batches

def save_canonical(
    data_dict: Dict[str, pd.DataFrame],
    output_dir: Path,
    config: Dict
) -> Dict[str, List[Path]]:
    """Основная функция сохранения канонических данных"""
    logger.info("Сохранение канонических данных...")
    
    # Создание канонического хранилища
    storage = CanonicalStorage(config)
    
    # Полное сохранение с индексацией и метаданными
    return storage.run_canonical_storage(data_dict, output_dir)

def run_canonical(
    data_dict: Dict[str, pd.DataFrame],
    config: Dict,
    output_dir: Path = Path("data/canon")
) -> Dict[str, List[Path]]:
    """Основная функция канонического слоя"""
    logger.info("Запуск канонического слоя...")
    
    # Сохранение канонических данных
    saved_files = save_canonical(data_dict, output_dir, config)
    
    logger.info("Канонический слой завершен")
    return saved_files
