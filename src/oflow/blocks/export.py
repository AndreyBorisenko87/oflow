"""
Блок 15: Экспорт разметки
Выгрузка событий/сделок в CSV для TradingView/ATAS
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import logging

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
        
        # 1. Форматирование времени для ATAS
        if 'ts_ns' in atas_df.columns:
            atas_df['DateTime'] = pd.to_datetime(atas_df['ts_ns'], unit='ns')
            atas_df['Date'] = atas_df['DateTime'].dt.strftime('%m/%d/%Y')
            atas_df['Time'] = atas_df['DateTime'].dt.strftime('%H:%M:%S')
        
        # 2. Форматирование цены
        if 'price_level' in atas_df.columns:
            atas_df['Price'] = atas_df['price_level'].round(6)
        elif 'mid' in atas_df.columns:
            atas_df['Price'] = atas_df['mid'].round(6)
        
        # 3. Создание меток ATAS
        atas_df['Marker'] = atas_df.apply(_create_atas_marker, axis=1)
        
        # 4. Цветовая схема ATAS
        atas_df['Color'] = atas_df.apply(_create_atas_color, axis=1)
        
        # 5. Форматирование типа паттерна
        if 'pattern_type' in atas_df.columns:
            atas_df['Pattern'] = atas_df['pattern_type'].str.upper()
            atas_df['PatternCode'] = atas_df['pattern_type'].str[:3].str.upper()
        
        # 6. Форматирование уверенности
        if 'confidence' in atas_df.columns:
            atas_df['Confidence'] = atas_df['confidence'].round(3)
            atas_df['ConfidenceLevel'] = atas_df['confidence'].apply(
                lambda x: 'HIGH' if x >= 0.8 else 'MEDIUM' if x >= 0.6 else 'LOW'
            )
        
        # 7. Дополнительные поля для ATAS
        if 'volume_ratio' in atas_df.columns:
            atas_df['VolumeRatio'] = atas_df['volume_ratio'].round(2)
        
        if 'exchange' in atas_df.columns:
            atas_df['Exchange'] = atas_df['exchange'].str.upper()
        
        # 8. Выбор и переименование колонок для ATAS
        atas_columns = {
            'Date': 'Date',
            'Time': 'Time',
            'Price': 'Price',
            'Marker': 'Marker',
            'Color': 'Color',
            'Pattern': 'Pattern',
            'Confidence': 'Confidence',
            'VolumeRatio': 'VolumeRatio',
            'Exchange': 'Exchange'
        }
        
        # Фильтруем только существующие колонки
        existing_columns = {k: v for k, v in atas_columns.items() if k in atas_df.columns}
        atas_df = atas_df[list(existing_columns.keys())].rename(columns=existing_columns)
        
        # 9. Сортировка по времени
        if 'Date' in atas_df.columns and 'Time' in atas_df.columns:
            atas_df = atas_df.sort_values(['Date', 'Time']).reset_index(drop=True)
        
        logger.info(f"✅ Форматирование для ATAS завершено: {len(atas_df)} строк")
        logger.info(f"🔤 Колонки: {', '.join(atas_df.columns)}")
        
        return atas_df
        
    except Exception as e:
        logger.error(f"❌ Ошибка при форматировании для ATAS: {e}")
        raise

def _create_tv_marker(row: pd.Series) -> str:
    """Создание метки для TradingView"""
    try:
        pattern = row.get('pattern_type', '').upper()
        confidence = row.get('confidence', 0.0)
        
        # Базовые метки по типу паттерна
        if 'liquidity_vacuum_break' in pattern:
            base_marker = 'LVB'
        elif 'iceberg_fade' in pattern:
            base_marker = 'ICB'
        elif 'stop_run' in pattern:
            base_marker = 'STP'
        elif 'momentum' in pattern:
            base_marker = 'MOM'
        else:
            base_marker = 'PTN'
        
        # Добавляем уровень уверенности
        if confidence >= 0.8:
            strength = 'S'
        elif confidence >= 0.6:
            strength = 'M'
        else:
            strength = 'W'
        
        return f"{base_marker}_{strength}"
        
    except Exception:
        return "PTN_M"

def _create_atas_marker(row: pd.Series) -> str:
    """Создание метки для ATAS"""
    try:
        pattern = row.get('pattern_type', '').upper()
        confidence = row.get('confidence', 0.0)
        
        # Создаем уникальные метки для ATAS
        if 'liquidity_vacuum_break' in pattern:
            if confidence >= 0.8:
                return 'LVB_HIGH'
            elif confidence >= 0.6:
                return 'LVB_MED'
            else:
                return 'LVB_LOW'
        elif 'iceberg_fade' in pattern:
            if confidence >= 0.8:
                return 'ICB_HIGH'
            elif confidence >= 0.6:
                return 'ICB_MED'
            else:
                return 'ICB_LOW'
        elif 'stop_run' in pattern:
            if confidence >= 0.8:
                return 'STP_HIGH'
            elif confidence >= 0.6:
                return 'STP_MED'
            else:
                return 'STP_LOW'
        elif 'momentum' in pattern:
            if confidence >= 0.8:
                return 'MOM_HIGH'
            elif confidence >= 0.6:
                return 'MOM_MED'
            else:
                return 'MOM_LOW'
        else:
            return 'PATTERN'
            
    except Exception:
        return "PATTERN"

def _create_atas_color(row: pd.Series) -> str:
    """Создание цвета для ATAS"""
    try:
        pattern = row.get('pattern_type', '')
        confidence = row.get('confidence', 0.0)
        
        # Цвета по типу паттерна
        if 'liquidity_vacuum_break' in pattern:
            base_color = 'RED'
        elif 'iceberg_fade' in pattern:
            base_color = 'BLUE'
        elif 'stop_run' in pattern:
            base_color = 'GREEN'
        elif 'momentum' in pattern:
            base_color = 'YELLOW'
        else:
            base_color = 'WHITE'
        
        # Интенсивность по уверенности
        if confidence >= 0.8:
            intensity = 'BRIGHT'
        elif confidence >= 0.6:
            intensity = 'NORMAL'
        else:
            intensity = 'DIM'
        
        return f"{base_color}_{intensity}"
        
    except Exception:
        return "WHITE_NORMAL"

def run_export(
    events_df: pd.DataFrame,
    config: Dict,
    output_dir: Path = Path("data/export")
) -> None:
    """Основная функция экспорта разметки"""
    logger.info("🚀 Запуск экспорта разметки...")
    
    try:
        # Проверяем входные данные
        if events_df.empty:
            logger.warning("⚠️ Нет данных для экспорта")
            return
        
        logger.info(f"📊 Экспорт {len(events_df)} событий")
        
        # Создаем директорию для экспорта
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Форматирование и экспорт для TradingView
        logger.info("📈 Подготовка данных для TradingView...")
        tv_df = format_for_tradingview(events_df, config)
        if not tv_df.empty:
            tv_path = output_dir / "marks_tradingview.csv"
            export_to_csv(tv_df, tv_path, "tradingview", config)
            logger.info(f"✅ TradingView экспорт: {tv_path}")
        
        # 2. Форматирование и экспорт для ATAS
        logger.info("📊 Подготовка данных для ATAS...")
        atas_df = format_for_atas(events_df, config)
        if not atas_df.empty:
            atas_path = output_dir / "marks_atas.csv"
            export_to_csv(atas_df, atas_path, "atas", config)
            logger.info(f"✅ ATAS экспорт: {atas_path}")
        
        # 3. Общий CSV экспорт
        logger.info("📋 Подготовка общего CSV...")
        general_path = output_dir / "marks_general.csv"
        export_to_csv(events_df, general_path, "general", config)
        logger.info(f"✅ Общий CSV экспорт: {general_path}")
        
        # 4. Создание сводного отчета
        logger.info("📝 Создание сводного отчета...")
        _create_export_summary(events_df, output_dir, config)
        
        # 5. Итоговая статистика
        logger.info("=== ИТОГИ ЭКСПОРТА ===")
        logger.info(f"📊 Всего событий: {len(events_df)}")
        
        if 'pattern_type' in events_df.columns:
            pattern_counts = events_df['pattern_type'].value_counts()
            logger.info("📈 Распределение по паттернам:")
            for pattern, count in pattern_counts.items():
                logger.info(f"  {pattern}: {count}")
        
        if 'confidence' in events_df.columns:
            avg_confidence = events_df['confidence'].mean()
            logger.info(f"🎯 Средняя уверенность: {avg_confidence:.3f}")
        
        if 'exchange' in events_df.columns:
            exchange_counts = events_df['exchange'].value_counts()
            logger.info("🏢 Распределение по биржам:")
            for exchange, count in exchange_counts.items():
                logger.info(f"  {exchange}: {count}")
        
        logger.info(f"📁 Файлы сохранены в: {output_dir}")
        logger.info("✅ Экспорт разметки успешно завершен!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при экспорте разметки: {e}")
        import traceback
        traceback.print_exc()
        raise

def _create_export_summary(events_df: pd.DataFrame, output_dir: Path, config: Dict) -> None:
    """Создание сводного отчета по экспорту"""
    try:
        summary_path = output_dir / "export_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== СВОДНЫЙ ОТЧЕТ ПО ЭКСПОРТУ ===\n\n")
            f.write(f"Дата экспорта: {pd.Timestamp.now()}\n")
            f.write(f"Всего событий: {len(events_df)}\n\n")
            
            # Статистика по паттернам
            if 'pattern_type' in events_df.columns:
                f.write("=== РАСПРЕДЕЛЕНИЕ ПО ПАТТЕРНАМ ===\n")
                pattern_counts = events_df['pattern_type'].value_counts()
                for pattern, count in pattern_counts.items():
                    percentage = (count / len(events_df)) * 100
                    f.write(f"{pattern}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Статистика по биржам
            if 'exchange' in events_df.columns:
                f.write("=== РАСПРЕДЕЛЕНИЕ ПО БИРЖАМ ===\n")
                exchange_counts = events_df['exchange'].value_counts()
                for exchange, count in exchange_counts.items():
                    percentage = (count / len(events_df)) * 100
                    f.write(f"{exchange}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Статистика по уверенности
            if 'confidence' in events_df.columns:
                f.write("=== СТАТИСТИКА УВЕРЕННОСТИ ===\n")
                f.write(f"Средняя уверенность: {events_df['confidence'].mean():.3f}\n")
                f.write(f"Медианная уверенность: {events_df['confidence'].median():.3f}\n")
                f.write(f"Минимальная уверенность: {events_df['confidence'].min():.3f}\n")
                f.write(f"Максимальная уверенность: {events_df['confidence'].max():.3f}\n")
                
                # Распределение по уровням уверенности
                high_conf = len(events_df[events_df['confidence'] >= 0.8])
                med_conf = len(events_df[(events_df['confidence'] >= 0.6) & (events_df['confidence'] < 0.8)])
                low_conf = len(events_df[events_df['confidence'] < 0.6])
                
                f.write(f"Высокая уверенность (≥0.8): {high_conf} ({high_conf/len(events_df)*100:.1f}%)\n")
                f.write(f"Средняя уверенность (0.6-0.8): {med_conf} ({med_conf/len(events_df)*100:.1f}%)\n")
                f.write(f"Низкая уверенность (<0.6): {low_conf} ({low_conf/len(events_df)*100:.1f}%)\n")
                f.write("\n")
            
            # Статистика по объему
            if 'volume_ratio' in events_df.columns:
                f.write("=== СТАТИСТИКА ПО ОБЪЕМУ ===\n")
                f.write(f"Среднее соотношение объема: {events_df['volume_ratio'].mean():.2f}\n")
                f.write(f"Медианное соотношение объема: {events_df['volume_ratio'].median():.2f}\n")
                f.write(f"Максимальное соотношение объема: {events_df['volume_ratio'].max():.2f}\n")
                f.write("\n")
            
            # Временной диапазон
            if 'ts_ns' in events_df.columns:
                f.write("=== ВРЕМЕННОЙ ДИАПАЗОН ===\n")
                start_time = pd.to_datetime(events_df['ts_ns'].min(), unit='ns')
                end_time = pd.to_datetime(events_df['ts_ns'].max(), unit='ns')
                f.write(f"Начало: {start_time}\n")
                f.write(f"Конец: {end_time}\n")
                f.write(f"Продолжительность: {end_time - start_time}\n")
                f.write("\n")
            
            # Информация о файлах
            f.write("=== ЭКСПОРТИРОВАННЫЕ ФАЙЛЫ ===\n")
            f.write("marks_tradingview.csv - Данные для TradingView\n")
            f.write("marks_atas.csv - Данные для ATAS\n")
            f.write("marks_general.csv - Общий CSV экспорт\n")
            f.write("export_summary.txt - Этот отчет\n")
        
        logger.info(f"✅ Сводный отчет создан: {summary_path}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при создании сводного отчета: {e}")
        # Не прерываем основной процесс экспорта
