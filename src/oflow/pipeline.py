"""
Главный файл для запуска всей цепочки блоков
Scalping Strategy Pipeline
"""

import logging
from pathlib import Path
from typing import Dict, List
import yaml

from .blocks import (
    # Блок 02: Импорт сырья
    import_raw_data,
    
    # Блок 04: Синхронизация времени
    run_time_sync,
    
    # Блок 05: Лучшие цены
    run_best_prices,
    
    # Блок 06: Топ-зона книги
    run_book_top,
    
    # Блок 07: Лента сделок
    run_trade_tape,
    
    # Блок 08: Кросс-биржевая агрегация
    run_cross_exchange,
    
    # Блок 09: Базис спот-фьючерс
    run_basis,
    
    # Блок 10: Фичи паттернов
    run_features,
    
    # Блок 11: Канонический слой
    run_canonical,
    
    # Блок 12: Детекторы
    run_detectors,
    
    # Блок 13: Сканер
    run_scanner,
    
    # Блок 14: Бэктест
    run_backtest,
    
    # Блок 15: Экспорт разметки
    export_data
)

def setup_logging(config: Dict) -> None:
    """Настройка логирования"""
    log_config = config.get("logging", {})
    
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_config.get("file", "logs/oflow.log"), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_config(config_dir: str = "configs") -> Dict:
    """Загрузка конфигурации"""
    config_path = Path(config_dir) / "config.yaml"
    sources_path = Path(config_dir) / "sources.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    with open(sources_path, "r", encoding="utf-8") as f:
        sources = yaml.safe_load(f)
    
    config["sources"] = sources
    return config

def run_full_pipeline(config_dir: str = "configs", test_mode: bool = True) -> Dict[str, any]:
    """
    Запуск полной цепочки обработки
    
    Args:
        config_dir: Директория с конфигурацией
        test_mode: Режим тестирования (обработка только тестовых файлов)
        
    Returns:
        Словарь с результатами каждого блока
    """
    logger = logging.getLogger(__name__)
    logger.info("Запуск полной цепочки обработки...")
    
    # 1. Загрузка конфигурации
    config = load_config(config_dir)
    setup_logging(config)
    
    results = {}
    
    try:
        # 2. Блок 02: Импорт сырья
        logger.info("=== БЛОК 02: ИМПОРТ СЫРЬЯ ===")
        
        # Подготавливаем данные для импорта
        import_data = {
            "data_paths": [],
            "start_time": None,
            "end_time": None
        }
        
        # Добавляем пути к данным из sources.yaml
        if "sources" in config:
            sources = config["sources"]
            for market_type in ["spot", "futures"]:
                if market_type in sources:
                    for exchange in sources[market_type]:
                        for data_type in ["trades", "depth"]:
                            if data_type in sources[market_type][exchange]:
                                import_data["data_paths"].append(sources[market_type][exchange][data_type])
        
        logger.info(f"Найдено {len(import_data['data_paths'])} путей к данным")
        
        if test_mode:
            trades_df, depth_df = import_raw_data(import_data, config)
        else:
            trades_df, depth_df = import_raw_data(import_data, config)
        
        results["import"] = {"trades": trades_df, "depth": depth_df}
        logger.info(f"Импорт завершен: {len(trades_df)} trades, {len(depth_df)} depth")
        
        # 3. Блок 04: Синхронизация времени
        logger.info("=== БЛОК 04: СИНХРОНИЗАЦИЯ ВРЕМЕНИ ===")
        synced_trades, synced_depth = run_time_sync(trades_df, depth_df, config)
        results["sync"] = {"trades": synced_trades, "depth": synced_depth}
        logger.info("Синхронизация времени завершена")
        
        # 4. Блок 05: Лучшие цены
        logger.info("=== БЛОК 05: ЛУЧШИЕ ЦЕНЫ ===")
        quotes_df = run_best_prices(synced_depth, config)
        results["quotes"] = quotes_df
        logger.info("Лучшие цены восстановлены")
        
        # 5. Блок 06: Топ-зона книги
        logger.info("=== БЛОК 06: ТОП-ЗОНА КНИГИ ===")
        book_df = run_book_top(synced_depth, config)
        results["book_top"] = book_df
        logger.info("Топ-зона книги проанализирована")
        
        # 6. Блок 07: Лента сделок
        logger.info("=== БЛОК 07: ЛЕНТА СДЕЛОК ===")
        tape_df = run_trade_tape(synced_trades, config)
        results["tape"] = tape_df
        logger.info("Лента сделок проанализирована")
        
        # 7. Блок 08: Кросс-биржевая агрегация
        logger.info("=== БЛОК 08: КРОСС-БИРЖЕВАЯ АГРЕГАЦИЯ ===")
        nbbo_df, volume_df = run_cross_exchange(quotes_df, synced_trades, config)
        results["nbbo"] = nbbo_df
        results["volume"] = volume_df
        logger.info("Кросс-биржевая агрегация завершена")
        
        # 8. Блок 09: Базис спот-фьючерс
        logger.info("=== БЛОК 09: БАЗИС СПОТ-ФЬЮЧЕРС ===")
        # TODO: Разделить данные на spot и futures
        basis_df = run_basis(synced_trades, synced_trades, config)  # Заглушка
        results["basis"] = basis_df
        logger.info("Базис спот-фьючерс вычислен")
        
        # 9. Блок 10: Фичи паттернов
        logger.info("=== БЛОК 10: ФИЧИ ПАТТЕРНОВ ===")
        features_df = run_features(book_df, tape_df, quotes_df, config)
        results["features"] = features_df
        logger.info("Признаки паттернов извлечены")
        
        # 10. Блок 11: Канонический слой
        logger.info("=== БЛОК 11: КАНОНИЧЕСКИЙ СЛОЙ ===")
        canonical_data = {
            "quotes": quotes_df,
            "book_top": book_df,
            "tape": tape_df,
            "nbbo": nbbo_df,
            "basis": basis_df,
            "features": features_df
        }
        run_canonical(canonical_data, config)
        results["canonical"] = canonical_data
        logger.info("Канонический слой сохранен")
        
        # 11. Блок 12: Детекторы
        logger.info("=== БЛОК 12: ДЕТЕКТОРЫ ===")
        lvb_detector = D1LiquidityVacuumBreak(config)
        detectors = [lvb_detector]
        detector_results = run_detectors(detectors, canonical_data)
        results["detectors"] = detector_results
        logger.info("Детекторы запущены")
        
        # 12. Блок 13: Сканер
        logger.info("=== БЛОК 13: СКАНЕР ===")
        scanner_results = run_scanner(config, detectors)
        results["scanner"] = scanner_results
        logger.info("Сканер завершен")
        
        # 13. Блок 14: Бэктест
        logger.info("=== БЛОК 14: БЭКТЕСТ ===")
        # TODO: Получить события из детекторов
        events_df = pd.DataFrame()  # Заглушка
        trades_df, metrics = run_backtest(events_df, quotes_df, config)
        results["backtest"] = {"trades": trades_df, "metrics": metrics}
        logger.info("Бэктест завершен")
        
        # 14. Блок 15: Экспорт разметки
        logger.info("=== БЛОК 15: ЭКСПОРТ РАЗМЕТКИ ===")
        export_data(events_df, config)
        results["export"] = {"status": "completed"}
        logger.info("Экспорт разметки завершен")
        
        logger.info("🎉 ВСЯ ЦЕПОЧКА ОБРАБОТКИ ЗАВЕРШЕНА УСПЕШНО!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в цепочке обработки: {e}")
        raise
    
    return results

def run_test_pipeline(config_dir: str = "configs") -> Dict[str, any]:
    """Запуск тестовой цепочки (только первые блоки)"""
    logger = logging.getLogger(__name__)
    logger.info("Запуск тестовой цепочки...")
    
    # Загружаем конфигурацию
    config = load_config(config_dir)
    setup_logging(config)
    
    results = {}
    
    try:
        # Блок 02: Импорт сырья
        logger.info("=== БЛОК 02: ИМПОРТ СЫРЬЯ ===")
        
        # Подготавливаем данные для импорта
        import_data = {
            "data_paths": [],
            "start_time": None,
            "end_time": None
        }
        
        # Добавляем пути к данным из sources.yaml
        if "sources" in config:
            sources = config["sources"]
            for market_type in ["spot", "futures"]:
                if market_type in sources:
                    for exchange in sources[market_type]:
                        for data_type in ["trades", "depth"]:
                            if data_type in sources[market_type][exchange]:
                                import_data["data_paths"].append(sources[market_type][exchange][data_type])
        
        logger.info(f"Найдено {len(import_data['data_paths'])} путей к данным")
        
        trades_df, depth_df = import_raw_data(import_data, config)
        results["import"] = {"trades": trades_df, "depth": depth_df}
        
        # Блок 04: Синхронизация времени
        logger.info("=== БЛОК 04: СИНХРОНИЗАЦИЯ ВРЕМЕНИ ===")
        synced_trades, synced_depth = run_time_sync(trades_df, depth_df, config)
        results["sync"] = {"trades": synced_trades, "depth": synced_depth}
        
        # Блок 05: Лучшие цены
        logger.info("=== БЛОК 05: ЛУЧШИЕ ЦЕНЫ ===")
        quotes_df = run_best_prices(synced_depth, config)
        results["quotes"] = quotes_df
        
        logger.info("✅ Тестовая цепочка завершена успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в тестовой цепочке: {e}")
        raise
    
    return results

if __name__ == "__main__":
    # Запуск тестовой цепочки по умолчанию
    results = run_test_pipeline()
    print(f"Результаты: {len(results)} блоков выполнено")
