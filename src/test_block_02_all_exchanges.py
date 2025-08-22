#!/usr/bin/env python3
"""
Тест блока 02: ImportRaw со всеми биржами
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'oflow'))

from oflow.blocks import ImportRawBlock
import pandas as pd
import time

def test_block_02_all_exchanges():
    """Тестируем блок 02 со всеми 12 биржами"""
    print("🧪 ТЕСТ БЛОКА 02: ImportRaw со всеми биржами")
    print("=" * 60)
    
    # Все 12 путей к данным (6 бирж x 2 типа данных)
    data_paths = [
        # Futures
        'C:/Users/andrei.barysenka/Documents/Trading/CryptoFeed/data/futures/binance_futures/2025-07-23-19-trades.parquet',
        'C:/Users/andrei.barysenka/Documents/Trading/CryptoFeed/data/futures/binance_futures/2025-07-23-19-depth.parquet',
        'C:/Users/andrei.barysenka/Documents/Trading/CryptoFeed/data/futures/bybit/2025-07-23-19-trades.parquet',
        'C:/Users/andrei.barysenka/Documents/Trading/CryptoFeed/data/futures/bybit/2025-07-23-19-depth.parquet',
        'C:/Users/andrei.barysenka/Documents/Trading/CryptoFeed/data/futures/okx/2025-07-23-19-trades.parquet',
        'C:/Users/andrei.barysenka/Documents/Trading/CryptoFeed/data/futures/okx/2025-07-23-19-depth.parquet',
        # Spot
        'C:/Users/andrei.barysenka/Documents/Trading/CryptoFeed/data/spot/binance/2025-07-23-19-trades.parquet',
        'C:/Users/andrei.barysenka/Documents/Trading/CryptoFeed/data/spot/binance/2025-07-23-19-depth.parquet',
        'C:/Users/andrei.barysenka/Documents/Trading/CryptoFeed/data/spot/bybit/2025-07-23-19-trades.parquet',
        'C:/Users/andrei.barysenka/Documents/Trading/CryptoFeed/data/spot/bybit/2025-07-23-19-depth.parquet',
        'C:/Users/andrei.barysenka/Documents/Trading/CryptoFeed/data/spot/okx/2025-07-23-19-trades.parquet',
        'C:/Users/andrei.barysenka/Documents/Trading/CryptoFeed/data/spot/okx/2025-07-23-19-depth.parquet'
    ]
    
    print(f"📁 Загружаем данные с {len(data_paths)} источников:")
    for i, path in enumerate(data_paths, 1):
        print(f"   {i:2d}. {os.path.basename(path)}")
    
    # Создаем блок 02 с правильным путем для сохранения
    config = {
        'test_mode': True,
        'output_dir': '../data/normalized'  # Путь относительно src/
    }
    block = ImportRawBlock(config)
    
    # Тестовые данные
    test_data = {'data_paths': data_paths}
    
    # Запускаем блок 02
    print("\n🚀 Запускаем блок 02: ImportRaw...")
    start_time = time.time()
    
    try:
        result = block.run(test_data, config)
        execution_time = time.time() - start_time
        
        print(f"✅ Блок 02 выполнен за {execution_time:.2f} секунд!")
        
        # Анализируем результат (блок 02 возвращает tuple)
        if isinstance(result, tuple) and len(result) >= 2:
            trades_df = result[0]
            depth_df = result[1]
        else:
            trades_df = result.get('trades_df') if isinstance(result, dict) else None
            depth_df = result.get('depth_df') if isinstance(result, dict) else None
        
        print(f"\n📊 РЕЗУЛЬТАТ ОБРАБОТКИ ВСЕХ БИРЖ:")
        print(f"   - Trades: {len(trades_df) if trades_df is not None else 0} строк")
        print(f"   - Depth: {len(depth_df) if depth_df is not None else 0} строк")
        
        # Проверяем какие биржи обработались
        if trades_df is not None and not trades_df.empty:
            exchanges = trades_df['exchange'].unique()
            print(f"   - Биржи в trades: {exchanges}")
            print(f"   - Количество бирж: {len(exchanges)}")
            
            # Статистика по биржам
            print(f"\n📈 СТАТИСТИКА ПО БИРЖАМ:")
            for exchange in exchanges:
                ex_data = trades_df[trades_df['exchange'] == exchange]
                print(f"   - {exchange}: {len(ex_data)} trades")
        
        if depth_df is not None and not depth_df.empty:
            exchanges_depth = depth_df['exchange'].unique()
            print(f"   - Биржи в depth: {exchanges_depth}")
            print(f"   - Количество бирж: {len(exchanges_depth)}")
            
            # Статистика по биржам
            print(f"\n📊 СТАТИСТИКА DEPTH ПО БИРЖАМ:")
            for exchange in exchanges_depth:
                ex_data = depth_df[depth_df['exchange'] == exchange]
                print(f"   - {exchange}: {len(ex_data)} depth records")
        
        print("\n✅ Блок 02 успешно протестирован со всеми биржами!")
        return True
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Ошибка в блоке 02 после {execution_time:.2f} сек: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_block_02_all_exchanges()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ТЕСТ БЛОКА 02 СО ВСЕМИ БИРЖАМИ ПРОЙДЕН!")
    else:
        print("❌ ТЕСТ БЛОКА 02 СО ВСЕМИ БИРЖАМИ ПРОВАЛЕН!")
    print("=" * 60)
