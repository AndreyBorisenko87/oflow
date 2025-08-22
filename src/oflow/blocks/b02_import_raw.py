from __future__ import annotations
import os, math, glob, json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import yaml
from .base_block import BaseBlock

# ---------- утилиты ----------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_ns(ts: pd.Series) -> pd.Series:
    # timestamp в секундах с дробью -> int64 наносекунды
    # или datetime -> наносекунды
    try:
        if pd.api.types.is_datetime64_any_dtype(ts):
            return ts.astype("int64")  # уже в наносекундах
        else:
            return (ts.astype("float64") * 1_000_000_000).round().astype("int64")
    except Exception as e:
        print(f"Ошибка в to_ns: {e}")
        # Возвращаем пустую Series в случае ошибки
        return pd.Series(dtype="int64")

def norm_instrument(sym: str) -> Tuple[str, str]:
    s = sym.replace("_", "-")
    if s.endswith("-SWAP") or s.endswith("-PERP"):
        base = s.replace("-SWAP", "").replace("-PERP", "")
        return f"{base}", "PERP"
    if "-" not in s:
        # ETHUSDT -> ETH-USDT
        s = s[:-4] + "-" + s[-4:]
    return s, "SPOT"

def to_float(x):
    if x is None: return np.nan
    if isinstance(x,(int,float,np.floating)): return float(x)
    try: return float(str(x))
    except: return np.nan

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

# ---------- нормализация TRADES ----------
REQ_TRADE_COLS = {"exchange","market","symbol","price","amount","side","id","timestamp"}
def read_trades(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        try:
            df = pd.read_parquet(p, engine="pyarrow")
            missing = REQ_TRADE_COLS - set(df.columns)
            if missing:
                print(f"[WARN] trades missing {missing} in {p}")
            df = df.assign(_src_file=p)
            dfs.append(df[list(REQ_TRADE_COLS & set(df.columns) | {"_src_file"})])
        except Exception as e:
            print(f"[ERR] read trades {p}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=list(REQ_TRADE_COLS)+["_src_file"])

def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    instr, kind = zip(*df["symbol"].map(norm_instrument))
    base_symbol = [s.replace("-","") for s in instr]
    
    # Создаем уникальный идентификатор биржи с учетом market
    # binance_futures -> binance_futures, binance -> binance_spot
    unique_exchange = []
    for ex, mk in zip(df["exchange"], df["market"]):
        if ex == "binance" and mk == "futures":
            unique_exchange.append("binance_futures")
        elif ex == "binance" and mk == "spot":
            unique_exchange.append("binance_spot")
        elif ex == "bybit" and mk == "futures":
            unique_exchange.append("bybit_futures")
        elif ex == "bybit" and mk == "spot":
            unique_exchange.append("bybit_spot")
        elif ex == "okx" and mk == "futures":
            unique_exchange.append("okx_futures")
        elif ex == "okx" and mk == "spot":
            unique_exchange.append("okx_spot")
        else:
            unique_exchange.append(f"{ex}_{mk}")
    
    out = pd.DataFrame({
        "ts_ns": to_ns(df["timestamp"]),
        "exchange": pd.Series(unique_exchange, dtype="string"),
        "market": df["market"].astype("string"),
        "instrument": pd.Series(instr, dtype="string"),
        "base_symbol": pd.Series(base_symbol, dtype="string"),
        "aggressor": df["side"].astype("string"),
        "price": df["price"].astype("float64"),
        "size": df["amount"].astype("float64"),
        "trade_id": df["id"].astype("string"),
        "_src_file": df["_src_file"].astype("string")
    })
    out.sort_values("ts_ns", inplace=True, kind="mergesort")
    return out

# ---------- нормализация DEPTH (delta из снапшотов) ----------
# ожидаем структуру 'delta' с ключами bid/ask или b/a/bids/asks
REQ_DEPTH_COLS = {"exchange","market","symbol","timestamp","delta"}
def _rows_from_side(side_name: str, arr) -> List[Tuple[str,float,float,str]]:
    rows=[]
    if arr is None: 
        return rows
    
    # Обрабатываем numpy array
    if hasattr(arr, '__len__') and len(arr) == 0:
        return rows
    
    for i, rec in enumerate(arr):
        # rec может быть ["4343.91","34.33"] или [4343.91,34.33] или numpy array
        if not isinstance(rec,(list,tuple,np.ndarray)) or len(rec)<2: 
            continue
        price = to_float(rec[0]); size = to_float(rec[1])
        if math.isnan(price): 
            continue
        action = "del" if (size==0 or np.isclose(size,0.0)) else "update"
        rows.append((side_name, price, size, action))
    
    return rows

def flatten_depth_delta(row) -> List[Tuple[str,float,float,str]]:
    d = row.get("delta")
    if d is None:
        return []
    
    # разные биржи: 'bid'/'ask', 'b'/'a', 'bids'/'asks'
    arr_bid = d.get("bid") if "bid" in d else (d.get("b") if "b" in d else (d.get("bids") if "bids" in d else None))
    arr_ask = d.get("ask") if "ask" in d else (d.get("a") if "a" in d else (d.get("asks") if "asks" in d else None))
    
    rows = []
    rows.extend(_rows_from_side("bid", arr_bid))
    rows.extend(_rows_from_side("ask", arr_ask))
    return rows

def read_depth(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        try:
            df = pd.read_parquet(p, engine="pyarrow")
            missing = REQ_DEPTH_COLS - set(df.columns)
            if missing:
                print(f"[WARN] depth missing {missing} in {p}")
            df = df.assign(_src_file=p)
            dfs.append(df[list(REQ_DEPTH_COLS & set(df.columns) | {"_src_file"})])
        except Exception as e:
            print(f"[ERR] read depth {p}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=list(REQ_DEPTH_COLS)+["_src_file"])

def normalize_depth(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    
    print(f"🔍 Нормализация depth: {len(df)} записей...")
    
    # Разворачиваем delta в строки
    rows = []
    total_rows = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 10000 == 0:  # Прогресс каждые 10k записей
            print(f"📊 Обработано {idx}/{total_rows} ({idx/total_rows*100:.1f}%)")
        
        delta_rows = flatten_depth_delta(row)
        for side, price, size, action in delta_rows:
            # Создаем уникальный идентификатор биржи с учетом market
            ex = row["exchange"]
            mk = row["market"]
            if ex == "binance" and mk == "futures":
                unique_exchange = "binance_futures"
            elif ex == "binance" and mk == "spot":
                unique_exchange = "binance_spot"
            elif ex == "bybit" and mk == "futures":
                unique_exchange = "bybit_futures"
            elif ex == "bybit" and mk == "spot":
                unique_exchange = "bybit_spot"
            elif ex == "okx" and mk == "futures":
                unique_exchange = "okx_futures"
            elif ex == "okx" and mk == "spot":
                unique_exchange = "okx_spot"
            else:
                unique_exchange = f"{ex}_{mk}"
            
            rows.append({
                "ts_ns": int(row["timestamp"].timestamp() * 1_000_000_000),
                "exchange": unique_exchange,
                "market": row["market"],
                "symbol": row["symbol"],
                "side": side,
                "price": price,
                "size": size,
                "action": action,
                "_src_file": row["_src_file"]
            })
    
    print(f"✅ Нормализация depth завершена: {len(rows)} строк")
    
    if not rows:
        return pd.DataFrame()
    
    out = pd.DataFrame(rows)
    out.sort_values("ts_ns", inplace=True, kind="mergesort")
    return out

# ---------- нормализация QUOTES ----------
REQ_QUOTE_COLS = {"exchange","market","symbol","timestamp","bid","ask"}
def read_quotes(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        try:
            df = pd.read_parquet(p, engine="pyarrow")
            missing = REQ_QUOTE_COLS - set(df.columns)
            if missing:
                print(f"[WARN] quotes missing {missing} in {p}")
            df = df.assign(_src_file=p)
            dfs.append(df[list(REQ_QUOTE_COLS & set(df.columns) | {"_src_file"})])
        except Exception as e:
            print(f"[ERR] read quotes {p}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=list(REQ_QUOTE_COLS)+["_src_file"])

def normalize_quotes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    
    out = pd.DataFrame({
        "ts_ns": to_ns(df["timestamp"]),
        "exchange": df["exchange"].astype("string"),
        "market": df["market"].astype("string"),
        "symbol": df["symbol"].astype("string"),
        "best_bid": df["bid"].astype("float64"),
        "best_ask": df["ask"].astype("float64"),
        "_src_file": df["_src_file"].astype("string")
    })
    
    # Вычисляем mid и spread
    out["mid"] = (out["best_bid"] + out["best_ask"]) / 2
    out["spread"] = out["best_ask"] - out["best_bid"]
    
    out.sort_values("ts_ns", inplace=True, kind="mergesort")
    return out

# ---------- основной класс блока ----------
class ImportRawBlock(BaseBlock):
    """Блок импорта и нормализации сырых данных"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.input_dir = config.get('input_dir', 'data/raw')
        self.output_dir = config.get('output_dir', 'data/normalized')
        self.symbols = config.get('symbols', ['ETHUSDT'])
        self.exchanges = config.get('exchanges', ['binance', 'bybit', 'okx'])
        
    def validate_data(self, data: Dict[str, Any], config: Dict) -> bool:
        """Расширенная валидация для блока импорта"""
        if not super().validate_data(data, config):
            return False
        
        # Проверяем наличие путей к данным
        data_paths = data.get('data_paths', [])
        if not data_paths:
            self.logger.error("Не указаны пути к данным")
            return False
        
        # Проверяем наличие файлов по указанным путям
        from glob import glob
        all_files = []
        for pattern in data_paths:
            files = glob(pattern)
            all_files.extend(files)
        
        if not all_files:
            self.logger.error(f"Не найдено файлов по указанным путям: {data_paths}")
            return False
        
        self.logger.info(f"Найдено {len(all_files)} файлов для обработки")
        return True
    
    def run(self, data: Dict[str, Any], config: Dict) -> Dict[str, pd.DataFrame]:
        """Основная логика импорта и нормализации"""
        data_paths = data.get('data_paths', [])
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        
        self.logger.info(f"Обработка данных по {len(data_paths)} путям")
        
        # Создаем выходную директорию
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Обрабатываем trades
        self.logger.info("Обработка trades...")
        trades_files = [p for p in data_paths if 'trades' in p]
        if trades_files:
            # Разрешаем wildcard для каждого пути
            from glob import glob
            resolved_trades_files = []
            for pattern in trades_files:
                resolved_files = glob(pattern)
                resolved_trades_files.extend(resolved_files)
            
            if resolved_trades_files:
                trades_df = read_trades(resolved_trades_files)
            if not trades_df.empty:
                normalized_trades = normalize_trades(trades_df)
                results['trades'] = normalized_trades
                self.logger.info(f"Обработано {len(normalized_trades)} trades")
            else:
                self.logger.warning("Trades файлы пустые")
        else:
            self.logger.warning("Trades файлы не найдены")
        
        # Обрабатываем depth
        self.logger.info("Обработка depth...")
        depth_files = [p for p in data_paths if 'depth' in p]
        if depth_files:
            # Разрешаем wildcard для каждого пути
            resolved_depth_files = []
            for pattern in depth_files:
                resolved_files = glob(pattern)
                resolved_depth_files.extend(resolved_files)
            
            if resolved_depth_files:
                depth_df = read_depth(resolved_depth_files)
            if not depth_df.empty:
                normalized_depth = normalize_depth(depth_df)
                results['book_top'] = normalized_depth
                self.logger.info(f"Обработано {len(normalized_depth)} depth записей")
            else:
                self.logger.warning("Depth файлы пустые")
        else:
            self.logger.warning("Depth файлы не найдены")
        
        # Обрабатываем quotes
        self.logger.info("Обработка quotes...")
        quotes_files = [p for p in data_paths if 'quotes' in p]
        if quotes_files:
            # Разрешаем wildcard для каждого пути
            resolved_quotes_files = []
            for pattern in quotes_files:
                resolved_files = glob(pattern)
                resolved_quotes_files.extend(resolved_files)
            
            if resolved_quotes_files:
                quotes_df = read_quotes(resolved_quotes_files)
            if not quotes_df.empty:
                normalized_quotes = normalize_quotes(quotes_df)
                results['quotes'] = normalized_quotes
                self.logger.info(f"Обработано {len(normalized_quotes)} quotes")
            else:
                self.logger.warning("Quotes файлы не найдены")
        else:
            self.logger.warning("Quotes файлы не найдены")
        
        # Создаем tape из trades
        if 'trades' in results and not results['trades'].empty:
            results['tape'] = results['trades'].copy()
            self.logger.info("Создан tape из trades")
        
        # Сохраняем результаты
        self.logger.info(f"🔍 Путь сохранения: {output_path.absolute()}")
        for key, df in results.items():
            if not df.empty:
                output_file = output_path / f"{key}.parquet"
                self.logger.info(f"💾 Сохраняю {key} в {output_file.absolute()}")
                try:
                    df.to_parquet(output_file, engine="pyarrow", index=False)
                    self.logger.info(f"✅ Сохранен {key} в {output_file}")
                except Exception as e:
                    self.logger.error(f"❌ Ошибка сохранения {key}: {e}")
                    raise
        
        # Возвращаем данные в формате, ожидаемом pipeline
        trades_df = results.get('trades', pd.DataFrame())
        depth_df = results.get('book_top', pd.DataFrame())
        
        return trades_df, depth_df

# ---------- функции для обратной совместимости ----------
def import_raw_data(data: Dict = None, config: Dict = None) -> Dict[str, pd.DataFrame]:
    """Функция для обратной совместимости"""
    if data is None:
        data = {}
    if config is None:
        config = {}
    
    block = ImportRawBlock(config)
    return block.run_with_progress(data, config)
