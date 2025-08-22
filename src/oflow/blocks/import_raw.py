from __future__ import annotations
import os, math, glob, json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import yaml

# ---------- утилиты ----------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_ns(ts: pd.Series) -> pd.Series:
    # timestamp в секундах с дробью -> int64 наносекунды
    # или datetime -> наносекунды
    if pd.api.types.is_datetime64_any_dtype(ts):
        return ts.astype("int64")  # уже в наносекундах
    else:
        return (ts.astype("float64") * 1_000_000_000).round().astype("int64")

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
    out = pd.DataFrame({
        "ts_ns": to_ns(df["timestamp"]),
        "exchange": df["exchange"].astype("string"),
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
    if arr is None: return rows
    
    # Обрабатываем numpy array
    if hasattr(arr, '__len__') and len(arr) == 0:
        return rows
    
    for rec in arr:
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
    d = row.get("delta", {})
    # разные биржи: 'bid'/'ask', 'b'/'a', 'bids'/'asks'
    arr_bid = d.get("bid") if "bid" in d else (d.get("b") if "b" in d else (d.get("bids") if "bids" in d else None))
    arr_ask = d.get("ask") if "ask" in d else (d.get("a") if "a" in d else (d.get("asks") if "asks" in d else None))
    rows = []
    rows += _rows_from_side("bid", arr_bid)
    rows += _rows_from_side("ask", arr_ask)
    return rows

def read_depth(paths: List[str]) -> pd.DataFrame:
    dfs=[]
    for p in paths:
        try:
            df = pd.read_parquet(p, engine="pyarrow")
            if "delta" not in df.columns:
                print(f"[WARN] depth has no 'delta' in {p}")
                continue
            df = df.assign(_src_file=p)
            dfs.append(df[["exchange","market","symbol","timestamp","delta","_src_file"]])
        except Exception as e:
            print(f"[ERR] read depth {p}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["exchange","market","symbol","timestamp","delta","_src_file"])

def normalize_depth(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    records=[]
    for i,row in df.iterrows():
        instrument,_ = norm_instrument(row["symbol"])
        base = instrument.replace("-","").replace("PERP","")
        for side, price, size, action in flatten_depth_delta(row):
            # Обрабатываем timestamp корректно
            if isinstance(row["timestamp"], pd.Timestamp):
                ts_ns = row["timestamp"].value  # уже в наносекундах
            else:
                ts_ns = int(round(row["timestamp"]*1_000_000_000))
            
            records.append((
                ts_ns,
                row["exchange"], row["market"], instrument, base,
                side, float(price), float(size), action, row["_src_file"]
            ))
    out = pd.DataFrame.from_records(
        records,
        columns=["ts_ns","exchange","market","instrument","base_symbol","side","price","size","action","_src_file"]
    )
    if out.empty: return out
    out["ts_ns"]=out["ts_ns"].astype("int64")
    out.sort_values("ts_ns", inplace=True, kind="mergesort")
    return out

# ---------- основной раннер ----------
def expand_masks(mask: str) -> List[str]:
    return sorted(glob.glob(mask))

def collect_paths(src_cfg: Dict) -> Tuple[List[str], List[str]]:
    trade_paths, depth_paths = [], []
    for mkt in ("futures","spot"):
        if mkt not in src_cfg: continue
        for exch,paths in src_cfg[mkt].items():
            if "trades" in paths:
                trade_paths += expand_masks(paths["trades"])
            if "depth" in paths:
                depth_paths += expand_masks(paths["depth"])
    return trade_paths, depth_paths

def run(config_dir: str = "configs") -> Dict[str,str]:
    src = load_yaml(Path(config_dir,"sources.yaml"))
    
    # Создаем директории по умолчанию
    canon_dir = Path("data/canon")
    out_trades = canon_dir / "trades"
    out_l2 = canon_dir / "l2_delta"
    ensure_dir(out_trades); ensure_dir(out_l2)

    trade_files, depth_files = collect_paths(src)
    print(f"[INFO] trades files: {len(trade_files)}, depth files: {len(depth_files)}")

    # TRADES
    df_t_raw = read_trades(trade_files)
    df_trades = normalize_trades(df_t_raw)
    if not df_trades.empty:
        # посуточно
        df_trades["date"] = pd.to_datetime(df_trades["ts_ns"]).dt.floor("D")
        for dt, dfg in df_trades.groupby("date", sort=True):
            fn = out_trades / f"ETHUSDT_{dt.strftime('%Y-%m-%d')}.parquet"
            dfg.drop(columns=["date"]).to_parquet(fn, engine="pyarrow", index=False)
            print(f"[SAVE] {fn} rows={len(dfg)}")

    # DEPTH
    df_d_raw = read_depth(depth_files)
    df_l2 = normalize_depth(df_d_raw)
    if not df_l2.empty:
        df_l2["date"] = pd.to_datetime(df_l2["ts_ns"]).dt.floor("D")
        for dt, dfg in df_l2.groupby("date", sort=True):
            fn = out_l2 / f"ETHUSDT_{dt.strftime('%Y-%m-%d')}.parquet"
            dfg.drop(columns=["date"]).to_parquet(fn, engine="pyarrow", index=False)
            print(f"[SAVE] {fn} rows={len(dfg)}")

    return {"trades_dir": str(out_trades), "l2_dir": str(out_l2)}

def run_test(config_dir: str = "configs") -> Dict[str,str]:
    """Тестовая версия - обрабатывает файлы с разных бирж для демонстрации агрегации"""
    src = load_yaml(Path(config_dir,"sources.yaml"))
    
    # Создаем директории по умолчанию
    canon_dir = Path("data/canon")
    out_trades = canon_dir / "trades"
    out_l2 = canon_dir / "l2_delta"
    ensure_dir(out_trades); ensure_dir(out_l2)

    trade_files, depth_files = collect_paths(src)
    print(f"[INFO] Всего файлов: trades={len(trade_files)}, depth={len(depth_files)}")
    
    # Берем файлы с разных бирж для демонстрации агрегации
    # Выбираем по одному файлу с каждой биржи (последние доступные)
    test_trades = []
    test_depth = []
    
    # Группируем файлы по биржам
    trades_by_exchange = {}
    depth_by_exchange = {}
    
    for f in trade_files:
        path_parts = f.split('\\')
        if len(path_parts) >= 4:
            exchange = path_parts[-2]
            if exchange not in trades_by_exchange:
                trades_by_exchange[exchange] = []
            trades_by_exchange[exchange].append(f)
    
    for f in depth_files:
        path_parts = f.split('\\')
        if len(path_parts) >= 4:
            exchange = path_parts[-2]
            if exchange not in depth_by_exchange:
                depth_by_exchange[exchange] = []
            depth_by_exchange[exchange].append(f)
    
    # Берем последний файл с каждой биржи
    for exchange in ['binance', 'bybit', 'okx']:
        if exchange in trades_by_exchange:
            test_trades.append(sorted(trades_by_exchange[exchange])[-1])
        if exchange in depth_by_exchange:
            test_depth.append(sorted(depth_by_exchange[exchange])[-1])
    
    print(f"[TEST] Тестируем с файлами:")
    print(f"  Trades ({len(test_trades)} файлов):")
    for f in test_trades:
        print(f"    {f}")
    print(f"  Depth ({len(test_depth)} файлов):")
    for f in test_depth:
        print(f"    {f}")

    # TRADES
    df_t_raw = read_trades(test_trades)
    print(f"[TRADES] Прочитано: {len(df_t_raw)} строк")
    df_trades = normalize_trades(df_t_raw)
    print(f"[TRADES] Нормализовано: {len(df_trades)} строк")
    
    if not df_trades.empty:
        # сохраняем как есть, без группировки по дням
        fn = out_trades / "ETHUSDT_test_trades_multi.parquet"
        df_trades.to_parquet(fn, engine="pyarrow", index=False)
        print(f"[SAVE] {fn} rows={len(df_trades)}")

    # DEPTH
    df_d_raw = read_depth(test_depth)
    print(f"[DEPTH] Прочитано: {len(df_d_raw)} строк")
    df_l2 = normalize_depth(df_d_raw)
    print(f"[DEPTH] Нормализовано: {len(df_l2)} строк")
    
    if not df_l2.empty:
        # сохраняем как есть, без группировки по дням
        fn = out_l2 / "ETHUSDT_test_depth_multi.parquet"
        df_l2.to_parquet(fn, engine="pyarrow", index=False)
        print(f"[SAVE] {fn} rows={len(df_l2)}")

    return {"trades_dir": str(out_trades), "l2_dir": str(out_l2)}

if __name__ == "__main__":
    run_test()  # используем тестовую версию
