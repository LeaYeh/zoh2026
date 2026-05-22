from pathlib import Path
import pandas as pd
import yfinance as yf

RAW = Path("data/raw")
PROCESSED = Path("data/processed")


def fetch_ticker(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def fetch_and_save(symbol: str, start: str, end: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{symbol}.parquet"
    df = fetch_ticker(symbol, start, end)
    df.to_parquet(path)
    return path


def load_raw(symbol: str) -> pd.DataFrame:
    return pd.read_parquet(RAW / f"{symbol}.parquet")


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def save_processed(df: pd.DataFrame, name: str) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path)
    return path
