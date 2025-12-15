import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from functools import reduce

def load_data(data_dir):
    """
    Загрузка данных
    """
    train_transaction = pd.read_csv(data_dir / "train_transaction.csv")
    test_transaction = pd.read_csv(data_dir / "test_transaction.csv")
    
    train_identity = pd.read_csv(data_dir / "train_identity.csv")
    test_identity = pd.read_csv(data_dir / "test_identity.csv")
    
    # Объединение по TransactionID всех фичей
    train = train_transaction.merge(train_identity, on="TransactionID", how="left")
    test = test_transaction.merge(test_identity, on="TransactionID", how="left")
    
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    
    del train_transaction, test_transaction, train_identity, test_identity
    
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")

    return train, test, sample_sub


def get_feature_types(df: pd.DataFrame, target: str = "isFraud") -> Dict[str, List[str]]:
    """
    Определение типов признаков
    """
    feature_types = {
        "numeric": [],
        "categorical": [],
        "binary": [],
        "datetime": []
    }
    
    exclude_cols = [target, "TransactionID"]
    
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        dtype = df[col].dtype
        nunique = df[col].nunique()
        
        if dtype == "object" or dtype.name == "category":
            feature_types["categorical"].append(col)
        elif nunique == 2:
            feature_types["binary"].append(col)
        elif nunique <= 50 and dtype in ["int8", "int16", "int32", "int64"]:
            feature_types["categorical"].append(col)
        else:
            feature_types["numeric"].append(col)
    
    print(f"Numeric: {len(feature_types['numeric'])}, "
          f"Categorical: {len(feature_types['categorical'])}, "
          f"Binary: {len(feature_types['binary'])}")
    
    return feature_types


def calculate_missing_stats(df):
    """
    Расчет статистики пропущенных значений
    """
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    missing_df = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values
    })
    
    missing_df = missing_df[missing_df["missing_count"] > 0]
    missing_df = missing_df.sort_values("missing_pct", ascending=False)
    
    return missing_df.reset_index(drop=True)


def create_time_features(df, time_col = "TransactionDT"):
    """
    Создание временных признаков из TransactionDT
    """
    df = df.copy()
    
    df["Transaction_hour"] = (df[time_col] // 3600) % 24
    df["Transaction_day"] = df[time_col] // (3600 * 24)
    df["Transaction_weekday"] = df["Transaction_day"] % 7
    df["Transaction_month_day"] = df["Transaction_day"] % 30
        
    return df


def create_aggregation_features(df, group_cols, agg_cols, agg_funcs = ["mean", "std", "min", "max"]):
    """
    Создание агрегированных признаков
    """
    df = df.copy()
    
    for group_col in group_cols:
        if group_col not in df.columns:
            continue
            
        for agg_col in agg_cols:
            if agg_col not in df.columns:
                continue
                
            for func in agg_funcs:
                new_col = f"{group_col}_{agg_col}_{func}"
                
                agg_values = df.groupby(group_col)[agg_col].transform(func)
                df[new_col] = agg_values
    
    return df


def create_frequency_features(df, cols):
    """
    Создание частотных признаков
    """
    df = df.copy()
    
    for col in cols:
        if col not in df.columns:
            continue
            
        freq = df[col].map(df[col].value_counts())
        df[f"{col}_count"] = freq
    
    return df


def create_uid_features(df):
    """
    Создание User ID признаков
    """
    df = df.copy()
    
    # UID
    df["uid"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
    
    df["uid2"] = (
        df["card1"].astype(str) + "_" + 
        df["addr1"].astype(str) + "_" +
        df["P_emaildomain"].astype(str)
    )
    
    # UID только по card
    df["uid3"] = (
        df["card1"].astype(str) + "_" + 
        df["card2"].astype(str) + "_" + 
        df["card3"].astype(str) + "_" +
        df["card4"].astype(str) + "_" +
        df["card5"].astype(str)
    )
        
    return df


def normalize_d_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализация D-признаков относительно TransactionDT
    """
    df = df.copy()
    
    d_cols = [f"D{i}" for i in range(1, 16)]
    
    for col in d_cols:
        if col in df.columns:
            df[f"{col}_normalized"] = df["TransactionDT"] / (24 * 3600) - df[col]
        
    return df
