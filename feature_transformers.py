import sys
import warnings
import gc
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin

class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Трансформер извлекает час, день, день недели из TransactionDT=
    """
    
    def __init__(self, time_col = 'TransactionDT'):
        self.time_col = time_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.time_col in X.columns:
            X['hour'] = (X[self.time_col] // 3600) % 24
            X['day'] = X[self.time_col] // (3600 * 24)
            X['weekday'] = X['day'] % 7
            X['is_weekend'] = (X['weekday'] >= 5).astype(int)
        return X


class UIDFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Трансформер для создания User ID на основе комбинации фичей
    """
    
    def __init__(self):
        self.uid_stats = {}
    
    def fit(self, X, y=None):
        X = X.copy()
        
        X['uid'] = X['card1'].astype(str) + '_' + X['addr1'].astype(str)
        
        self.uid_stats['uid_count'] = X['uid'].value_counts().to_dict()
        self.uid_stats['uid_amt_mean'] = X.groupby('uid')['TransactionAmt'].mean().to_dict()
        self.uid_stats['uid_amt_std'] = X.groupby('uid')['TransactionAmt'].std().to_dict()
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X['uid'] = X['card1'].astype(str) + '_' + X['addr1'].astype(str)
        
        X['uid_count'] = X['uid'].map(self.uid_stats['uid_count']).fillna(1)
        X['uid_amt_mean'] = X['uid'].map(self.uid_stats['uid_amt_mean']).fillna(X['TransactionAmt'].mean())
        X['uid_amt_std'] = X['uid'].map(self.uid_stats['uid_amt_std']).fillna(0)
        X['uid_amt_diff'] = X['TransactionAmt'] - X['uid_amt_mean']
        
        X.drop('uid', axis=1, inplace=True)
        
        return X


class DNormalizationTransformer(BaseEstimator, TransformerMixin):
    """
    Трансформер для нормализации D-признаков
    """
    
    def __init__(self):
        self.d_cols = [f'D{i}' for i in range(1, 16)]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.d_cols:
            if col in X.columns:
                X[f'{col}_normalized'] = X['TransactionDT'] / 86400 - X[col]
        
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Частотное кодирование категориальных признаков
    """
    
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.freq_maps = {}
    
    def fit(self, X, y=None):
        for col in self.cols:
            if col in X.columns:
                self.freq_maps[col] = X[col].value_counts(normalize=True).to_dict()
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.cols:
            if col in X.columns and col in self.freq_maps:
                X[f'{col}_freq'] = X[col].map(self.freq_maps[col]).fillna(0)
        
        return X


class AggregationTransformer(BaseEstimator, TransformerMixin):
    """
    Создание агрегированных признаков по ключам группировки group_cols
    """
    
    def __init__(self, group_cols, agg_cols, agg_funcs = ['mean', 'std']):
        self.group_cols = group_cols
        self.agg_cols = agg_cols
        self.agg_funcs = agg_funcs
        self.agg_stats = {}
    
    def fit(self, X, y=None):
        for group_col in self.group_cols:
            if group_col not in X.columns:
                continue
            
            self.agg_stats[group_col] = {}
            
            for agg_col in self.agg_cols:
                if agg_col not in X.columns:
                    continue
                
                self.agg_stats[group_col][agg_col] = {}
                
                for func in self.agg_funcs:
                    agg_values = X.groupby(group_col)[agg_col].agg(func).to_dict()
                    self.agg_stats[group_col][agg_col][func] = agg_values
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for group_col, col_stats in self.agg_stats.items():
            if group_col not in X.columns:
                continue
                
            for agg_col, func_stats in col_stats.items():
                for func, values in func_stats.items():
                    new_col = f'{group_col}_{agg_col}_{func}'
                    X[new_col] = X[group_col].map(values)
        
        return X


class MissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Создание признаков по статистикам пропущенных значений - у мошенников таких может быть больше
    """
    
    def __init__(self, cols: List[str] = None, threshold: float = 0.01):
        self.cols = cols
        self.threshold = threshold
        self.missing_cols = []
    
    def fit(self, X, y=None):
        if self.cols is None:
            missing_pct = X.isnull().mean()
            self.missing_cols = missing_pct[
                (missing_pct > self.threshold) & (missing_pct < 0.99)
            ].index.tolist()
        else:
            self.missing_cols = self.cols
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X['n_missing'] = X.isnull().sum(axis=1)
        
        for col in self.missing_cols[:20]: 
            if col in X.columns:
                X[f'{col}_isna'] = X[col].isnull().astype(int)
        
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Отбор признаков для финальной модели
    """
    
    def __init__(self, exclude_cols: List[str] = None):
        self.exclude_cols = exclude_cols or []
        self.feature_cols = []
    
    def fit(self, X, y=None):
        self.feature_cols = [
            col for col in X.columns 
            if col not in self.exclude_cols 
            and X[col].dtype in ['int8', 'int16', 'int32', 'int64', 
                                  'float16', 'float32', 'float64']
        ]
        return self
    
    def transform(self, X):
        available_cols = [col for col in self.feature_cols if col in X.columns]
        X_selected = X[available_cols].copy()
        
        X_selected = X_selected.fillna(-999)
        
        return X_selected
