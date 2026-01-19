"""
Data Processing Utilities for Smart Water Quality Monitoring System

This module provides functions for loading, cleaning, and preprocessing
water quality sensor data from multiple monitoring stations.

Author: AAI-530 Team
Date: 2026
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime


def load_single_station(filepath: str) -> pd.DataFrame:
    """
    Load data from a single monitoring station CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with parsed timestamps and station name added
    """
    # Extract station name from filename
    station_name = os.path.basename(filepath).replace('_joined.csv', '').replace('_', ' ')
    
    # Load the data
    df = pd.read_csv(filepath)
    
    # Parse timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Add station identifier
    df['Station'] = station_name
    
    return df


def load_all_stations(data_dir: str) -> pd.DataFrame:
    """
    Load and concatenate data from all monitoring stations.
    
    Parameters:
    -----------
    data_dir : str
        Path to directory containing CSV files
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame from all stations
    """
    all_data = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            df = load_single_station(filepath)
            all_data.append(df)
            print(f"Loaded: {filename} - {len(df)} records")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records loaded: {len(combined_df)}")
    
    return combined_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the water quality dataset.
    
    Operations:
    - Remove duplicates
    - Handle missing values
    - Remove obvious outliers
    - Sort by timestamp
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['Timestamp', 'Station'])
    print(f"Removed {initial_rows - len(df_clean)} duplicate rows")
    
    # Sort by timestamp and station
    df_clean = df_clean.sort_values(['Station', 'Timestamp']).reset_index(drop=True)
    
    # Handle negative values (physically impossible for most parameters)
    numeric_cols = ['Conductivity', 'NO3', 'Temp', 'Turbidity', 'Level']
    for col in numeric_cols:
        if col in df_clean.columns:
            negative_count = (df_clean[col] < 0).sum()
            if negative_count > 0:
                df_clean.loc[df_clean[col] < 0, col] = np.nan
                print(f"Set {negative_count} negative values to NaN in {col}")
    
    # Remove extreme outliers using IQR method
    for col in ['Conductivity', 'Turbidity']:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.01)
            Q3 = df_clean[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outlier_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = np.nan
            print(f"Set {outlier_count} outliers to NaN in {col}")
    
    return df_clean


def fill_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Fill missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with missing values
    method : str
        Method for filling: 'interpolate', 'forward', 'mean'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with filled values
    """
    df_filled = df.copy()
    
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if method == 'interpolate':
        # Interpolate within each station
        for station in df_filled['Station'].unique():
            mask = df_filled['Station'] == station
            df_filled.loc[mask, numeric_cols] = df_filled.loc[mask, numeric_cols].interpolate(
                method='time', limit=6  # Limit to 6 hours gap
            )
    elif method == 'forward':
        for station in df_filled['Station'].unique():
            mask = df_filled['Station'] == station
            df_filled.loc[mask, numeric_cols] = df_filled.loc[mask, numeric_cols].fillna(method='ffill', limit=6)
    elif method == 'mean':
        for col in numeric_cols:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    
    return df_filled


def classify_water_quality(row: pd.Series) -> str:
    """
    Classify water quality based on sensor readings.
    
    Based on Australian water quality guidelines:
    - Safe: Normal operating conditions
    - Warning: Parameters approaching concerning levels
    - Unsafe: Parameters exceed safe thresholds
    
    Parameters:
    -----------
    row : pd.Series
        Row of sensor readings
        
    Returns:
    --------
    str
        Classification label: 'Safe', 'Warning', or 'Unsafe'
    """
    # Initialize scores
    unsafe_count = 0
    warning_count = 0
    
    # Check Turbidity (NTU)
    if pd.notna(row.get('Turbidity')):
        if row['Turbidity'] > 50:
            unsafe_count += 1
        elif row['Turbidity'] > 5:
            warning_count += 1
    
    # Check Conductivity (µS/cm) - High values indicate contamination
    if pd.notna(row.get('Conductivity')):
        if row['Conductivity'] > 50000:  # Very high for river water
            unsafe_count += 1
        elif row['Conductivity'] > 30000:
            warning_count += 1
    
    # Check Temperature (°C)
    if pd.notna(row.get('Temp')):
        if row['Temp'] < 5 or row['Temp'] > 35:
            unsafe_count += 1
        elif row['Temp'] < 10 or row['Temp'] > 30:
            warning_count += 1
    
    # Determine classification
    if unsafe_count >= 1:
        return 'Unsafe'
    elif warning_count >= 1:
        return 'Warning'
    else:
        return 'Safe'


def add_water_quality_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add water quality classification labels to the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sensor readings
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'Quality_Status' column
    """
    df_labeled = df.copy()
    df_labeled['Quality_Status'] = df_labeled.apply(classify_water_quality, axis=1)
    
    # Print distribution
    print("\nWater Quality Distribution:")
    print(df_labeled['Quality_Status'].value_counts())
    
    return df_labeled


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional time-based features for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Timestamp column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional time features
    """
    df_features = df.copy()
    
    df_features['Hour'] = df_features['Timestamp'].dt.hour
    df_features['DayOfWeek'] = df_features['Timestamp'].dt.dayofweek
    df_features['DayOfMonth'] = df_features['Timestamp'].dt.day
    df_features['WeekOfYear'] = df_features['Timestamp'].dt.isocalendar().week
    df_features['Month'] = df_features['Timestamp'].dt.month
    df_features['Year'] = df_features['Timestamp'].dt.year
    df_features['IsWeekend'] = df_features['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Cyclical encoding for hour
    df_features['Hour_sin'] = np.sin(2 * np.pi * df_features['Hour'] / 24)
    df_features['Hour_cos'] = np.cos(2 * np.pi * df_features['Hour'] / 24)
    
    # Cyclical encoding for month
    df_features['Month_sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
    df_features['Month_cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
    
    return df_features


def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features for time series modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with time series data
    columns : List[str]
        Columns to create lags for
    lags : List[int]
        List of lag periods (in hours)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag features
    """
    df_lagged = df.copy()
    
    for station in df_lagged['Station'].unique():
        mask = df_lagged['Station'] == station
        station_df = df_lagged.loc[mask].copy()
        
        for col in columns:
            if col in station_df.columns:
                for lag in lags:
                    df_lagged.loc[mask, f'{col}_lag_{lag}h'] = station_df[col].shift(lag)
    
    return df_lagged


def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Create rolling statistics features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with time series data
    columns : List[str]
        Columns to calculate rolling stats for
    windows : List[int]
        List of window sizes (in hours)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling features
    """
    df_rolling = df.copy()
    
    for station in df_rolling['Station'].unique():
        mask = df_rolling['Station'] == station
        station_df = df_rolling.loc[mask].copy()
        
        for col in columns:
            if col in station_df.columns:
                for window in windows:
                    df_rolling.loc[mask, f'{col}_rolling_mean_{window}h'] = station_df[col].rolling(window=window, min_periods=1).mean()
                    df_rolling.loc[mask, f'{col}_rolling_std_{window}h'] = station_df[col].rolling(window=window, min_periods=1).std()
    
    return df_rolling


def prepare_lstm_sequences(data: np.ndarray, target_col_idx: int, 
                           sequence_length: int = 24, 
                           forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM model training.
    
    Parameters:
    -----------
    data : np.ndarray
        Normalized feature array
    target_col_idx : int
        Index of target column in data
    sequence_length : int
        Number of past time steps to use
    forecast_horizon : int
        Number of future steps to predict
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        X (sequences) and y (targets) arrays
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data) - forecast_horizon + 1):
        X.append(data[i - sequence_length:i])
        y.append(data[i + forecast_horizon - 1, target_col_idx])
    
    return np.array(X), np.array(y)


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for each station.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics by station
    """
    numeric_cols = ['Conductivity', 'NO3', 'Temp', 'Turbidity', 'Level', 'Q']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    summary = df.groupby('Station')[available_cols].agg([
        'count', 'mean', 'std', 'min', 'max', 
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
    ]).round(2)
    
    return summary


if __name__ == "__main__":
    # Example usage
    data_dir = "../archive"
    
    # Load all data
    df = load_all_stations(data_dir)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Fill missing values
    df_filled = fill_missing_values(df_clean, method='interpolate')
    
    # Add classification labels
    df_labeled = add_water_quality_labels(df_filled)
    
    # Create time features
    df_features = create_time_features(df_labeled)
    
    # Display summary
    print("\n" + "="*50)
    print("Dataset Summary")
    print("="*50)
    print(f"Total records: {len(df_features)}")
    print(f"Date range: {df_features['Timestamp'].min()} to {df_features['Timestamp'].max()}")
    print(f"Stations: {df_features['Station'].nunique()}")
    print(f"\nColumns: {df_features.columns.tolist()}")

