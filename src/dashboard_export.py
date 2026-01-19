"""
Dashboard Data Export Script for Tableau

This script prepares all data files needed for the Tableau dashboard.
Run this AFTER running the three notebooks.

Usage:
    python src/dashboard_export.py

Author: AAI-530 Team
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'archive')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all_stations():
    """Load and combine data from all monitoring stations."""
    all_data = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(DATA_DIR, filename)
            station_name = filename.replace('_joined.csv', '').replace('_', ' ').title()
            df = pd.read_csv(filepath)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Station'] = station_name
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


def classify_water_quality(row):
    """Classify water quality based on sensor readings."""
    unsafe_count = 0
    warning_count = 0
    
    if pd.notna(row.get('Turbidity')):
        if row['Turbidity'] > 50:
            unsafe_count += 1
        elif row['Turbidity'] > 5:
            warning_count += 1
    
    if pd.notna(row.get('Conductivity')):
        if row['Conductivity'] > 50000:
            unsafe_count += 1
        elif row['Conductivity'] > 30000:
            warning_count += 1
    
    if pd.notna(row.get('Temp')):
        if row['Temp'] < 5 or row['Temp'] > 35:
            unsafe_count += 1
        elif row['Temp'] < 10 or row['Temp'] > 30:
            warning_count += 1
    
    if unsafe_count >= 1:
        return 'Unsafe'
    elif warning_count >= 1:
        return 'Warning'
    else:
        return 'Safe'


def create_dashboard_data():
    """Create all data files needed for Tableau dashboard."""
    
    print("=" * 60)
    print("CREATING TABLEAU DASHBOARD DATA")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data from all stations...")
    df = load_all_stations()
    print(f"   Loaded {len(df):,} records from {df['Station'].nunique()} stations")
    
    # Add classification
    print("\n2. Classifying water quality...")
    df['Quality_Status'] = df.apply(classify_water_quality, axis=1)
    
    # Add time features
    df['Date'] = df['Timestamp'].dt.date
    df['Hour'] = df['Timestamp'].dt.hour
    df['Month'] = df['Timestamp'].dt.month
    df['Year'] = df['Timestamp'].dt.year
    df['DayOfWeek'] = df['Timestamp'].dt.day_name()
    
    # === FILE 1: Current Status Data ===
    print("\n3. Creating current status data...")
    # Get most recent readings per station
    current_status = df.sort_values('Timestamp').groupby('Station').last().reset_index()
    current_status = current_status[['Station', 'Timestamp', 'Conductivity', 'Turbidity', 
                                      'Temp', 'NO3', 'Level', 'Quality_Status']]
    current_status.to_csv(os.path.join(OUTPUT_DIR, 'dashboard_current_status.csv'), index=False)
    print(f"   Saved: dashboard_current_status.csv ({len(current_status)} stations)")
    
    # === FILE 2: Historical Summary Data ===
    print("\n4. Creating historical summary data...")
    # Daily aggregations
    daily_summary = df.groupby(['Station', 'Date']).agg({
        'Turbidity': ['mean', 'max', 'min', 'std'],
        'Conductivity': ['mean', 'max', 'min'],
        'Temp': ['mean', 'max', 'min'],
        'Quality_Status': lambda x: (x == 'Safe').mean() * 100  # % Safe
    }).reset_index()
    daily_summary.columns = ['Station', 'Date', 'Turbidity_Mean', 'Turbidity_Max', 
                             'Turbidity_Min', 'Turbidity_Std', 'Conductivity_Mean',
                             'Conductivity_Max', 'Conductivity_Min', 'Temp_Mean',
                             'Temp_Max', 'Temp_Min', 'Safe_Percentage']
    daily_summary.to_csv(os.path.join(OUTPUT_DIR, 'dashboard_daily_summary.csv'), index=False)
    print(f"   Saved: dashboard_daily_summary.csv ({len(daily_summary)} daily records)")
    
    # === FILE 3: Quality Distribution Data ===
    print("\n5. Creating quality distribution data...")
    quality_dist = df.groupby(['Station', 'Quality_Status']).size().reset_index(name='Count')
    quality_dist_pct = df.groupby('Station')['Quality_Status'].value_counts(normalize=True).reset_index(name='Percentage')
    quality_dist_pct['Percentage'] = quality_dist_pct['Percentage'] * 100
    quality_dist = quality_dist.merge(quality_dist_pct, on=['Station', 'Quality_Status'])
    quality_dist.to_csv(os.path.join(OUTPUT_DIR, 'dashboard_quality_distribution.csv'), index=False)
    print(f"   Saved: dashboard_quality_distribution.csv")
    
    # === FILE 4: Hourly Patterns ===
    print("\n6. Creating hourly pattern data...")
    hourly_patterns = df.groupby(['Station', 'Hour']).agg({
        'Turbidity': 'mean',
        'Conductivity': 'mean',
        'Temp': 'mean'
    }).reset_index()
    hourly_patterns.to_csv(os.path.join(OUTPUT_DIR, 'dashboard_hourly_patterns.csv'), index=False)
    print(f"   Saved: dashboard_hourly_patterns.csv")
    
    # === FILE 5: Monthly Trends ===
    print("\n7. Creating monthly trend data...")
    monthly_trends = df.groupby(['Station', 'Year', 'Month']).agg({
        'Turbidity': 'mean',
        'Conductivity': 'mean',
        'Temp': 'mean',
        'Quality_Status': lambda x: (x == 'Safe').mean() * 100
    }).reset_index()
    monthly_trends.columns = ['Station', 'Year', 'Month', 'Avg_Turbidity', 
                              'Avg_Conductivity', 'Avg_Temp', 'Safe_Percentage']
    monthly_trends.to_csv(os.path.join(OUTPUT_DIR, 'dashboard_monthly_trends.csv'), index=False)
    print(f"   Saved: dashboard_monthly_trends.csv")
    
    # === FILE 6: Full processed data ===
    print("\n8. Saving full processed data...")
    df.to_csv(os.path.join(OUTPUT_DIR, 'water_quality_full_processed.csv'), index=False)
    print(f"   Saved: water_quality_full_processed.csv ({len(df):,} records)")
    
    print("\n" + "=" * 60)
    print("DASHBOARD DATA EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print("  1. dashboard_current_status.csv      - Current sensor readings")
    print("  2. dashboard_daily_summary.csv       - Daily aggregated statistics")
    print("  3. dashboard_quality_distribution.csv - Quality status counts")
    print("  4. dashboard_hourly_patterns.csv     - Hourly patterns by station")
    print("  5. dashboard_monthly_trends.csv      - Monthly trends")
    print("  6. water_quality_full_processed.csv  - Full dataset for Tableau")
    

if __name__ == "__main__":
    create_dashboard_data()

