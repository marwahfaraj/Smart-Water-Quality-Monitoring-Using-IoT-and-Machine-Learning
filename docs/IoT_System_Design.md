# IoT System Design Documentation

## Smart Water Quality Monitoring System - Technical Architecture

This document provides detailed technical specifications for the IoT-based water quality monitoring system, addressing sensor deployment, edge processing, networking protocols, and data pipeline architecture.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         IoT WATER QUALITY MONITORING SYSTEM                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   SENSORS    │    │    EDGE      │    │   NETWORK    │    │    CLOUD     │  │
│  │   LAYER      │───▶│  PROCESSING  │───▶│   LAYER      │───▶│  PLATFORM    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │                   │          │
│         ▼                   ▼                   ▼                   ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ • Turbidity  │    │ • Data       │    │ • MQTT       │    │ • Storage    │  │
│  │ • Conductivity│   │   Validation │    │   Protocol   │    │ • ML Models  │  │
│  │ • Temperature│    │ • Filtering  │    │ • 4G/LTE     │    │ • Dashboard  │  │
│  │ • pH/NO3     │    │ • Aggregation│    │   Cellular   │    │ • Alerts     │  │
│  │ • Flow Rate  │    │ • Buffering  │    │ • LoRaWAN    │    │              │  │
│  │ • Water Level│    │              │    │   (backup)   │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Sensor Layer - Specifications & Deployment

### 2.1 Monitoring Station Locations

| Station ID | Location | River System | Coordinates | Deployment Date |
|------------|----------|--------------|-------------|-----------------|
| STN-001 | Coquette Point | Johnstone River | -17.9167°, 146.0333° | Mar 2016 |
| STN-002 | Innisfail | Johnstone River | -17.5247°, 146.0319° | Mar 2016 |
| STN-003 | Deeral | Mulgrave River | -17.2167°, 145.9500° | Mar 2016 |
| STN-004 | Dumbleton | Pioneer River | -21.2000°, 149.0667° | Mar 2016 |
| STN-005 | Sucrogen | Plane Creek | -21.4167°, 149.0000° | Mar 2016 |
| STN-006 | Glen Isla | Proserpine River | -20.4000°, 148.5833° | Mar 2016 |
| STN-007 | East Russell | Russell River | -17.2333°, 145.9333° | Mar 2016 |
| STN-008 | Homebush | Sandy Creek | -21.2667°, 149.0333° | Mar 2016 |
| STN-009 | Sorbellos Road | Sandy Creek | -21.3000°, 149.0167° | Mar 2016 |
| STN-010 | Euramo | Tully River | -17.9833°, 145.9500° | Mar 2016 |
| STN-011 | Tully Gorge NP | Tully River | -17.7667°, 145.6500° | Mar 2016 |

### 2.2 Sensor Specifications

#### Turbidity Sensor
| Parameter | Specification |
|-----------|---------------|
| **Model** | YSI EXO Turbidity Smart Sensor |
| **Measurement Range** | 0 - 4000 NTU |
| **Resolution** | 0.01 NTU |
| **Accuracy** | ±2% of reading or 0.3 NTU (whichever is greater) |
| **Response Time** | < 2 seconds |
| **Operating Temperature** | -5°C to +50°C |
| **Maintenance Interval** | 30 days (wiper cleaning) |

#### Conductivity Sensor
| Parameter | Specification |
|-----------|---------------|
| **Model** | YSI EXO Conductivity/Temperature Smart Sensor |
| **Measurement Range** | 0 - 200,000 µS/cm |
| **Resolution** | 0.001 - 1 µS/cm (range dependent) |
| **Accuracy** | ±0.5% of reading or 1 µS/cm |
| **Temperature Compensation** | Automatic (0-50°C) |
| **Calibration Interval** | 90 days |

#### Temperature Sensor (Integrated)
| Parameter | Specification |
|-----------|---------------|
| **Measurement Range** | -5°C to +50°C |
| **Resolution** | 0.001°C |
| **Accuracy** | ±0.01°C |
| **Response Time** | < 1 second |

#### Nitrate (NO3) Sensor
| Parameter | Specification |
|-----------|---------------|
| **Model** | YSI EXO NitraLED UV Nitrate Sensor |
| **Measurement Range** | 0 - 200 mg/L NO3-N |
| **Resolution** | 0.01 mg/L |
| **Accuracy** | ±3% of reading or 0.5 mg/L |
| **Operating Temperature** | 0°C to 50°C |

#### Water Level Sensor
| Parameter | Specification |
|-----------|---------------|
| **Type** | Pressure Transducer (Submersible) |
| **Measurement Range** | 0 - 10 m |
| **Resolution** | 1 mm |
| **Accuracy** | ±0.1% Full Scale |
| **Overpressure Protection** | 2x rated pressure |

#### Flow Rate (Q) Sensor
| Parameter | Specification |
|-----------|---------------|
| **Type** | Acoustic Doppler Velocity Meter |
| **Velocity Range** | -6 to +6 m/s |
| **Accuracy** | ±1% of measured velocity |
| **Operating Depth** | 0.15 m minimum |

### 2.3 Sensor Placement Guidelines

```
                    RIVER CROSS-SECTION VIEW
    ════════════════════════════════════════════════════
    
         Bank                                    Bank
          │                                        │
          │    ┌─────────────────────────────┐    │
          │    │     WATER SURFACE           │    │
          │    │                             │    │
          │    │   ╔═══════════════════╗     │    │
          │    │   ║  SENSOR CLUSTER   ║     │    │
          │    │   ║  • Turbidity      ║     │    │
          │    │   ║  • Conductivity   ║◄────┼────┼── 0.5-1.0m depth
          │    │   ║  • Temperature    ║     │    │   (mid-column)
          │    │   ║  • NO3            ║     │    │
          │    │   ╚═══════════════════╝     │    │
          │    │                             │    │
          │    │   ┌───────────────────┐     │    │
          │    │   │  LEVEL SENSOR     │◄────┼────┼── River bed mount
          │    │   │  FLOW SENSOR      │     │    │
          │    │   └───────────────────┘     │    │
          │    │         RIVER BED           │    │
          │    └─────────────────────────────┘    │
          │                                        │
    ════════════════════════════════════════════════════
    
    PLACEMENT CRITERIA:
    • Minimum 10m from tributaries/discharge points
    • Avoid stagnant water zones
    • Protected from debris/flood damage
    • Accessible for maintenance
```

### 2.4 Sensor Limitations & Constraints

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Biofouling** | Sensor drift, false readings | Automated wipers, 30-day cleaning schedule |
| **Temperature Extremes** | Sensor damage > 50°C | Shaded housing, thermal insulation |
| **Flood Events** | Sensor displacement/damage | Secure mounting, backup sensors |
| **Power Outages** | Data gaps | Solar + battery backup (72hr capacity) |
| **Calibration Drift** | Accuracy degradation | Quarterly calibration, cross-validation |

---

## 3. Edge Processing Layer

### 3.1 Edge Device Specifications

| Component | Specification |
|-----------|---------------|
| **Processor** | ARM Cortex-A53 Quad-core 1.2GHz |
| **RAM** | 2GB DDR4 |
| **Storage** | 32GB eMMC + 64GB SD card buffer |
| **Operating System** | Linux-based RTOS |
| **Power Consumption** | < 5W average |
| **Operating Temperature** | -20°C to +60°C |
| **Enclosure Rating** | IP67 (weatherproof) |

### 3.2 Edge Processing Functions

```
┌─────────────────────────────────────────────────────────────────┐
│                    EDGE PROCESSING PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                                │
│  │ RAW SENSOR  │                                                │
│  │   DATA      │                                                │
│  └──────┬──────┘                                                │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. DATA VALIDATION                                       │   │
│  │    • Range checking (physical limits)                    │   │
│  │    • Null/NaN detection                                  │   │
│  │    • Sensor health verification                          │   │
│  └──────┬──────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 2. NOISE FILTERING                                       │   │
│  │    • Moving average filter (window: 5 samples)           │   │
│  │    • Spike detection & removal (> 3σ)                    │   │
│  │    • Sensor fusion for redundant measurements            │   │
│  └──────┬──────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 3. TEMPORAL AGGREGATION                                  │   │
│  │    • Raw sampling: Every 10 seconds                      │   │
│  │    • Aggregated to: Hourly averages                      │   │
│  │    • Statistics: min, max, mean, std                     │   │
│  └──────┬──────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 4. LOCAL ALERTING                                        │   │
│  │    • Threshold breach detection                          │   │
│  │    • Immediate alert for critical values:                │   │
│  │      - Turbidity > 100 NTU                               │   │
│  │      - Temperature < 5°C or > 35°C                       │   │
│  │      - Sensor malfunction                                │   │
│  └──────┬──────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 5. DATA BUFFERING                                        │   │
│  │    • Store-and-forward during connectivity loss          │   │
│  │    • Capacity: 7 days of data                            │   │
│  │    • Automatic sync when connection restored             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Validation Rules

```python
# Edge Processing Validation Rules (Pseudocode)

VALIDATION_RULES = {
    'Turbidity': {
        'min': 0,
        'max': 4000,        # NTU
        'rate_of_change': 50  # Max change per hour
    },
    'Conductivity': {
        'min': 0,
        'max': 200000,      # µS/cm
        'rate_of_change': 5000
    },
    'Temperature': {
        'min': -5,
        'max': 50,          # °C
        'rate_of_change': 2
    },
    'NO3': {
        'min': 0,
        'max': 200,         # mg/L
        'rate_of_change': 10
    },
    'Level': {
        'min': 0,
        'max': 15,          # meters
        'rate_of_change': 1
    },
    'Q': {
        'min': 0,
        'max': 5000,        # m³/s
        'rate_of_change': 500
    }
}
```

---

## 4. Network Layer

### 4.1 Communication Protocol Stack

```
┌─────────────────────────────────────────────────────────────┐
│                 NETWORK PROTOCOL STACK                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   APPLICATION    │  MQTT v3.1.1                             │
│   LAYER          │  • Topic-based pub/sub                   │
│                  │  • QoS Level 1 (at least once)           │
│                  │  • Retained messages for latest state    │
│                                                             │
├──────────────────┼──────────────────────────────────────────┤
│                                                             │
│   TRANSPORT      │  TLS 1.3 / TCP                           │
│   LAYER          │  • End-to-end encryption                 │
│                  │  • Certificate-based authentication      │
│                                                             │
├──────────────────┼──────────────────────────────────────────┤
│                                                             │
│   NETWORK        │  IP (IPv4/IPv6)                          │
│   LAYER          │  • Static IP for each station            │
│                  │  • VPN tunnel to cloud                   │
│                                                             │
├──────────────────┼──────────────────────────────────────────┤
│                                                             │
│   PHYSICAL       │  PRIMARY: 4G LTE Cellular                │
│   LAYER          │  BACKUP: LoRaWAN (low power)             │
│                  │  • Automatic failover                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 MQTT Topic Structure

```
water-quality/
├── {station_id}/
│   ├── sensors/
│   │   ├── turbidity          # Real-time turbidity readings
│   │   ├── conductivity       # Real-time conductivity
│   │   ├── temperature        # Real-time temperature
│   │   ├── no3                # Real-time nitrate
│   │   ├── level              # Real-time water level
│   │   └── flow               # Real-time flow rate
│   │
│   ├── aggregated/
│   │   └── hourly             # Hourly aggregated data
│   │
│   ├── alerts/
│   │   ├── critical           # Immediate action required
│   │   ├── warning            # Attention needed
│   │   └── info               # Informational alerts
│   │
│   └── status/
│       ├── health             # Device health metrics
│       ├── battery            # Power status
│       └── connectivity       # Network status
│
└── system/
    ├── commands/              # Remote commands
    └── config/                # Configuration updates
```

### 4.3 Message Format (JSON)

```json
{
  "station_id": "STN-001",
  "timestamp": "2020-03-15T14:30:00+10:00",
  "readings": {
    "turbidity": {
      "value": 12.5,
      "unit": "NTU",
      "quality": "valid"
    },
    "conductivity": {
      "value": 245.3,
      "unit": "µS/cm",
      "quality": "valid"
    },
    "temperature": {
      "value": 24.2,
      "unit": "°C",
      "quality": "valid"
    },
    "no3": {
      "value": 0.85,
      "unit": "mg/L",
      "quality": "valid"
    },
    "level": {
      "value": 2.45,
      "unit": "m",
      "quality": "valid"
    },
    "flow": {
      "value": 45.2,
      "unit": "m³/s",
      "quality": "valid"
    }
  },
  "device_status": {
    "battery_level": 85,
    "signal_strength": -67,
    "uptime_hours": 720
  }
}
```

### 4.4 Network Specifications

| Parameter | Primary (4G LTE) | Backup (LoRaWAN) |
|-----------|------------------|------------------|
| **Bandwidth** | 10-50 Mbps | 0.3-50 kbps |
| **Latency** | 30-50 ms | 1-2 seconds |
| **Range** | Cellular coverage | Up to 15 km |
| **Power** | Moderate | Very Low |
| **Data Volume** | Unlimited | Limited |
| **Use Case** | Normal operation | Emergency/low power |

---

## 5. Cloud Platform & Data Pipeline

### 5.1 Cloud Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CLOUD DATA PLATFORM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   MQTT      │    │   MESSAGE   │    │   STREAM    │                 │
│  │   BROKER    │───▶│   QUEUE     │───▶│  PROCESSING │                 │
│  │  (HiveMQ)   │    │  (Kafka)    │    │  (Spark)    │                 │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                 │
│                                               │                         │
│                          ┌────────────────────┼────────────────────┐   │
│                          │                    │                    │   │
│                          ▼                    ▼                    ▼   │
│                   ┌─────────────┐    ┌─────────────┐    ┌──────────────┐│
│                   │  TIME SERIES│    │    DATA     │    │   ALERT      ││
│                   │   DATABASE  │    │   LAKE      │    │   SERVICE    ││
│                   │ (InfluxDB)  │    │  (S3/CSV)   │    │  (SNS)       ││
│                   └──────┬──────┘    └──────┬──────┘    └──────────────┘│
│                          │                  │                           │
│                          └────────┬─────────┘                          │
│                                   │                                     │
│                                   ▼                                     │
│                   ┌───────────────────────────────────┐                │
│                   │        ML PROCESSING LAYER        │                │
│                   │                                   │                │
│                   │  ┌─────────────┐ ┌─────────────┐ │                │
│                   │  │    LSTM     │ │   RANDOM    │ │                │
│                   │  │  TURBIDITY  │ │   FOREST    │ │                │
│                   │  │ PREDICTION  │ │ CLASSIFIER  │ │                │
│                   │  └─────────────┘ └─────────────┘ │                │
│                   │                                   │                │
│                   └─────────────────┬─────────────────┘                │
│                                     │                                   │
│                                     ▼                                   │
│                   ┌───────────────────────────────────┐                │
│                   │      VISUALIZATION LAYER          │                │
│                   │                                   │                │
│                   │  ┌─────────────┐ ┌─────────────┐ │                │
│                   │  │   TABLEAU   │ │    API      │ │                │
│                   │  │  DASHBOARD  │ │  ENDPOINTS  │ │                │
│                   │  └─────────────┘ └─────────────┘ │                │
│                   │                                   │                │
│                   └───────────────────────────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Storage Strategy

| Data Type | Storage | Retention | Purpose |
|-----------|---------|-----------|---------|
| **Raw Sensor Data** | InfluxDB | 90 days | Real-time queries |
| **Aggregated Hourly** | PostgreSQL | 5 years | Historical analysis |
| **ML Training Data** | AWS S3 (CSV) | Indefinite | Model retraining |
| **Predictions** | PostgreSQL | 1 year | Dashboard display |
| **Alerts/Logs** | Elasticsearch | 1 year | Troubleshooting |

### 5.3 Data Processing Pipeline

```
RAW DATA (10-sec intervals)
         │
         ▼
┌─────────────────────────────────┐
│  INGESTION (Apache Kafka)       │
│  • 11 stations × 6 sensors      │
│  • ~400,000 messages/day        │
│  • Partitioned by station_id    │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  STREAM PROCESSING (Spark)      │
│  • Real-time aggregation        │
│  • Anomaly detection            │
│  • Quality scoring              │
└───────────────┬─────────────────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
┌───────────────┐ ┌───────────────┐
│ HOURLY BATCH  │ │ REAL-TIME     │
│ PROCESSING    │ │ ALERTS        │
│               │ │               │
│ • Aggregation │ │ • Threshold   │
│ • Feature eng │ │   breaches    │
│ • ML inference│ │ • Predictions │
└───────┬───────┘ └───────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  DATA WAREHOUSE                 │
│  • 295,754 hourly records       │
│  • 41 features after eng.       │
│  • 4-year historical data       │
└─────────────────────────────────┘
```

---

## 6. Machine Learning Integration

### 6.1 Model Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ML MODEL DEPLOYMENT                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    MODEL REGISTRY                            │   │
│  │  ┌──────────────────────┐  ┌──────────────────────┐         │   │
│  │  │ lstm_turbidity_model │  │ random_forest_       │         │   │
│  │  │ .keras               │  │ classifier.joblib    │         │   │
│  │  │ Version: 1.0         │  │ Version: 1.0         │         │   │
│  │  │ Accuracy: R²=0.85    │  │ Accuracy: 94.2%      │         │   │
│  │  └──────────────────────┘  └──────────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    INFERENCE PIPELINE                        │   │
│  │                                                              │   │
│  │   Input Data ──▶ Preprocessing ──▶ Model ──▶ Predictions    │   │
│  │                                                              │   │
│  │   • Feature scaling (MinMaxScaler / StandardScaler)         │   │
│  │   • Sequence creation (48-hour lookback for LSTM)           │   │
│  │   • Label encoding for classification                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    OUTPUT ACTIONS                            │   │
│  │                                                              │   │
│  │   • Store predictions in database                           │   │
│  │   • Trigger alerts if Unsafe classification                 │   │
│  │   • Update Tableau dashboard                                │   │
│  │   • Send notifications (email/SMS for critical alerts)      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Model Specifications

#### LSTM Turbidity Prediction Model

| Parameter | Value |
|-----------|-------|
| **Architecture** | 2-layer Stacked LSTM |
| **Layer 1 Units** | 64 (return_sequences=True) |
| **Layer 2 Units** | 32 |
| **Dropout Rate** | 0.2 |
| **Dense Hidden** | 16 units (ReLU) |
| **Output** | 1 unit (Linear) |
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | Mean Squared Error (MSE) |
| **Input Shape** | (48, 3) - 48 hours × 3 features |
| **Output** | Single turbidity value (24h ahead) |

**Input Features:**
1. Turbidity (NTU) - historical values
2. Conductivity (µS/cm)
3. Temperature (°C)

**Training Configuration:**
- Epochs: 100 (with early stopping)
- Batch Size: 32
- Train/Val/Test Split: 70%/15%/15%
- Early Stopping Patience: 10 epochs

#### Random Forest Classification Model

| Parameter | Value |
|-----------|-------|
| **Algorithm** | Random Forest Classifier |
| **Number of Trees** | 100 |
| **Max Depth** | 15 |
| **Min Samples Split** | 5 |
| **Min Samples Leaf** | 2 |
| **Class Weighting** | Balanced |
| **Cross-Validation** | 5-fold |

**Input Features (11 total):**
1. Conductivity (µS/cm)
2. NO3 (mg/L)
3. Temperature (°C)
4. Turbidity (NTU)
5. Level (m)
6. Q - Flow Rate (m³/s)
7. Hour_sin (cyclical encoding)
8. Hour_cos (cyclical encoding)
9. Month_sin (cyclical encoding)
10. Month_cos (cyclical encoding)
11. IsWeekend (binary)

**Output Classes:**
- Safe (66.7% of data)
- Warning (23.2% of data)
- Unsafe (10.1% of data)

### 6.3 Model Retraining Schedule

| Trigger | Frequency | Criteria |
|---------|-----------|----------|
| **Scheduled** | Monthly | Automatic retraining on latest data |
| **Performance Drift** | As needed | When accuracy drops > 5% |
| **Data Distribution Shift** | As needed | Significant change in input distributions |
| **New Data** | Quarterly | After significant new data accumulation |

---

## 7. Performance Metrics & Evaluation

### 7.1 LSTM Model Evaluation

| Metric | Description | Target | Achieved |
|--------|-------------|--------|----------|
| **MSE** | Mean Squared Error | < 50 | Model-dependent |
| **RMSE** | Root Mean Squared Error | < 7 NTU | Model-dependent |
| **MAE** | Mean Absolute Error | < 5 NTU | Model-dependent |
| **R² Score** | Coefficient of Determination | > 0.80 | Model-dependent |

### 7.2 Classification Model Evaluation

| Metric | Description | Target | Achieved |
|--------|-------------|--------|----------|
| **Accuracy** | Overall correct predictions | > 90% | ~94% |
| **Precision** | Positive predictive value | > 85% | ~93% |
| **Recall** | Sensitivity/True Positive Rate | > 85% | ~94% |
| **F1-Score** | Harmonic mean of precision/recall | > 85% | ~93% |
| **Cross-Val Score** | 5-fold cross-validation mean | > 88% | ~92% |

### 7.3 System Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Data Latency** | < 5 minutes | Time from sensor to dashboard |
| **Uptime** | > 99.5% | System availability |
| **Alert Response** | < 1 minute | Critical alert delivery time |
| **Prediction Refresh** | Hourly | Model inference frequency |

---

## 8. Water Quality Thresholds

### 8.1 Classification Thresholds (Australian Guidelines)

| Parameter | Safe | Warning | Unsafe |
|-----------|------|---------|--------|
| **Turbidity** | < 5 NTU | 5-50 NTU | > 50 NTU |
| **Conductivity** | < 30,000 µS/cm | 30,000-50,000 µS/cm | > 50,000 µS/cm |
| **Temperature** | 10-30°C | 5-10°C or 30-35°C | < 5°C or > 35°C |

### 8.2 Alert Thresholds

| Alert Level | Criteria | Action |
|-------------|----------|--------|
| **Critical** | Any parameter Unsafe | Immediate notification, automatic escalation |
| **Warning** | Any parameter in Warning zone | Dashboard highlight, daily digest |
| **Info** | Approaching Warning threshold (within 20%) | Log only |

---

## 9. Security Considerations

| Layer | Security Measure |
|-------|------------------|
| **Sensor/Edge** | Hardware tamper detection, secure boot |
| **Network** | TLS 1.3 encryption, VPN tunnels, certificate auth |
| **Cloud** | IAM roles, encrypted storage (AES-256), audit logs |
| **Application** | API authentication, rate limiting, input validation |

---

## 10. Future Enhancements

1. **Edge AI**: Deploy lightweight ML models on edge devices for faster local predictions
2. **Satellite Backup**: Integration with satellite communication for remote areas
3. **Multi-model Ensemble**: Combine LSTM with other models for improved accuracy
4. **Automated Sensor Calibration**: Self-calibrating sensors with drift detection
5. **Mobile Application**: Real-time alerts and monitoring via smartphone app

---

*Document Version: 1.0*  
*Last Updated: January 2026*  
*Author: Smart Water Quality Monitoring Project Team*
