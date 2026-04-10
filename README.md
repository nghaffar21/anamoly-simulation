# 🔥 Indoor Fire Anomaly Detection Dataset

> Temperature & humidity sensor data with injected fire anomalies — for ML anomaly detection benchmarking.

---

## Overview

This repository provides a labeled dataset and injection script for ML engineers building or evaluating anomaly detection models on indoor fire events. It includes ~4 days and 3 hours of real office temperature and humidity readings collected by a DHT22 sensor, with synthetic fire anomalies applied to the validation and test splits.

Anomaly graphs are available in the `Graphs/` directory. You can also download the script, adjust parameters, and define custom anomalies of your own.

---

## Raw data — `office_sensor_data.csv`

Original office sensor readings spanning approximately 4 days and 3 hours. Data points are sampled roughly every 2 seconds, though some intervals exceed 2 seconds due to occasional sensor capture failures.

---

## Dataset splits

The data is divided into training, validation, and test sets at a **70 / 15 / 15** ratio. Fire anomalies are injected into the validation and test sets only.

| File | Split | Contains anomalies? |
|---|---|---|
| `training_data.csv` | 70% | No |
| `validation_data.csv` | 15% | Yes |
| `test_data.csv` | 15% | Yes |

---

## Injected anomalies

### Validation set (3 events)

| Distance | Duration (pts) | Peak temp increase (°C) | Humidity drop (%) |
|---|---|---|---|
| far | 1503 | 6.69 | 3.25 |
| mid | 997 | 21.57 | 10.72 |
| mid | 1156 | 18.64 | 6.60 |

### Test set (4 events)

| Distance | Duration (pts) | Peak temp increase (°C) | Humidity drop (%) |
|---|---|---|---|
| mid | 824 | 11.75 | 10.27 |
| mid | 1026 | 10.44 | 9.04 |
| far | 942 | 8.37 | 1.42 |
| mid | 1010 | 19.78 | 8.20 |

---

## Field definitions

**Duration** — measured in number of data points. Approximately 660 data points equals one hour.

**Distance** — proximity of the sensor to the fire source: `close`, `mid`, or `far`. This determines the magnitude of the temperature increase and humidity drop applied to the data.

---

## Running the script

1. Place `Anomaly_Injection_Office_Data.py` and `office_sensor_data.csv` in the same directory.
2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate <env-name>
   ```
3. Run the script:
   ```bash
   python3 Anomaly_Injection_Office_Data.py
   ```
4. Inspect `validation_data.csv` and `test_data.csv` to review the injected anomalies.
