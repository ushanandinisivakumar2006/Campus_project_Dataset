# Campus Digital Twin: 1-Year Synchronized Multi-Domain Dataset

This repository contains the complete 1-year multi-domain synchronized dataset (8,760 hours, 1 July 2025 to 30 June 2026) for the Campus Digital Twin deployed across six core facilities.

The dataset integrates physical telemetry from RFID occupant portals, three-phase power submeters, 350 kWp rooftop solar PV string inverters, and authentic local meteorology (ERA5/IMD) coupled via calibrated thermodynamic and occupancy transfer functions.

---

## 🏛️ Campus Facility Registry

| Building ID | Facility Name | Signage / HUD Label | Design Area (m²) | Design Capacity | Primary Disciplines |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `college_campus_main` | **Administration Block** | `ADMINISTRATION BLOCK` | 18,500 | 1,500 | Executive Admin, Principal Suites, Computer Science |
| `college_campus_mother_teresa` | **Science Block** | `SCIENCE BLOCK` | 14,200 | 280 | Electronics & Communication Engineering (ECE), Labs |
| `college_campus_auditorium` / `college_campus_new_building` | **CSE / Auditorium Wing** | `CSE / AUDITORIUM` | 8,800 | 110 | 1,200-seat Auditorium, Electrical & Electronics (EEE) |
| `college_campus_mba_lib` | **Library** | `LIBRARY` | 9,800 | 280 | Management Studies (MBA), Central Knowledge Center |
| `college_campus_canteen` | **Hostel / Dining Hall** | `HOSTEL / DINING` | 2,400 | 300 | Student Residential Dining Pavilion, Food Services |
| `college_campus_pt_room` | **Sports Complex** | `SPORTS COMPLEX` | 1,200 | 170 | Physical Training Gymnasium, Indoor Athletics Arena |

---

## 📁 Repository Structure & Manifest

| File | Size | Description |
| :--- | :---: | :--- |
| `energy_1year.csv` | ~4.2 MB | Hourly electrical active power (kW), cumulative energy (kWh), rooftop solar generation, net grid import, and disaggregated loads (HVAC, lighting, plug loads) for 6 facilities (8,760 hours × 6 = 52,560 rows). |
| `occupancy_1year.csv` | ~3.5 MB | Hourly occupant headcounts, design seating capacities, space utilization percentages, student counts, and RFID card transit frequencies (52,560 rows). |
| `weatherdatasameifet.csv` | ~468 KB | Full-year hourly meteorological reanalysis data (ERA5/IMD dry-bulb temperature, relative humidity, cloud cover, solar irradiance, wind speed, pressure). |
| `buildings.json` | ~3.6 KB | Geometric areas, capacities, baseline parasitic loads ($P_{\\text{base}}$), and empirical occupant load sensitivities ($\\alpha$). |
| `infrastructure.json` | ~3.3 KB | Feeder ratings, AHU equipment counts, substation capacities, and live indoor climate averages. |
| `rooms.json` | ~21.6 KB | Spatial room configurations, 3D zoning, and indoor environmental sensor parameters (CO₂, VOC, PM2.5, illuminance). |
| `timetable.json` | ~5.6 KB | Academic calendar, semester cycles, instructional blocks, examination cycles, and vacation periods. |
| `weather.json` | ~260 B | Campus geographical coordinates (11.9156° N, 79.5067° E) and climate reference summary. |
| `generate_1year_dataset.py` | ~7.8 KB | Standalone Python generator script reproducing the multi-domain time series with seed 42. |
| `scenario_from_data.py` | ~9.6 KB | Meta-Twin operational dispatch simulation script (baseline, heatwave, exam surge, optimized consolidation). |
| `multi_block_eval.py` | ~3.9 KB | Seasonal multi-block cross-validation script evaluating predictive models across 4 seasonal 14-day test blocks. |

---

## 📊 Dataset Schema

### 1. `energy_1year.csv`
* `timestamp`: UTC/Local timestamp (`YYYY-MM-DD HH:MM:SS`)
* `building_id`: Facility identifier (`college_campus_main`, `college_campus_mother_teresa`, etc.)
* `power_kw`: Total active electrical demand (kW)
* `energy_kwh`: Hourly cumulative energy consumption (kWh)
* `solar_generation_kw`: Apportioned solar generation from the 350 kWp array (kW)
* `net_demand_kw`: Net active grid demand, $\\max(0, \\text{power\\_kw} - \\text{solar\\_generation\\_kw})$ (kW)
* `hvac_kw`: Disaggregated cooling and thermal ventilation power (kW)
* `lighting_kw`: Disaggregated interior and architectural lighting power (kW)
* `lab_equipment_kw`: Disaggregated laboratory equipment, computing, and plug loads (kW)

### 2. `occupancy_1year.csv`
* `timestamp`: UTC/Local timestamp (`YYYY-MM-DD HH:MM:SS`)
* `building_id`: Facility identifier
* `headcount`: Concurrent occupant headcount (persons)
* `capacity`: Registered spatial seating capacity (persons)
* `utilization_pct`: Capacity utilization percentage (%)
* `student_headcount`: Student portion of active occupants
* `rfid_swipes`: Portal transit RFID card swipe frequency per hour
* `occupancy_rate_pct`: Percentage occupancy rate (%)

---

## 🚀 Getting Started & Reproduction

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### 1. Run Seasonal Cross-Validation Evaluation
To compute out-of-sample prediction metrics ($R^2$, MAE, RMSE, MAPE) across the four 14-day seasonal test blocks:
```bash
python multi_block_eval.py
```

### 2. Run Meta-Twin Operational Scenarios
To run what-if simulation scenarios (baseline, heatwave, examination surge, and Meta-Twin-guided operation):
```bash
python scenario_from_data.py
```

### 3. Regenerate Datasets from Raw Weather (Optional)
To regenerate the full 1-year time series from the meteorological records:
```bash
python generate_1year_dataset.py
```

---

## 📜 License
This dataset and associated code are licensed under the **Creative Commons Attribution 4.0 International (CC-BY-4.0)** License.
