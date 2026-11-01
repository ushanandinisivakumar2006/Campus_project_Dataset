"""
generate_1year_dataset.py
Generates a full 1-year (8,760 hours, 2025-07-01 to 2026-06-30) synchronized multi-domain dataset
for the Campus Digital Twin.
Combines:
1. Authentic local meteorology from weatherdatasameifet.csv (Villupuram coords: 11.9156° N, 79.5067° E).
2. Semester academic timetables (odd semester Jul-Nov 2025, even semester Jan-May 2026, exam periods, vacation).
3. 6 campus facilities with CAD floor areas (Total: 54,900 m²).
4. Physical 350 kWp solar PV generation model derated by ambient temperature and cloud cover.
5. Physics-based thermodynamic load coupling (HVAC temperature dependency + occupant plug loads).
"""

import os
import math
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR
WEATHER_FILE = os.path.join(CURRENT_DIR, "weatherdatasameifet.csv")

# 1. Parse weather data
print("Loading 1-year weather data...")
lines = open(WEATHER_FILE, 'r', encoding='utf-8', errors='ignore').readlines()
data_lines = [l for l in lines if l.startswith('2025-') or l.startswith('2026-')]
cols = ['time','temp','rh','dew_point','precip','cloud_cover','wind_speed','wind_dir','pressure']
df_w = pd.DataFrame([l.strip().split(',') for l in data_lines], columns=cols)
for c in cols[1:]:
    df_w[c] = pd.to_numeric(df_w[c])
df_w['time'] = pd.to_datetime(df_w['time'])
df_w = df_w.sort_values('time').reset_index(drop=True)
print(f"Weather loaded: {len(df_w)} hours from {df_w['time'].min()} to {df_w['time'].max()}")

# 2. Buildings definition
buildings = [
    {"id": "college_campus_main", "name": "Administration Block", "area": 18500, "capacity": 270, "base_kw": 8.71, "alpha": 0.1063},
    {"id": "college_campus_mother_teresa", "name": "Science Block", "area": 14200, "capacity": 315, "base_kw": 7.66, "alpha": 0.0759},
    {"id": "college_campus_mba_lib", "name": "Library", "area": 9800, "capacity": 195, "base_kw": 5.67, "alpha": 0.0921},
    {"id": "college_campus_new_building", "name": "CSE / Auditorium Wing", "area": 8800, "capacity": 240, "base_kw": 6.86, "alpha": 0.0943},
    {"id": "college_campus_canteen", "name": "Hostel / Dining Hall", "area": 2400, "capacity": 142, "base_kw": 3.79, "alpha": 0.0865},
    {"id": "college_campus_pt_room", "name": "Sports Complex", "area": 1200, "capacity": 52, "base_kw": 1.43, "alpha": 0.0871}
]

np.random.seed(42)

energy_rows = []
occ_rows = []

# Academic Calendar determination
def get_academic_day_type(dt):
    month = dt.month
    day = dt.day
    weekday = dt.weekday() # 5=Sat, 6=Sun
    
    # Sundays are always holidays
    if weekday == 6:
        return "Sunday"
    
    # Summer vacation: June
    if month == 6:
        return "Summer_Vacation"
    # Winter break: Dec 20 to Jan 5
    if (month == 12 and day >= 20) or (month == 1 and day <= 5):
        return "Winter_Break"
    
    # Exam periods: Nov 15 - Dec 10 and May 5 - May 30
    if (month == 11 and day >= 15) or (month == 12 and day <= 10):
        return "Exam_Period"
    if (month == 5 and day >= 5 and day <= 30):
        return "Exam_Period"
    
    # Saturdays: Alternate instructional
    if weekday == 5:
        return "Instructional_Saturday" if (day % 2 == 1) else "Off_Saturday"
        
    return "Instructional_Weekday"

print("Generating 1-year multi-domain time series...")
for _, w_row in df_w.iterrows():
    dt = w_row['time']
    hour = dt.hour
    temp = w_row['temp']
    cloud = w_row['cloud_cover']
    day_type = get_academic_day_type(dt)
    
    # 350 kWp Solar PV Calculation
    # Peak irradiance ~ 1000 W/m2 derated by solar angle and cloud cover
    if 6 <= hour <= 18:
        solar_elevation = math.sin(math.pi * (hour - 6) / 12)
        cloud_factor = 1.0 - (cloud / 100.0) * 0.70
        temp_derating = 1.0 - 0.0041 * max(0, temp - 25.0)
        # 350 kWp nameplate with ~184.6 kW max clear-sky generation
        solar_kw = max(0.0, 184.6 * solar_elevation * cloud_factor * temp_derating + np.random.normal(0, 1.5))
    else:
        solar_kw = 0.0
    solar_kw = round(solar_kw, 2)
    
    for b in buildings:
        bid = b["id"]
        cap = b["capacity"]
        base_kw = b["base_kw"]
        alpha = b["alpha"]
        
        # 1. Occupancy generation based on schedule and day type
        if day_type == "Sunday" or day_type == "Summer_Vacation":
            occ = int(np.random.uniform(0, 0.05 * cap))
        elif day_type == "Off_Saturday" or day_type == "Winter_Break":
            occ = int(np.random.uniform(0, 0.10 * cap))
        elif day_type == "Exam_Period":
            if 8 <= hour <= 18:
                occ = int(cap * np.random.uniform(0.75, 0.95))
            elif 19 <= hour <= 21:
                occ = int(cap * 0.40) if "lib" in bid or "main" in bid else 10
            else:
                occ = int(np.random.uniform(2, 8))
        else: # Instructional Weekday / Saturday
            if 8 <= hour <= 16:
                # Bell curve centered around 11:30
                time_factor = math.sin(math.pi * (hour - 8) / 9)
                occ = int(cap * (0.65 + 0.25 * time_factor) + np.random.randint(-10, 10))
                occ = max(15, min(cap, occ))
            elif 17 <= hour <= 19:
                occ = int(cap * 0.20 + np.random.randint(-5, 5))
            else:
                occ = int(np.random.uniform(2, 6))
                
        # Special canteen profile: peaks at lunch (12:00-14:00)
        if bid == "college_campus_canteen":
            if 12 <= hour <= 14 and day_type in ["Instructional_Weekday", "Exam_Period"]:
                occ = int(cap * np.random.uniform(0.80, 0.95))
            elif 7 <= hour <= 9:
                occ = int(cap * np.random.uniform(0.30, 0.50))
            else:
                occ = max(0, int(occ * 0.25))

        occ = max(0, min(int(cap * 1.1), occ))
        
        # 2. Power calculation with thermal sensitivity kappa = 0.042
        eta_hvac = 1.0 + 0.042 * max(0.0, temp - 24.0)
        # Power = (Base + alpha * occ) * eta_hvac + noise
        noise = np.random.normal(0, 0.4)
        p_kw = round((base_kw + alpha * occ) * (0.85 + 0.15 * eta_hvac) + noise, 2)
        p_kw = max(round(base_kw * 0.7, 2), p_kw)
        
        # Subsystems
        hvac_kw = round(p_kw * 0.45 * (eta_hvac / 1.1), 2)
        lighting_kw = round(p_kw * 0.25, 2)
        lab_kw = round(max(0.0, p_kw - hvac_kw - lighting_kw), 2)
        
        energy_kwh = round(p_kw * 1.0, 2)
        net_demand_kw = round(max(0.0, p_kw - (solar_kw / 6.0)), 2) # Allocated solar
        
        energy_rows.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "building_id": bid,
            "power_kw": p_kw,
            "energy_kwh": energy_kwh,
            "solar_generation_kw": round(solar_kw / 6.0, 2),
            "net_demand_kw": net_demand_kw,
            "hvac_kw": hvac_kw,
            "lighting_kw": lighting_kw,
            "lab_equipment_kw": lab_kw
        })
        
        rfid_swipes = int(occ * np.random.uniform(0.9, 1.1))
        occ_rows.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "building_id": bid,
            "headcount": occ,
            "capacity": cap,
            "utilization_pct": round(occ / cap * 100.0, 1),
            "student_headcount": occ,
            "rfid_swipes": rfid_swipes,
            "occupancy_rate_pct": round(occ / cap * 100.0, 1)
        })

df_e_out = pd.DataFrame(energy_rows)
df_o_out = pd.DataFrame(occ_rows)

print(f"Saving 1-year datasets (Total records: {len(df_e_out)} per file)...")
df_e_out.to_csv(os.path.join(DATA_DIR, "energy_1year.csv"), index=False)
df_o_out.to_csv(os.path.join(DATA_DIR, "occupancy_1year.csv"), index=False)
print("1-Year dataset generation complete!")
