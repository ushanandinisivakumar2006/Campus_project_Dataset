import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = os.path.dirname(os.path.abspath(__file__))
df_e = pd.read_csv(os.path.join(DATA, 'energy_1year.csv'))
df_o = pd.read_csv(os.path.join(DATA, 'occupancy_1year.csv'))
df = pd.merge(df_e, df_o, on=['timestamp', 'building_id'])
df['dt'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['dt'].dt.hour
df['sin_h'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_h'] = np.cos(2 * np.pi * df['hour'] / 24)
df = df.sort_values(['building_id', 'dt']).reset_index(drop=True)

BUILDINGS = {
    'college_campus_main': 'Administration Block',
    'college_campus_mother_teresa': 'Science Block',
    'college_campus_mba_lib': 'Library',
    'college_campus_new_building': 'CSE / Auditorium Wing',
    'college_campus_canteen': 'Hostel / Dining Hall',
    'college_campus_pt_room': 'Sports Complex',
}


def mape_excl_zero(y_true, y_pred, thr=0.1):
    mask = y_true > thr
    if mask.sum() == 0:
        return float('nan')
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def calc(y_true, y_pred):
    return {
        'R2': round(float(r2_score(y_true, y_pred)), 4),
        'MAE': round(float(mean_absolute_error(y_true, y_pred)), 4),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        'MAPE': round(float(mape_excl_zero(y_true, y_pred)), 2),
    }


BLOCKS = {
    'Sep-2025 Odd Semester': ('2025-09-01', '2025-09-14'),
    'Nov-2025 Exam Period': ('2025-11-15', '2025-11-28'),
    'May-2026 Hot Spell': ('2026-05-10', '2026-05-23'),
    'Jun-2026 Summer Holiday': ('2026-06-17', '2026-06-30'),
}

for block_name, (start_str, end_str) in BLOCKS.items():
    print('')
    print('=' * 70)
    print('BLOCK: ' + block_name + '  (' + start_str + ' to ' + end_str + ')')
    start_ts = pd.Timestamp(start_str)
    end_ts = pd.Timestamp(end_str) + pd.Timedelta(hours=23)
    all_r2 = []
    all_mae = []
    all_rmse = []
    all_mape = []

    for bid, bname in BUILDINGS.items():
        b = df[df['building_id'] == bid].sort_values('dt').reset_index(drop=True)
        train = b[b['dt'] < start_ts]
        test = b[(b['dt'] >= start_ts) & (b['dt'] <= end_ts)]
        if len(train) < 48 or len(test) < 24:
            print('  ' + bname + ': insufficient data (train=' + str(len(train)) + ', test=' + str(len(test)) + ')')
            continue
        y_tr = train['power_kw'].values
        y_te = test['power_kw'].values
        X_tr = np.c_[train['headcount'].values, train['sin_h'].values, train['cos_h'].values]
        X_te = np.c_[test['headcount'].values, test['sin_h'].values, test['cos_h'].values]
        reg = LinearRegression().fit(X_tr, y_tr)
        p = reg.predict(X_te)
        r = calc(y_te, p)
        n = len(y_te)
        line = ('  ' + bname.ljust(28)
                + '  R2=' + str(r['R2']).rjust(7)
                + '  MAE=' + str(r['MAE']).rjust(6)
                + '  RMSE=' + str(r['RMSE']).rjust(6)
                + '  MAPE=' + str(r['MAPE']).rjust(6) + '%'
                + '  (n=' + str(n) + 'h)')
        print(line)
        all_r2.append(r['R2'])
        all_mae.append(r['MAE'])
        all_rmse.append(r['RMSE'])
        all_mape.append(r['MAPE'])

    if all_r2:
        line = ('  ' + 'Six-building Mean'.ljust(28)
                + '  R2=' + str(round(float(np.mean(all_r2)), 4)).rjust(7)
                + '  MAE=' + str(round(float(np.mean(all_mae)), 4)).rjust(6)
                + '  RMSE=' + str(round(float(np.mean(all_rmse)), 4)).rjust(6)
                + '  MAPE=' + str(round(float(np.mean(all_mape)), 2)).rjust(6) + '%')
        print(line)

print('')
print('All blocks use Eq.(16): headcount + sin(2*pi*h/24) + cos(2*pi*h/24).')
print('Each block trains on all data before its start date.')
print('MAPE excludes hours with P_meas < 0.10 kW.')
