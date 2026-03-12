"""
NASA C-MAPSS Turbofan Engine Degradation - Synthetic Data Generator
Simulates 21 sensors + 3 operating conditions for 100 engines
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_engine_data(engine_id, max_cycles=None):
    if max_cycles is None:
        max_cycles = np.random.randint(150, 350)

    cycles = np.arange(1, max_cycles + 1)
    n = len(cycles)

    # 3 operating conditions (discrete settings)
    op_cond = np.random.choice([0, 1, 2], size=n)
    op1 = op_cond * 10 + np.random.normal(0, 0.5, n)
    op2 = op_cond * 5 + np.random.normal(0, 0.3, n)
    op3 = [100 if c == 0 else 90 if c == 1 else 60 for c in op_cond] + np.random.normal(0, 1, n)

    # Degradation signal (nonlinear, accelerates near end of life)
    t = cycles / max_cycles
    degradation = 0.2 * t + 0.8 * t**3 + np.random.normal(0, 0.01, n)

    # 21 sensors with different degradation behaviors
    sensors = {}

    # Sensors that INCREASE with degradation
    sensors['s1']  = 489.05 + 0.5 * degradation * 10  + np.random.normal(0, 0.5, n)
    sensors['s2']  = 604.10 + degradation * 15         + np.random.normal(0, 0.4, n)
    sensors['s3']  = 1590.0 - degradation * 20         + np.random.normal(0, 2.0, n)
    sensors['s4']  = 1400.0 + degradation * 30         + np.random.normal(0, 3.0, n)
    sensors['s5']  = 14.62  - degradation * 0.1        + np.random.normal(0, 0.1, n)
    sensors['s6']  = 21.61  + degradation * 0.2        + np.random.normal(0, 0.2, n)
    sensors['s7']  = 554.0  - degradation * 8          + np.random.normal(0, 1.0, n)
    sensors['s8']  = 2388.0 - degradation * 15         + np.random.normal(0, 5.0, n)
    sensors['s9']  = 9065.0 - degradation * 50         + np.random.normal(0, 20,  n)
    sensors['s10'] = 1.30   + degradation * 0.05       + np.random.normal(0, 0.01, n)
    sensors['s11'] = 47.47  + degradation * 1.5        + np.random.normal(0, 0.3, n)
    sensors['s12'] = 522.0  - degradation * 10         + np.random.normal(0, 1.5, n)
    sensors['s13'] = 2388.0 - degradation * 12         + np.random.normal(0, 5.0, n)
    sensors['s14'] = 8138.0 - degradation * 40         + np.random.normal(0, 15,  n)
    sensors['s15'] = 8.4195 + degradation * 0.2        + np.random.normal(0, 0.05, n)
    sensors['s16'] = 0.03   + degradation * 0.005      + np.random.normal(0, 0.001, n)
    sensors['s17'] = 391.0  + degradation * 5          + np.random.normal(0, 2.0, n)
    sensors['s18'] = 2388.0 - degradation * 10         + np.random.normal(0, 5.0, n)
    sensors['s19'] = 100.0  + np.random.normal(0, 0.5, n)          # stable
    sensors['s20'] = 38.86  + degradation * 0.5        + np.random.normal(0, 0.1, n)
    sensors['s21'] = 23.419 - degradation * 0.3        + np.random.normal(0, 0.1, n)

    # RUL = remaining cycles until failure
    rul = max_cycles - cycles

    df = pd.DataFrame({
        'engine_id': engine_id,
        'cycle': cycles,
        'op_setting_1': op1,
        'op_setting_2': op2,
        'op_setting_3': op3,
        **sensors,
        'RUL': rul
    })
    return df


def generate_dataset(n_engines=100):
    dfs = [generate_engine_data(i+1) for i in range(n_engines)]
    return pd.concat(dfs, ignore_index=True)


if __name__ == '__main__':
    print("Generating training dataset (100 engines)...")
    train_df = generate_dataset(100)
    train_df.to_csv('/home/claude/predictive_maintenance/data/train_data.csv', index=False)
    print(f"  Train: {len(train_df):,} rows × {train_df.shape[1]} cols")

    print("Generating test dataset (20 engines)...")
    test_df = generate_dataset(20)
    test_df.to_csv('/home/claude/predictive_maintenance/data/test_data.csv', index=False)
    print(f"  Test:  {len(test_df):,} rows × {test_df.shape[1]} cols")
    print("Done. Data saved to data/")
