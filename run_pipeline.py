"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   AI-DRIVEN PREDICTIVE MAINTENANCE SYSTEM                                   ║
║   NASA Turbofan Engine Degradation Analysis                                 ║
║   Pipeline: Preprocessing → EDA → Features → ML Models → Optimization      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import scipy.stats as stats
import os, json, time

# ─── Styling ────────────────────────────────────────────────────────────────
DARK_BG   = '#0D1117'
PANEL_BG  = '#161B22'
ACCENT1   = '#58A6FF'   # blue
ACCENT2   = '#3FB950'   # green
ACCENT3   = '#F78166'   # red/orange
ACCENT4   = '#D2A8FF'   # purple
ACCENT5   = '#FFA657'   # orange
GOLD      = '#E3B341'
TEXT_CLR  = '#E6EDF3'
GRID_CLR  = '#21262D'

SENSOR_COLS = [f's{i}' for i in range(1, 22)]
OP_COLS     = ['op_setting_1', 'op_setting_2', 'op_setting_3']
BASE_PATH   = '/home/claude/predictive_maintenance'
FIG_PATH    = f'{BASE_PATH}/figures'
RES_PATH    = f'{BASE_PATH}/results'

def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  DARK_BG,
        'axes.facecolor':    PANEL_BG,
        'axes.edgecolor':    '#30363D',
        'axes.labelcolor':   TEXT_CLR,
        'xtick.color':       TEXT_CLR,
        'ytick.color':       TEXT_CLR,
        'text.color':        TEXT_CLR,
        'grid.color':        GRID_CLR,
        'grid.linewidth':    0.5,
        'font.family':       'DejaVu Sans',
        'axes.titlesize':    13,
        'axes.labelsize':    11,
        'legend.facecolor':  PANEL_BG,
        'legend.edgecolor':  '#30363D',
    })

set_dark_style()

print("="*70)
print("  AI-DRIVEN PREDICTIVE MAINTENANCE SYSTEM")
print("  NASA Turbofan Engine Degradation Analysis")
print("="*70)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading & Preprocessing Data...")

train = pd.read_csv(f'{BASE_PATH}/data/train_data.csv')
test  = pd.read_csv(f'{BASE_PATH}/data/test_data.csv')

print(f"  Train shape: {train.shape}")
print(f"  Test  shape: {test.shape}")
print(f"  Engines (train): {train['engine_id'].nunique()}")
print(f"  Missing values: {train.isnull().sum().sum()}")

# Remove near-constant sensors (std < threshold)
sensor_std = train[SENSOR_COLS].std()
low_var_sensors = sensor_std[sensor_std < 0.005].index.tolist()
print(f"  Low-variance sensors removed: {low_var_sensors if low_var_sensors else 'None'}")
useful_sensors = [s for s in SENSOR_COLS if s not in low_var_sensors]
print(f"  Useful sensors: {len(useful_sensors)}/21")

# ── Cap RUL at 125 (piece-wise linear target — common in literature) ─────────
RUL_CAP = 125
train['RUL_capped'] = train['RUL'].clip(upper=RUL_CAP)
test['RUL_capped']  = test['RUL'].clip(upper=RUL_CAP)

# ── Scale sensors per engine (MinMax) ────────────────────────────────────────
scaler = MinMaxScaler()
train[useful_sensors] = scaler.fit_transform(train[useful_sensors])
test[useful_sensors]  = scaler.transform(test[useful_sensors])

print("  Normalization: complete (MinMaxScaler)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Feature Engineering...")

def engineer_features(df, sensors, window=10):
    df = df.copy().sort_values(['engine_id', 'cycle'])
    grp = df.groupby('engine_id')

    for s in sensors:
        # Rolling mean and std
        df[f'{s}_rmean'] = grp[s].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'{s}_rstd']  = grp[s].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))

    # Cycle-based health proxy (normalized cycle fraction)
    max_cycle = grp['cycle'].transform('max')
    df['cycle_norm'] = df['cycle'] / max_cycle

    # Cumulative degradation: mean of all sensor rolling means
    rmean_cols = [f'{s}_rmean' for s in sensors]
    df['health_index'] = 1 - df[rmean_cols].mean(axis=1)   # proxy: higher = healthier

    return df

train_fe = engineer_features(train, useful_sensors)
test_fe  = engineer_features(test,  useful_sensors)

feature_cols = (useful_sensors
              + [f'{s}_rmean' for s in useful_sensors]
              + [f'{s}_rstd'  for s in useful_sensors]
              + OP_COLS
              + ['cycle_norm', 'health_index'])

print(f"  Original features: {len(useful_sensors)}")
print(f"  Engineered features: {len(feature_cols)} total")

X_train = train_fe[feature_cols].fillna(0)
y_train = train_fe['RUL_capped']
X_test  = test_fe[feature_cols].fillna(0)
y_test  = test_fe['RUL_capped']


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: TRAIN ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 3] Training ML Models...")

models = {
    'Linear Regression':     LinearRegression(),
    'Ridge Regression':      Ridge(alpha=10),
    'Random Forest':         RandomForestRegressor(n_estimators=200, max_depth=12,
                                                   min_samples_leaf=5, n_jobs=-1, random_state=42),
    'Gradient Boosting\n(XGBoost-equiv)': GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                                   learning_rate=0.05, subsample=0.8,
                                                   random_state=42),
}

results = {}
predictions = {}

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, RUL_CAP)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'Train Time (s)': round(elapsed,2)}
    predictions[name] = y_pred
    print(f"  ✓ {name.replace(chr(10),' '):<30}  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}  [{elapsed:.1f}s]")

results_df = pd.DataFrame(results).T
results_df.to_csv(f'{RES_PATH}/model_comparison.csv')
print(f"\n  Best model by RMSE: {results_df['RMSE'].idxmin()}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: MAINTENANCE OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 4] Maintenance Optimization...")

best_model_name = results_df['RMSE'].idxmin()
best_model = models[best_model_name]
best_preds = predictions[best_model_name]

# Get last known RUL prediction per engine in test set
test_fe['pred_RUL'] = best_preds
engine_status = (test_fe.sort_values('cycle')
                 .groupby('engine_id')
                 .last()[['cycle', 'RUL', 'pred_RUL']]
                 .reset_index())
engine_status['engine_id'] = engine_status['engine_id'].astype(int)

# ── Simple optimization: schedule maintenance at right window ────────────────
MAINTENANCE_COST   = 5000   # $ per planned maintenance
EMERGENCY_COST     = 25000  # $ per unplanned failure
DOWNTIME_COST_DAY  = 3000   # $ per day of downtime
SAFETY_MARGIN      = 15     # cycles buffer before predicted failure

def optimize_schedule(engine_df, safety_margin=15):
    schedule = []
    total_planned = 0
    total_emergency = 0

    for _, row in engine_df.iterrows():
        rul = row['pred_RUL']
        eid = row['engine_id']
        current_cycle = row['cycle']
        actual_rul = row['RUL']

        if rul <= safety_margin:
            action = 'IMMEDIATE'
            cost = MAINTENANCE_COST
            urgency = 'CRITICAL'
            total_planned += cost
        elif rul <= safety_margin * 3:
            action = f'Schedule in {int(rul - safety_margin)} cycles'
            cost = MAINTENANCE_COST
            urgency = 'HIGH'
            total_planned += cost
        elif rul <= safety_margin * 6:
            action = f'Schedule in {int(rul - safety_margin)} cycles'
            cost = MAINTENANCE_COST * 0.8   # discounted (plan ahead)
            urgency = 'MEDIUM'
            total_planned += cost
        else:
            action = 'Monitor — healthy'
            cost = 0
            urgency = 'LOW'

        # Check if we'd miss it (emergency)
        if actual_rul <= 0 and urgency == 'LOW':
            cost += EMERGENCY_COST
            total_emergency += EMERGENCY_COST
            action = 'FAILURE — emergency repair'

        schedule.append({
            'Engine': eid,
            'Current Cycle': int(current_cycle),
            'Predicted RUL': round(float(rul), 1),
            'Actual RUL': int(actual_rul),
            'Urgency': urgency,
            'Action': action,
            'Estimated Cost ($)': int(cost)
        })

    total_cost = total_planned + total_emergency
    savings = total_emergency  # avoided by planning
    return pd.DataFrame(schedule), total_cost, savings

schedule_df, total_cost, savings = optimize_schedule(engine_status)
schedule_df.to_csv(f'{RES_PATH}/maintenance_schedule.csv', index=False)

print(f"  Engines scheduled: {len(schedule_df)}")
print(f"  CRITICAL/immediate: {(schedule_df['Urgency']=='CRITICAL').sum()}")
print(f"  HIGH priority:      {(schedule_df['Urgency']=='HIGH').sum()}")
print(f"  MEDIUM priority:    {(schedule_df['Urgency']=='MEDIUM').sum()}")
print(f"  Total estimated cost: ${total_cost:,.0f}")
print(f"  Savings vs reactive: ${savings:,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: VISUALIZATIONS (8 Professional Plots)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Generating Visualizations...")

# ─── Figure 1: System Architecture Diagram ───────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16); ax.set_ylim(0, 7)
ax.axis('off')
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

ax.text(8, 6.5, 'AI-Driven Predictive Maintenance System — Architecture',
        ha='center', va='center', fontsize=16, fontweight='bold', color=TEXT_CLR)

boxes = [
    (1.2, 3.5, '🔧 Sensor Data\n21 Sensors\n3 Op Conditions',    ACCENT1),
    (3.5, 3.5, '⚙️  Preprocessing\nNormalize\nClean',             ACCENT4),
    (5.8, 3.5, '🔬 Feature Eng.\nRolling Stats\nHealth Index',     ACCENT2),
    (8.1, 3.5, '🤖 ML Models\nLinReg / RF\nGrad. Boost',          ACCENT5),
    (10.4,3.5, '📊 Failure\nPrediction\nRUL Output',              ACCENT3),
    (12.7,3.5, '🏭 Maintenance\nOptimization\nSchedule',          GOLD),
]

for (x, y, label, color) in boxes:
    bbox = FancyBboxPatch((x-0.9, y-1), 1.8, 2, boxstyle="round,pad=0.1",
                          facecolor=color+'33', edgecolor=color, linewidth=2)
    ax.add_patch(bbox)
    ax.text(x, y, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color=color)

# Arrows
for i in range(len(boxes)-1):
    x1 = boxes[i][0] + 0.9
    x2 = boxes[i+1][0] - 0.9
    y  = boxes[i][1]
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='#58A6FF', lw=2))

# Data flow label
ax.text(8, 1.8, 'Data Flow Pipeline', ha='center', color='#8B949E', fontsize=10)
ax.annotate('', xy=(13.6, 2.5), xytext=(1.2, 2.5),
            arrowprops=dict(arrowstyle='->', color='#30363D', lw=1.5,
                            connectionstyle='arc3,rad=0'))

# Metrics box
metrics_text = f"Best Model: {best_model_name.replace(chr(10),' ')}\nRMSE: {results_df.loc[best_model_name,'RMSE']:.2f}  |  R²: {results_df.loc[best_model_name,'R²']:.4f}"
ax.text(8, 0.8, metrics_text, ha='center', va='center', fontsize=10,
        color=ACCENT2,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=ACCENT2+'22', edgecolor=ACCENT2))

plt.tight_layout()
plt.savefig(f'{FIG_PATH}/01_system_architecture.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Plot 1: System Architecture")


# ─── Figure 2: Sensor Trends + Degradation Curves ────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(DARK_BG)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

sample_engines = [1, 5, 10, 20]
colors_eng = [ACCENT1, ACCENT2, ACCENT3, ACCENT4]

# Panel 1-3: Sensor trends for 3 key sensors
key_sensors = ['s4', 's9', 's11']
sensor_labels = ['Turbine Temperature (s4)', 'Fan Speed (s9)', 'HPC Pressure (s11)']

for i, (s, lbl) in enumerate(zip(key_sensors, sensor_labels)):
    ax = fig.add_subplot(gs[0, i])
    for eid, c in zip(sample_engines, colors_eng):
        eng = train_fe[train_fe['engine_id'] == eid].sort_values('cycle')
        ax.plot(eng['cycle'], eng[s], color=c, alpha=0.8, linewidth=1.5,
                label=f'Engine {eid}')
    ax.set_title(lbl, color=TEXT_CLR, fontsize=10)
    ax.set_xlabel('Cycle', color=TEXT_CLR)
    ax.set_ylabel('Normalized Value', color=TEXT_CLR)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc='upper left')

# Panel 4: RUL distribution
ax4 = fig.add_subplot(gs[0, 3])
ax4.hist(train['RUL'], bins=50, color=ACCENT1, alpha=0.7, edgecolor='none')
ax4.axvline(RUL_CAP, color=ACCENT3, linestyle='--', lw=2, label=f'Cap={RUL_CAP}')
ax4.set_title('RUL Distribution', color=TEXT_CLR)
ax4.set_xlabel('Remaining Useful Life (cycles)', color=TEXT_CLR)
ax4.set_ylabel('Count', color=TEXT_CLR)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Panel 5: Degradation curves (health index)
ax5 = fig.add_subplot(gs[1, :2])
for eid, c in zip(range(1, 9), [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
                                  GOLD, '#FF79C6', '#8BE9FD']):
    eng = train_fe[train_fe['engine_id'] == eid].sort_values('cycle')
    ax5.plot(eng['cycle_norm'], eng['health_index'], color=c, alpha=0.7, linewidth=1.5)
ax5.set_title('Engine Degradation Curves (8 Engines)', color=TEXT_CLR)
ax5.set_xlabel('Normalized Lifecycle (0=start, 1=failure)', color=TEXT_CLR)
ax5.set_ylabel('Health Index', color=TEXT_CLR)
ax5.grid(True, alpha=0.3)
ax5.axhline(0.4, color=ACCENT3, linestyle='--', lw=1.5, label='Critical threshold')
ax5.legend(fontsize=8)

# Panel 6: Sensor volatility
ax6 = fig.add_subplot(gs[1, 2:])
sensor_std_plot = train_fe[useful_sensors].std().sort_values(ascending=False)
bars = ax6.barh(sensor_std_plot.index, sensor_std_plot.values,
                color=[ACCENT1 if v > 0.15 else ACCENT4 for v in sensor_std_plot.values])
ax6.set_title('Sensor Variability (Std Dev)', color=TEXT_CLR)
ax6.set_xlabel('Standard Deviation', color=TEXT_CLR)
ax6.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, sensor_std_plot.values):
    ax6.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=7, color=TEXT_CLR)

fig.suptitle('Step 1 & 2 — Data Overview & Sensor Analysis', fontsize=14,
             color=TEXT_CLR, y=1.01, fontweight='bold')
plt.savefig(f'{FIG_PATH}/02_sensor_analysis.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Plot 2: Sensor Analysis & Degradation Curves")


# ─── Figure 3: Correlation Heatmap ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(DARK_BG)

# Sensor-sensor correlation
corr_data = train[useful_sensors + ['RUL']].corr()
mask = np.zeros_like(corr_data, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True

sns.heatmap(corr_data, ax=axes[0], cmap='RdBu_r', center=0,
            annot=False, linewidths=0.3, linecolor='#0D1117',
            cbar_kws={'shrink': 0.8})
axes[0].set_title('Sensor Correlation Matrix\n(incl. RUL)', color=TEXT_CLR, fontsize=12)
axes[0].tick_params(colors=TEXT_CLR, labelsize=8)
axes[0].set_facecolor(PANEL_BG)

# Sensor-RUL correlation bar
rul_corr = train[useful_sensors].corrwith(train['RUL']).sort_values()
colors_corr = [ACCENT2 if v > 0 else ACCENT3 for v in rul_corr.values]
axes[1].barh(rul_corr.index, rul_corr.values, color=colors_corr)
axes[1].set_title('Sensor Correlation with RUL', color=TEXT_CLR, fontsize=12)
axes[1].set_xlabel('Pearson Correlation Coefficient', color=TEXT_CLR)
axes[1].axvline(0, color=TEXT_CLR, lw=0.5)
axes[1].grid(True, alpha=0.3, axis='x')

fig.suptitle('Step 2 — Correlation Analysis', fontsize=14, color=TEXT_CLR,
             fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}/03_correlation_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Plot 3: Correlation Heatmap")


# ─── Figure 4: Feature Engineering Visualization ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(DARK_BG)
axes = axes.flatten()

sample_eng = train_fe[train_fe['engine_id'] == 3].sort_values('cycle')

# Raw vs rolling mean
for i, s in enumerate(['s4', 's9', 's11']):
    ax = axes[i]
    ax.plot(sample_eng['cycle'], sample_eng[s],        color=ACCENT1, alpha=0.4,
            linewidth=1, label='Raw')
    ax.plot(sample_eng['cycle'], sample_eng[f'{s}_rmean'], color=ACCENT2, linewidth=2,
            label='Rolling Mean (10)')
    ax.fill_between(sample_eng['cycle'],
                    sample_eng[f'{s}_rmean'] - sample_eng[f'{s}_rstd'],
                    sample_eng[f'{s}_rmean'] + sample_eng[f'{s}_rstd'],
                    color=ACCENT2, alpha=0.15, label='±1 Std')
    ax.set_title(f'Feature Engineering: {s}', color=TEXT_CLR)
    ax.set_xlabel('Cycle'); ax.set_ylabel('Value')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Health index over lifecycle
axes[3].plot(sample_eng['cycle_norm'], sample_eng['health_index'],
             color=GOLD, linewidth=2)
axes[3].fill_between(sample_eng['cycle_norm'], sample_eng['health_index'],
                     alpha=0.2, color=GOLD)
axes[3].axhline(0.3, color=ACCENT3, linestyle='--', label='Alert zone')
axes[3].set_title('Health Index Over Lifecycle', color=TEXT_CLR)
axes[3].set_xlabel('Normalized Lifecycle'); axes[3].set_ylabel('Health Index')
axes[3].legend(fontsize=8); axes[3].grid(True, alpha=0.3)

# Rolling std (volatility)
axes[4].plot(sample_eng['cycle'], sample_eng['s4_rstd'],  color=ACCENT3, label='s4 volatility')
axes[4].plot(sample_eng['cycle'], sample_eng['s9_rstd'],  color=ACCENT1, label='s9 volatility')
axes[4].plot(sample_eng['cycle'], sample_eng['s11_rstd'], color=ACCENT4, label='s11 volatility')
axes[4].set_title('Sensor Volatility (Rolling Std)', color=TEXT_CLR)
axes[4].set_xlabel('Cycle'); axes[4].set_ylabel('Std Dev')
axes[4].legend(fontsize=8); axes[4].grid(True, alpha=0.3)

# Feature count summary
feat_types = ['Raw Sensors', 'Rolling Mean', 'Rolling Std', 'Op Conditions', 'Derived']
feat_counts = [len(useful_sensors), len(useful_sensors), len(useful_sensors), 3, 2]
bars = axes[5].bar(feat_types, feat_counts,
                   color=[ACCENT1, ACCENT2, ACCENT3, ACCENT4, GOLD])
axes[5].set_title('Feature Engineering Summary', color=TEXT_CLR)
axes[5].set_ylabel('Number of Features')
for bar, val in zip(bars, feat_counts):
    axes[5].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 str(val), ha='center', color=TEXT_CLR, fontsize=10)
axes[5].grid(True, alpha=0.3, axis='y')

fig.suptitle('Step 3 — Feature Engineering', fontsize=14, color=TEXT_CLR,
             fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}/04_feature_engineering.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Plot 4: Feature Engineering")


# ─── Figure 5: Model Comparison Table ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor(DARK_BG)

model_names_short = ['Linear\nRegression', 'Ridge\nRegression',
                     'Random\nForest', 'Gradient\nBoosting']
rmse_vals = [results[k]['RMSE']  for k in results]
mae_vals  = [results[k]['MAE']   for k in results]
r2_vals   = [results[k]['R²']    for k in results]
bar_colors = [ACCENT4, ACCENT1, ACCENT2, ACCENT5]

# RMSE
bars = axes[0].bar(model_names_short, rmse_vals, color=bar_colors, width=0.5)
axes[0].set_title('RMSE (lower = better)', color=TEXT_CLR, fontsize=12)
axes[0].set_ylabel('RMSE (cycles)')
axes[0].grid(True, alpha=0.3, axis='y')
best_idx = np.argmin(rmse_vals)
bars[best_idx].set_edgecolor(GOLD); bars[best_idx].set_linewidth(3)
for bar, v in zip(bars, rmse_vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f'{v:.2f}', ha='center', color=TEXT_CLR, fontsize=9)

# MAE
bars = axes[1].bar(model_names_short, mae_vals, color=bar_colors, width=0.5)
axes[1].set_title('MAE (lower = better)', color=TEXT_CLR, fontsize=12)
axes[1].set_ylabel('MAE (cycles)')
axes[1].grid(True, alpha=0.3, axis='y')
best_idx = np.argmin(mae_vals)
bars[best_idx].set_edgecolor(GOLD); bars[best_idx].set_linewidth(3)
for bar, v in zip(bars, mae_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f'{v:.2f}', ha='center', color=TEXT_CLR, fontsize=9)

# R²
bars = axes[2].bar(model_names_short, r2_vals, color=bar_colors, width=0.5)
axes[2].set_title('R² Score (higher = better)', color=TEXT_CLR, fontsize=12)
axes[2].set_ylabel('R² Score')
axes[2].set_ylim(0, 1.05)
axes[2].grid(True, alpha=0.3, axis='y')
best_idx = np.argmax(r2_vals)
bars[best_idx].set_edgecolor(GOLD); bars[best_idx].set_linewidth(3)
for bar, v in zip(bars, r2_vals):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{v:.4f}', ha='center', color=TEXT_CLR, fontsize=9)

axes[2].text(0.98, 0.05, '★ = Best model', transform=axes[2].transAxes,
             ha='right', color=GOLD, fontsize=9)

fig.suptitle('Step 4 — Model Comparison (RMSE | MAE | R²)', fontsize=14,
             color=TEXT_CLR, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}/05_model_comparison.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Plot 5: Model Comparison")


# ─── Figure 6: Predicted vs Actual RUL ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(DARK_BG)
axes = axes.flatten()

for ax, (name, y_pred) in zip(axes, predictions.items()):
    # Sample 2000 points for clarity
    idx = np.random.choice(len(y_test), min(2000, len(y_test)), replace=False)
    yt = np.array(y_test)[idx]
    yp = y_pred[idx]

    # Density scatter
    ax.scatter(yt, yp, alpha=0.2, s=8, color=ACCENT1)
    mn, mx = 0, RUL_CAP
    ax.plot([mn, mx], [mn, mx], color=ACCENT3, lw=2, linestyle='--', label='Perfect fit')
    ax.plot([mn, mx], [mn+10, mx+10], color=ACCENT5, lw=1, linestyle=':', alpha=0.5)
    ax.plot([mn, mx], [mn-10, mx-10], color=ACCENT5, lw=1, linestyle=':', alpha=0.5)
    ax.fill_between([mn, mx], [mn-10, mx-10], [mn+10, mx+10],
                    color=ACCENT5, alpha=0.07, label='±10 cycle band')

    r2  = results[name]['R²']
    rmse= results[name]['RMSE']
    ax.set_title(f"{name.replace(chr(10),' ')}\nRMSE={rmse:.2f} | R²={r2:.4f}",
                 color=TEXT_CLR, fontsize=10)
    ax.set_xlabel('Actual RUL (cycles)'); ax.set_ylabel('Predicted RUL (cycles)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, RUL_CAP+5); ax.set_ylim(0, RUL_CAP+5)

fig.suptitle('Step 4 — Predicted vs Actual RUL', fontsize=14, color=TEXT_CLR,
             fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}/06_predicted_vs_actual.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Plot 6: Predicted vs Actual RUL")


# ─── Figure 7: Feature Importance + Failure Probability ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(DARK_BG)

# Feature importance from best tree model
best_tree = models[best_model_name]
if hasattr(best_tree, 'feature_importances_'):
    importances = best_tree.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).nlargest(20)
    color_map = []
    for f in feat_imp.index:
        if f in useful_sensors: color_map.append(ACCENT1)
        elif 'rmean' in f:       color_map.append(ACCENT2)
        elif 'rstd' in f:        color_map.append(ACCENT4)
        elif f in OP_COLS:       color_map.append(ACCENT5)
        else:                    color_map.append(GOLD)

    axes[0].barh(feat_imp.index[::-1], feat_imp.values[::-1], color=color_map[::-1])
    axes[0].set_title(f'Top 20 Feature Importances\n({best_model_name.replace(chr(10)," ")})',
                      color=TEXT_CLR, fontsize=12)
    axes[0].set_xlabel('Importance')
    axes[0].grid(True, alpha=0.3, axis='x')

    legend_patches = [
        mpatches.Patch(color=ACCENT1, label='Raw Sensor'),
        mpatches.Patch(color=ACCENT2, label='Rolling Mean'),
        mpatches.Patch(color=ACCENT4, label='Rolling Std'),
        mpatches.Patch(color=ACCENT5, label='Op Condition'),
        mpatches.Patch(color=GOLD,    label='Derived'),
    ]
    axes[0].legend(handles=legend_patches, fontsize=8, loc='lower right')

# Failure probability curve
rul_range = np.linspace(0, 150, 300)
# Logistic-style failure probability: P(fail) = 1 / (1 + exp((RUL-mu)/sigma))
mu, sigma = 20, 15
fail_prob = 1 / (1 + np.exp((rul_range - mu) / sigma))

axes[1].fill_between(rul_range, fail_prob, alpha=0.3, color=ACCENT3)
axes[1].plot(rul_range, fail_prob, color=ACCENT3, lw=3, label='Failure Probability')
axes[1].axvline(SAFETY_MARGIN, color=GOLD, linestyle='--', lw=2,
                label=f'Safety margin ({SAFETY_MARGIN} cycles)')
axes[1].axhline(0.5, color=ACCENT1, linestyle=':', lw=1, label='50% threshold')

# Color zones
axes[1].axvspan(0, SAFETY_MARGIN,        alpha=0.1, color=ACCENT3, label='CRITICAL zone')
axes[1].axvspan(SAFETY_MARGIN, 45,       alpha=0.05, color=GOLD)
axes[1].axvspan(45, 150,                 alpha=0.05, color=ACCENT2)

axes[1].set_title('Failure Probability Curve', color=TEXT_CLR, fontsize=12)
axes[1].set_xlabel('Remaining Useful Life (cycles)')
axes[1].set_ylabel('Probability of Failure')
axes[1].set_ylim(0, 1.05); axes[1].set_xlim(0, 150)
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

fig.suptitle('Step 4 — Feature Importance & Failure Probability', fontsize=14,
             color=TEXT_CLR, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}/07_feature_importance.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Plot 7: Feature Importance & Failure Probability")


# ─── Figure 8: Maintenance Optimization Dashboard ────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(DARK_BG)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Urgency breakdown pie
ax1 = fig.add_subplot(gs[0, 0])
urg_counts = schedule_df['Urgency'].value_counts()
urg_colors = {'CRITICAL': ACCENT3, 'HIGH': ACCENT5, 'MEDIUM': GOLD, 'LOW': ACCENT2}
pie_colors = [urg_colors.get(u, ACCENT1) for u in urg_counts.index]
wedges, texts, autotexts = ax1.pie(urg_counts.values, labels=urg_counts.index,
                                    colors=pie_colors, autopct='%1.0f%%',
                                    startangle=90, textprops={'color': TEXT_CLR})
for at in autotexts: at.set_fontsize(10)
ax1.set_title('Maintenance Urgency\nBreakdown', color=TEXT_CLR, fontsize=11)

# Predicted RUL scatter per engine
ax2 = fig.add_subplot(gs[0, 1:])
colors_sched = [urg_colors.get(u, ACCENT1) for u in schedule_df['Urgency']]
scatter = ax2.scatter(schedule_df['Engine'], schedule_df['Predicted RUL'],
                      c=colors_sched, s=80, zorder=5, edgecolors='white', linewidths=0.5)
ax2.axhline(SAFETY_MARGIN, color=ACCENT3, linestyle='--', lw=2,
            label=f'Critical zone (<{SAFETY_MARGIN} cycles)')
ax2.axhline(45, color=GOLD, linestyle='--', lw=1.5, label='High priority zone (<45 cycles)')
ax2.axhline(90, color=ACCENT2, linestyle='--', lw=1.5, label='Medium priority zone (<90 cycles)')
ax2.set_title('Predicted RUL per Engine', color=TEXT_CLR, fontsize=11)
ax2.set_xlabel('Engine ID'); ax2.set_ylabel('Predicted RUL (cycles)')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

legend_patches = [mpatches.Patch(color=v, label=k) for k,v in urg_colors.items()]
ax2.legend(handles=legend_patches, fontsize=8, loc='upper right')

# Cost analysis bar
ax3 = fig.add_subplot(gs[1, 0])
cost_by_urg = schedule_df.groupby('Urgency')['Estimated Cost ($)'].sum()
bars = ax3.bar(cost_by_urg.index, cost_by_urg.values,
               color=[urg_colors.get(u, ACCENT1) for u in cost_by_urg.index])
ax3.set_title('Maintenance Cost by\nUrgency Level', color=TEXT_CLR, fontsize=11)
ax3.set_ylabel('Total Cost ($)')
ax3.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, cost_by_urg.values):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
             f'${v:,.0f}', ha='center', color=TEXT_CLR, fontsize=8)

# Gantt-style maintenance schedule (next 50 cycles)
ax4 = fig.add_subplot(gs[1, 1:])
critical_eng = schedule_df[schedule_df['Urgency'].isin(['CRITICAL','HIGH'])].head(12)
yticks = []
yticklabels = []

for i, (_, row) in enumerate(critical_eng.iterrows()):
    rul  = row['Predicted RUL']
    eid  = row['Engine']
    urg  = row['Urgency']
    c    = urg_colors.get(urg, ACCENT1)

    # Current op bar
    ax4.barh(i, row['Current Cycle'], left=0, height=0.5,
             color='#21262D', alpha=0.6)
    # Remaining life bar
    ax4.barh(i, rul, left=row['Current Cycle'], height=0.5, color=c, alpha=0.8)
    # Maintenance point
    ax4.scatter(row['Current Cycle'] + max(rul - SAFETY_MARGIN, 1),
                i, color=GOLD, s=60, zorder=5, marker='v')
    yticks.append(i)
    yticklabels.append(f"Engine {eid}")

ax4.set_yticks(yticks); ax4.set_yticklabels(yticklabels, fontsize=8)
ax4.set_xlabel('Machine Cycle')
ax4.set_title('Maintenance Schedule (Gantt View)\n▼ = Recommended maintenance point',
              color=TEXT_CLR, fontsize=11)
ax4.grid(True, alpha=0.3, axis='x')

legend_patches = [
    mpatches.Patch(color=ACCENT3, label='CRITICAL'),
    mpatches.Patch(color=ACCENT5, label='HIGH'),
    mpatches.Patch(color='#21262D', label='Operated cycles'),
    mpatches.Patch(color=GOLD,    label='Maintenance point'),
]
ax4.legend(handles=legend_patches, fontsize=8, loc='lower right')

fig.suptitle('Step 5 — Maintenance Optimization Dashboard', fontsize=14,
             color=TEXT_CLR, fontweight='bold')
plt.savefig(f'{FIG_PATH}/08_maintenance_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Plot 8: Maintenance Dashboard")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 6] Saving Results...")

# Model comparison table
print("\n" + "="*65)
print(f"{'MODEL':<32} {'RMSE':>7} {'MAE':>7} {'R²':>8} {'TIME(s)':>8}")
print("-"*65)
for name, row in results_df.iterrows():
    marker = " ★" if name == best_model_name else "  "
    print(f"{name.replace(chr(10),' '):<32} {row['RMSE']:>7.2f} {row['MAE']:>7.2f} "
          f"{row['R²']:>8.4f} {row['Train Time (s)']:>7.2f}{marker}")
print("="*65)
print(f"\n  ★ Best model: {best_model_name.replace(chr(10),' ')}")

# Summary JSON
summary = {
    'dataset': 'NASA C-MAPSS Turbofan (synthetic simulation)',
    'train_engines': int(train['engine_id'].nunique()),
    'test_engines':  int(test['engine_id'].nunique()),
    'train_samples': len(X_train),
    'features':      len(feature_cols),
    'rul_cap':       RUL_CAP,
    'best_model':    best_model_name.replace('\n', ' '),
    'metrics': {k: {m: round(v, 4) for m, v in v2.items()}
                for k, v2 in results.items()},
    'optimization': {
        'total_cost':    total_cost,
        'critical_count': int((schedule_df['Urgency']=='CRITICAL').sum()),
        'high_count':    int((schedule_df['Urgency']=='HIGH').sum()),
    }
}
with open(f'{RES_PATH}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n  Figures saved:  {FIG_PATH}/")
print(f"  Results saved:  {RES_PATH}/")
print("\n" + "="*70)
print("  ✅  PIPELINE COMPLETE")
print("="*70)
