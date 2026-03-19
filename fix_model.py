"""
=============================================================
fix_model.py  —  Run this ONCE to fix your model.pkl
WHAT IT DOES:
  1. Reads your existing model.pkl (bare RandomForest object)
  2. Re-trains a StandardScaler on the original dataset
  3. Wraps everything into a proper dict bundle
  4. Saves the new model.pkl (overwrites the old one)

HOW TO RUN:
  cd "c:\\Users\\Dell\\New folder\\Food-Waste_Detection"
  python fix_model.py

After it prints "SUCCESS", just run app.py normally.
=============================================================
"""

import os, sys, joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── File paths ─────────────────────────────────────────────
MODEL_PATH   = os.path.join(BASE_DIR, 'model.pkl')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.csv')

# ── Validate files exist ───────────────────────────────────
print("=" * 55)
print("  FoodSense AI — model.pkl Fix Utility")
print("=" * 55)

if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] model.pkl not found at:\n  {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(DATASET_PATH):
    print(f"\n[ERROR] dataset.csv not found at:\n  {DATASET_PATH}")
    print("  Place dataset.csv in the same folder as this script.")
    sys.exit(1)

print(f"\n[1/5] Found model.pkl  : {MODEL_PATH}")
print(f"[1/5] Found dataset.csv: {DATASET_PATH}")

# ── Load existing model ────────────────────────────────────
print("\n[2/5] Loading existing model.pkl …")
raw = joblib.load(MODEL_PATH)

if isinstance(raw, dict):
    print("      model.pkl is already a dict bundle — checking keys …")
    required = {'model', 'scaler', 'le_day', 'features'}
    if required.issubset(raw.keys()):
        print("      All keys present. Your model.pkl is already fixed!")
        print("\n✅  Nothing to do. Run app.py directly.")
        sys.exit(0)
    else:
        missing = required - set(raw.keys())
        print(f"      Missing keys: {missing}. Rebuilding …")
        bare_model = raw.get('model', None)
        if bare_model is None:
            print("[ERROR] Cannot find 'model' key inside the dict either.")
            sys.exit(1)
else:
    bare_model = raw
    print(f"      Type: {type(bare_model).__name__}  (bare model — will wrap into dict)")

# ── Verify the model has the right number of features ──────
n_features = getattr(bare_model, 'n_features_in_', None)
print(f"      Model expects {n_features} input features")

# ── Constants — match your training notebook exactly ───────
DAY_ORDER = ['Monday','Tuesday','Wednesday','Thursday',
             'Friday','Saturday','Sunday']

# The 14 features from your model_metadata.json
FEATURE_COLS = [
    'Day_Encoded',
    'Festival',
    'Expected_Customers',
    'Previous_Day_Consumption',
    'Previous_Week_Same_Day',
    'Weather_Sunny',
    'Weather_Cloudy',
    'Weather_Rainy',
    'Weather_Stormy',
    'Is_Weekend',
    'Avg_Historical_Demand',
    'Customer_Demand_Ratio',
    'Festival_Customer_Interaction',
    'Demand_Momentum',
]

if n_features and n_features != len(FEATURE_COLS):
    print(f"\n[WARNING] Model expects {n_features} features but FEATURE_COLS has {len(FEATURE_COLS)}.")
    print("          Attempting to use model's own feature_names_in_ …")
    if hasattr(bare_model, 'feature_names_in_'):
        FEATURE_COLS = list(bare_model.feature_names_in_)
        print(f"          Using: {FEATURE_COLS}")
    else:
        print("[ERROR] Cannot determine correct feature list. Aborting.")
        sys.exit(1)

# ── Load dataset and rebuild scaler ───────────────────────
print("\n[3/5] Loading dataset and rebuilding StandardScaler …")
df = pd.read_csv(DATASET_PATH)
df = df.dropna()

day_map = {d: i for i, d in enumerate(DAY_ORDER)}
df['Day_Encoded'] = df['Day_of_Week'].map(day_map)

w = pd.get_dummies(df['Weather'], prefix='Weather')
df = pd.concat([df, w], axis=1)
for col in ['Weather_Sunny','Weather_Cloudy','Weather_Rainy','Weather_Stormy']:
    if col not in df.columns:
        df[col] = 0

df['Is_Weekend'] = df['Day_of_Week'].isin(['Saturday','Sunday']).astype(int)
df['Avg_Historical_Demand']          = (df['Previous_Day_Consumption'] + df['Previous_Week_Same_Day']) / 2.0
df['Customer_Demand_Ratio']          = df['Previous_Day_Consumption'] / df['Expected_Customers'].replace(0, 1)
df['Festival_Customer_Interaction']  = df['Festival'] * df['Expected_Customers']
df['Demand_Momentum']                = df['Previous_Day_Consumption'] - df['Previous_Week_Same_Day']

X = df[FEATURE_COLS]
y = df['Meals_Consumed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)
print(f"      Scaler fitted on {len(X_train)} training rows, {len(FEATURE_COLS)} features")

# Quick accuracy check
X_test_s = scaler.transform(X_test)
preds = bare_model.predict(X_test_s)
mae = float(np.mean(np.abs(y_test.values - preds)))
print(f"      Quick MAE check: {mae:.2f} meals  (original model still intact)")

# ── Build LabelEncoder ─────────────────────────────────────
le_day = LabelEncoder()
le_day.fit(DAY_ORDER)

# ── Wrap into dict bundle ──────────────────────────────────
print("\n[4/5] Wrapping into dict bundle …")
bundle = {
    'model'   : bare_model,
    'scaler'  : scaler,
    'le_day'  : le_day,
    'features': FEATURE_COLS,
}

# ── Save (overwrites old model.pkl) ───────────────────────
print(f"\n[5/5] Saving fixed model.pkl to:\n      {MODEL_PATH}")
joblib.dump(bundle, MODEL_PATH)

# ── Verify the saved file ──────────────────────────────────
check = joblib.load(MODEL_PATH)
assert isinstance(check, dict),                     "VERIFY FAILED: not a dict"
assert 'model'    in check,                         "VERIFY FAILED: missing 'model'"
assert 'scaler'   in check,                         "VERIFY FAILED: missing 'scaler'"
assert 'le_day'   in check,                         "VERIFY FAILED: missing 'le_day'"
assert 'features' in check,                         "VERIFY FAILED: missing 'features'"
assert check['features'] == FEATURE_COLS,           "VERIFY FAILED: feature list mismatch"
assert type(check['model']).__name__ == type(bare_model).__name__, "VERIFY FAILED: model type changed"

print("\n" + "=" * 55)
print("  SUCCESS — model.pkl fixed and verified")
print("=" * 55)
print(f"\n  Model    : {type(check['model']).__name__}")
print(f"  Scaler   : {type(check['scaler']).__name__}")
print(f"  Features : {len(check['features'])}")
print(f"  MAE      : {mae:.2f} meals")
print(f"\n  Now run:  python app.py")
print("=" * 55)
