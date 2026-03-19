"""
=============================================================
Smart Food Waste Prediction System — Flask Application
=============================================================


"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# CONSTANTS — must match training pipeline exactly
# ─────────────────────────────────────────────────────────────

# Ordinal map: preserves Mon → Sun weekly order
# LabelEncoder would give alphabetical: Fri=0, Mon=1… (WRONG)
DAY_ORDER = ['Monday','Tuesday','Wednesday','Thursday',
             'Friday','Saturday','Sunday']
DAY_MAP   = {d: i for i, d in enumerate(DAY_ORDER)}

WEATHER_CATS = ['Sunny','Cloudy','Rainy','Stormy']

# Exact 14 features the trained model expects (from model_metadata.json)
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

# ─────────────────────────────────────────────────────────────
# LOAD MODEL BUNDLE
# ─────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

print("Loading model bundle …")
model_data = joblib.load(MODEL_PATH)

# Guard: fail fast with a clear message if pkl is still the bare model
if not isinstance(model_data, dict):
    raise TypeError(
        f"\n\nmodel.pkl contains a {type(model_data).__name__}, not a dict.\n"
        "This is the root cause of the TypeError crash.\n"
        "Replace model.pkl with the fixed version provided alongside this app.py.\n"
    )

model             = model_data['model']
scaler            = model_data['scaler']
le_day            = model_data['le_day']        # kept for reference; not used for encoding
expected_features = model_data['features']      # 14-column list

print(f"  Model    : {type(model).__name__}")
print(f"  Features : {len(expected_features)}")
print("  Status   : OK\n")


# ─────────────────────────────────────────────────────────────
# FEATURE BUILDER
# Mirrors the training pipeline exactly — all 14 columns
# ─────────────────────────────────────────────────────────────

def build_feature_row(day_of_week: str,
                      weather: str,
                      festival: int,
                      expected_customers: float,
                      prev_day: float,
                      prev_week: float) -> pd.DataFrame:
   

    #  ordinal encoding, not LabelEncoder
    day_encoded = DAY_MAP.get(day_of_week, 0)

    # Weather one-hot (all 4 columns, no drop_first)
    weather_sunny  = 1 if weather == 'Sunny'  else 0
    weather_cloudy = 1 if weather == 'Cloudy' else 0
    weather_rainy  = 1 if weather == 'Rainy'  else 0
    weather_stormy = 1 if weather == 'Stormy' else 0

    is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0

    # Engineered features — must match training notebook exactly
    avg_hist_demand   = (prev_day + prev_week) / 2.0
    cust_demand_ratio = prev_day / max(expected_customers, 1)
    festival_cust_int = festival * expected_customers

    #'Demand_Momentum' not 'Historical_Trend'
    demand_momentum   = prev_day - prev_week

    row = {
        'Day_Encoded'                  : day_encoded,
        'Festival'                     : festival,
        'Expected_Customers'           : expected_customers,
        'Previous_Day_Consumption'     : prev_day,
        'Previous_Week_Same_Day'       : prev_week,
        'Weather_Sunny'                : weather_sunny,
        'Weather_Cloudy'               : weather_cloudy,
        'Weather_Rainy'                : weather_rainy,
        'Weather_Stormy'               : weather_stormy,
        'Is_Weekend'                   : is_weekend,
        'Avg_Historical_Demand'        : avg_hist_demand,
        'Customer_Demand_Ratio'        : cust_demand_ratio,
        'Festival_Customer_Interaction': festival_cust_int,
        'Demand_Momentum'              : demand_momentum,
    }

    # Force exact column order from saved feature list
    return pd.DataFrame([row])[expected_features]


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/notebook/<path:filename>')
def serve_notebook(filename):
    notebook_dir = os.path.join(BASE_DIR, 'notebook')
    return send_from_directory(notebook_dir, filename)


@app.route('/predict', methods=['POST'])
def predict():
    """
    accepts both JSON (fetch/axios from index.html)
    and form-encoded POST from HTML forms.
    """
    try:
        # Detect content type and parse accordingly
        if request.is_json:
            data = request.get_json(force=True)
        else:
            data = request.form

        # ── Parse & type-cast inputs ──────────────────────
        day_of_week        = str(data.get('Day_of_Week',               'Monday')).strip()
        weather            = str(data.get('Weather',                   'Sunny')).strip()
        festival           = int(data.get('Festival',                   0))
        expected_customers = float(data.get('Expected_Customers',       300))
        prev_day           = float(data.get('Previous_Day_Consumption', 300))
        prev_week          = float(data.get('Previous_Week_Same_Day',   300))

        # ── Input validation ──────────────────────────────
        if day_of_week not in DAY_MAP:
            return jsonify({
                'success': False,
                'error'  : f"Invalid day '{day_of_week}'. Choose from: {DAY_ORDER}"
            }), 400

        if weather not in WEATHER_CATS:
            return jsonify({
                'success': False,
                'error'  : f"Invalid weather '{weather}'. Choose from: {WEATHER_CATS}"
            }), 400

        # ── Build feature vector ──────────────────────────
        df_input = build_feature_row(
            day_of_week, weather, festival,
            expected_customers, prev_day, prev_week
        )

        # ── Scale ─────────────────────────────────────────
        X_scaled = scaler.transform(df_input)

        # ── Predict ───────────────────────────────────────
        raw        = float(model.predict(X_scaled)[0])
        prediction = max(0, round(raw))

        # ── Confidence band (±1 MAE ≈ 25 meals) ──────────
        MAE        = 25
        lower      = max(0, round(prediction - MAE))
        upper      = round(prediction + MAE)

        # ── Safety buffer (+8%) ───────────────────────────
        buffer      = round(prediction * 0.08)
        recommended = prediction + buffer

        return jsonify({
            'success'    : True,
            'prediction' : prediction,
            'lower_bound': lower,
            'upper_bound': upper,
            'recommended': recommended,
            'waste_buffer': buffer,
            'message'    : 'Prediction successful',
        })

    except Exception as e:
        print("Prediction Error:\n", traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/health')
def health():
    return jsonify({
        'status'  : 'ok',
        'model'   : type(model).__name__,
        'features': len(expected_features),
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
