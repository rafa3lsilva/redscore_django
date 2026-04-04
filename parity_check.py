
from apps.matches.services import ia_features, ia_predictor
import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from unittest.mock import MagicMock

# Setup paths
ML_PATH = '/home/rafael/Documentos/redscore_ML'
DJANGO_PATH = '/home/rafael/Documentos/redscore_django'
sys.path.append(DJANGO_PATH)

# Mock Django BEFORE importing services
mock_django = MagicMock()
sys.modules['django'] = mock_django
sys.modules['django.core'] = MagicMock()
sys.modules['django.core.cache'] = MagicMock()

# Import Django-side services

# 1. Load historical data
RAW_DATA_PATH = os.path.join(ML_PATH, 'dados_redscore.csv')
df = pd.read_csv(RAW_DATA_PATH)
df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
df = df.dropna(subset=['Data']).sort_values('Data')

# 2. Pick a sample match from history
sample_match = df.iloc[-1]
home, away, liga = sample_match['Home'], sample_match['Away'], sample_match['Liga']
odd_h, odd_d, odd_a = 2.0, 3.2, 3.8
data_jogo = sample_match['Data']

print(f"Testing Parity for: {home} vs {away} ({liga}) on {data_jogo}")

# --- PARITY EXECUTION ---
try:
    # 1. Load model and features
    bundle = joblib.load(os.path.join(
        DJANGO_PATH, 'apps/matches/services/modelo_redscore_v2.pkl'))
    model = bundle['model']
    feature_names = bundle['features']

    # 2. Run Django-side prediction
    print("\nRunning Django-side prediction...")

    class MockCache:
        def get(self, k): return None
        def set(self, k, v, timeout=0): pass

    ia_features.cache = MockCache()
    ia_predictor.cache = MockCache()

    res = ia_predictor.calcular_probabilidades_ia(
        home, away, liga, odd_h, odd_d, odd_a, df, peso_recente=50
    )

    print(f"Probabilities (Django): {res}")

    # 3. Validation: Check features
    original_predict_proba = model.predict_proba
    captured_X_list = []

    def mocked_predict_proba(X):
        captured_X_list.append(X)
        return original_predict_proba(X)

    model.predict_proba = mocked_predict_proba

    ia_predictor.calcular_probabilidades_ia(
        home, away, liga, odd_h, odd_d, odd_a, df, peso_recente=50
    )

    captured_X = captured_X_list[0]
    print("\nFeature Check (X passed to model):")
    print(captured_X)

    nan_cols = captured_X.columns[captured_X.isna().any()].tolist()
    if nan_cols:
        print(f"❌ ERROR: NaNs found in features: {nan_cols}")
    else:
        print("✅ SUCCESS: No NaNs found in features.")

    if list(captured_X.columns) == feature_names:
        print("✅ SUCCESS: Feature names and order match exactly.")
    else:
        print(f"❌ ERROR: Feature mismatch.")
        print(f"Expected: {feature_names}")
        print(f"Got:      {list(captured_X.columns)}")

except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
