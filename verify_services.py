
import os
import django
import sys
import pandas as pd
from datetime import datetime

# Setup Django environment
sys.path.append('/home/rafael/Documentos/redscore_django')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from apps.matches.services import data_service, estatisticas, ia_predictor

def test_services():
    print("Testing services...")
    
    # 1. Test data_service
    df = data_service.get_historico()
    print(f"DataService: Loaded {len(df)} historical records.")
    assert not df.empty, "Historical data should not be empty"
    assert 'Data' in df.columns, "Data column should exist"
    assert pd.api.types.is_datetime64_any_dtype(df['Data']), "Data column should be datetime"

    # 2. Test estatisticas
    home, away, liga = "Flamengo", "Palmeiras", "Serie A"
    
    print(f"Testing statistics for {home} vs {away}...")
    ft = estatisticas.prever_gols_ft(home, away, liga, df)
    assert 'prob_over_25' in ft, "FT prediction should have prob_over_25"
    print(f"FT Prob Over 2.5: {ft['prob_over_25']}%")

    ht = estatisticas.prever_gols_ht(home, away, df)
    assert 'prob_gol_ht' in ht, "HT prediction should have prob_gol_ht"
    print(f"HT Prob Gol: {ht['prob_gol_ht']}%")

    cantos = estatisticas.prever_escanteios(home, away, df)
    assert 'total_esperado' in cantos, "Cantos prediction should have total_esperado"
    print(f"Expected Corners: {cantos['total_esperado']}")

    # 3. Test ia_predictor
    print("Testing IA Predictor...")
    ia_res = ia_predictor.calcular_probabilidades_ia(home, away, liga, 2.0, 3.4, 3.8, df)
    assert 'prob_casa' in ia_res, "IA prediction should have prob_casa"
    print(f"IA Prob Casa: {ia_res['prob_casa']}%")
    
    print("✅ All services verified successfully!")

if __name__ == "__main__":
    try:
        test_services()
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
