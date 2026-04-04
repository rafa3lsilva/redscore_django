import numpy as np
import joblib
import pandas as pd
from .ia_features import calcular_stats_time, calcular_media_liga
from django.core.cache import cache
import hashlib

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_redscore_v2.pkl")

MODEL_ERROR = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {MODEL_PATH}")
    BUNDLE = joblib.load(MODEL_PATH)
    IA_MODEL = BUNDLE["model"]
    FEATURE_NAMES = BUNDLE["features"]
except Exception as e:
    MODEL_ERROR = str(e)
    print(f"⚠️ Erro ao carregar modelo IA. Certifique-se de que o arquivo pkl existe: {e}")
    BUNDLE = None
    IA_MODEL = None
    FEATURE_NAMES = []


def remover_juice_odds(odd_h, odd_d, odd_a):
    p_h = 1 / odd_h
    p_d = 1 / odd_d
    p_a = 1 / odd_a
    over = p_h + p_d + p_a

    return (
        p_h / over,
        p_d / over,
        p_a / over
    )


def calcular_probabilidades_ia(
    home: str,
    away: str,
    liga: str,
    odd_h: float,
    odd_d: float,
    odd_a: float,
    df_historico,
    peso_recente: int = 50
):
    # Caching predictions
    cache_key_str = f"ia_pred_{home}_{away}_{liga}_{odd_h}_{odd_d}_{odd_a}_{peso_recente}_{df_historico.shape}"
    cache_key = f"ia_pred_{hashlib.md5(cache_key_str.encode()).hexdigest()}"
    cached_res = cache.get(cache_key)
    if cached_res:
        return cached_res

    if IA_MODEL is None:
        return {'erro': f'Modelo de IA não disponível: {MODEL_ERROR}'}
    
    model = IA_MODEL
    feature_names = FEATURE_NAMES

    # --- Data do jogo ---
    # df_historico['Data'] is already datetime from data_service
    data_jogo = df_historico['Data'].max() + pd.Timedelta(days=1)

    # --- Stats ---
    stats_home = calcular_stats_time(df_historico, home, data_jogo)
    stats_away = calcular_stats_time(df_historico, away, data_jogo)
    stats_liga = calcular_media_liga(df_historico, liga, data_jogo)

    # --- Fallback ---
    if stats_home is None or stats_away is None or stats_liga is None:
        prob_h, prob_d, prob_a = remover_juice_odds(odd_h, odd_d, odd_a)
        fallback = True

    else:
        features = {}
        
        # Blending Factor (Site feature: allows user to weight recent form)
        # 1.0 means 100% recent form (matches training logic)
        factor = peso_recente / 50.0

        # 1. Gols Pro (Relative to League)
        base_gols = stats_liga['gols'] + 0.001
        avg_gols = stats_liga['gols']
        
        adj_h_gols = avg_gols + (stats_home['gols_pro'] - avg_gols) * factor
        adj_a_gols = avg_gols + (stats_away['gols_pro'] - avg_gols) * factor
        
        features['home_gols_pro_rel'] = adj_h_gols / base_gols
        features['away_gols_pro_rel'] = adj_a_gols / base_gols
        features['diff_gols_pro'] = adj_h_gols - adj_a_gols

        # 2. Resultado Num (Forma) - Pure differentials
        adj_h_res = stats_home['resultado_num'] * factor
        adj_a_res = stats_away['resultado_num'] * factor
        
        features['home_resultado_num'] = adj_h_res
        features['away_resultado_num'] = adj_a_res
        features['diff_resultado_num'] = adj_h_res - adj_a_res

        # 3. Eficiência e Perigo (Pure differentials)
        features['home_eficiencia_ofensiva'] = stats_home['eficiencia_ofensiva'] * factor
        features['away_eficiencia_ofensiva'] = stats_away['eficiencia_ofensiva'] * factor
        features['diff_eficiencia_ofensiva'] = (
            features['home_eficiencia_ofensiva'] - features['away_eficiencia_ofensiva']
        )

        features['home_perigo_defensivo'] = stats_home['perigo_defensivo'] * factor
        features['away_perigo_defensivo'] = stats_away['perigo_defensivo'] * factor
        features['diff_perigo_defensivo'] = (
            features['home_perigo_defensivo'] - features['away_perigo_defensivo']
        )

        # 4. xG Estimado (Pure differentials)
        features['home_xG_estimado'] = stats_home['xG_estimado'] * factor
        features['away_xG_estimado'] = stats_away['xG_estimado'] * factor
        features['diff_xG_estimado'] = features['home_xG_estimado'] - features['away_xG_estimado']

        # 5. Odds and Probabilidades Justas
        features['Odd_H'] = odd_h
        features['Odd_D'] = odd_d
        features['Odd_A'] = odd_a

        p_h_j, p_d_j, p_a_j = remover_juice_odds(odd_h, odd_d, odd_a)
        features['prob_justa_h'] = p_h_j
        features['prob_justa_d'] = p_d_j
        features['prob_justa_a'] = p_a_j

        # --- DataFrame alinhado com a ordem exata do treinamento ---
        X = pd.DataFrame([features], columns=feature_names)
        probs = model.predict_proba(X)[0]

        prob_h, prob_d, prob_a = probs
        fallback = False

    res = {
        'prob_casa': round(float(prob_h * 100), 2),
        'prob_empate': round(float(prob_d * 100), 2),
        'prob_fora': round(float(prob_a * 100), 2),
        'odd_justa_casa': round(float(1 / prob_h), 2) if prob_h > 0 else 0.0,
        'odd_justa_empate': round(float(1 / prob_d), 2) if prob_d > 0 else 0.0,
        'odd_justa_fora': round(float(1 / prob_a), 2) if prob_a > 0 else 0.0,
        'fallback': fallback
    }
    
    cache.set(cache_key, res, timeout=1800)
    return res
