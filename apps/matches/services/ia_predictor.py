import numpy as np
import joblib
import pandas as pd
from .ia_features import calcular_stats_time, calcular_media_liga


MODEL_PATH = "apps/matches/services/modelo_redscore_v2.pkl"

try:
    BUNDLE = joblib.load(MODEL_PATH)
    IA_MODEL = BUNDLE["model"]
    FEATURE_NAMES = BUNDLE["features"]
except Exception as e:
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
    if IA_MODEL is None:
        return {'erro': 'Modelo de IA não disponível'}
    
    model = IA_MODEL
    feature_names = FEATURE_NAMES

    # --- Data do jogo ---
    df_historico['Data'] = pd.to_datetime(df_historico['Data'], errors='coerce')
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

        mapa = {
            'gols_pro': 'gols',
            'gols_contra': 'gols',
            'chutes_pro': 'chutes',
            'chutes_gol_pro': 'chutes_gol',
            'chutes_contra': 'chutes',
            'chutes_gol_contra': 'chutes_gol',
            'ataques_pro': 'ataques',
            'ataques_contra': 'ataques',
            'escanteios_pro': 'escanteios',
            'escanteios_contra': 'escanteios'
        }

        # Blending Factor
        factor = peso_recente / 50.0

        for col, col_liga in mapa.items():
            base = stats_liga[col_liga] + 0.001
            raw_h = stats_home[col]
            raw_a = stats_away[col]
            avg_l = stats_liga[col_liga]
            
            adj_h = avg_l + (raw_h - avg_l) * factor
            adj_a = avg_l + (raw_a - avg_l) * factor

            features[f'home_{col}'] = adj_h / base
            features[f'away_{col}'] = adj_a / base
            features[f'diff_{col}'] = adj_h - adj_a

        # Stats whose baseline is roughly 0
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

        features['odd_h'] = odd_h
        features['odd_d'] = odd_d
        features['odd_a'] = odd_a

        # --- DataFrame alinhado ---
        X = pd.DataFrame([features], columns=feature_names)
        probs = model.predict_proba(X)[0]

        prob_h, prob_d, prob_a = probs
        fallback = False

    return {
        'prob_casa': round(float(prob_h * 100), 2),
        'prob_empate': round(float(prob_d * 100), 2),
        'prob_fora': round(float(prob_a * 100), 2),
        'odd_justa_casa': round(float(1 / prob_h), 2) if prob_h > 0 else 0.0,
        'odd_justa_empate': round(float(1 / prob_d), 2) if prob_d > 0 else 0.0,
        'odd_justa_fora': round(float(1 / prob_a), 2) if prob_a > 0 else 0.0,
        'fallback': fallback
    }
