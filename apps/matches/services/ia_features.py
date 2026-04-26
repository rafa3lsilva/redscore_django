import numpy as np
import pandas as pd
from django.core.cache import cache
import hashlib

WINDOW = 6


def _get_cache_key(func_name, team_or_liga, data, df):
    # Use shape and last date of DF to ensure cache invalidation if data changes
    df_info = f"{df.shape}_{df['Data'].max() if not df.empty else 'empty'}"
    key_str = f"{func_name}:{team_or_liga}:{data}:{df_info}"
    return f"ia_feat_{hashlib.md5(key_str.encode()).hexdigest()}"


def calcular_stats_time(df, team, data):
    cache_key = _get_cache_key('stats_time', team, data, df)
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result

    # Optimized filtering: df is already sorted by Data in data_service
    # Optimized filtering: case-insensitive to handle different data sources
    mask = ((df['Home'].str.lower() == team.lower()) | (df['Away'].str.lower() == team.lower())) & (df['Data'] < data)
    df_t = df[mask].tail(WINDOW)

    if len(df_t) < WINDOW:
        return None

    is_home = df_t['Home'] == team

    stats = {}

    # Gols e Chutes
    stats['gols_pro'] = np.where(
        is_home, df_t['H_Gols_FT'], df_t['A_Gols_FT']).mean()
    stats['gols_contra'] = np.where(
        is_home, df_t['A_Gols_FT'], df_t['H_Gols_FT']).mean()

    stats['chutes_pro'] = np.where(
        is_home, df_t['H_Chute'], df_t['A_Chute']).mean()
    stats['chutes_gol_pro'] = np.where(
        is_home, df_t['H_Chute_Gol'], df_t['A_Chute_Gol']).mean()

    stats['chutes_contra'] = np.where(
        is_home, df_t['A_Chute'], df_t['H_Chute']).mean()
    stats['chutes_gol_contra'] = np.where(
        is_home, df_t['A_Chute_Gol'], df_t['H_Chute_Gol']).mean()

    # Pontuação de Forma (Últimos jogos)
    # 3 pts vitória, 1 empate, 0 derrota
    # Logic from feature_engineering.py
    is_home_win = (df_t['Home'] == team) & (
        df_t['H_Gols_FT'] > df_t['A_Gols_FT'])
    is_away_win = (df_t['Away'] == team) & (
        df_t['A_Gols_FT'] > df_t['H_Gols_FT'])
    is_draw = df_t['H_Gols_FT'] == df_t['A_Gols_FT']

    stats['resultado_num'] = np.where(
        is_home_win | is_away_win, 3, np.where(is_draw, 1, 0)).mean()

    # Novas Features: Eficiência e Perigo
    stats['eficiencia_ofensiva'] = stats['chutes_gol_pro'] / \
        (stats['chutes_pro'] + 0.001)
    stats['perigo_defensivo'] = stats['chutes_gol_contra'] / \
        (stats['chutes_contra'] + 0.001)

    # xG Simplificado: média de gols na janela (alinhado com feature_engineering.py)
    stats['xG_estimado'] = stats['gols_pro']

    cache.set(cache_key, stats, timeout=3600)
    return stats


def calcular_media_liga(df, liga, data):
    cache_key = _get_cache_key('media_liga', liga, data, df)
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result

    # Case-insensitive league filtering
    mask = (df['League'].str.lower() == liga.lower()) & (df['Data'] < data)
    df_l = df[mask].tail(WINDOW)

    if len(df_l) < WINDOW:
        return None

    stats = {
        'gols': ((df_l['H_Gols_FT'] + df_l['A_Gols_FT']) / 2).mean(),
        'chutes': ((df_l['H_Chute'] + df_l['A_Chute']) / 2).mean(),
        'chutes_gol': ((df_l['H_Chute_Gol'] + df_l['A_Chute_Gol']) / 2).mean(),
        'ataques': ((df_l['H_Ataques'] + df_l['A_Ataques']) / 2).mean(),
        'escanteios': ((df_l['H_Escanteios'] + df_l['A_Escanteios']) / 2).mean()
    }

    cache.set(cache_key, stats, timeout=3600)
    return stats
