import numpy as np
import pandas as pd

WINDOW = 6


def calcular_stats_time(df, team, data):
    df_t = df[(df['Home'] == team) | (df['Away'] == team)].copy()
    df_t = df_t[df_t['Data'] < data].sort_values('Data').tail(WINDOW)

    if len(df_t) < WINDOW:
        return None

    is_home = df_t['Home'] == team

    stats = {}

    stats['gols_pro'] = np.where(is_home, df_t['H_Gols_FT'], df_t['A_Gols_FT']).mean()
    stats['gols_contra'] = np.where(is_home, df_t['A_Gols_FT'], df_t['H_Gols_FT']).mean()

    stats['chutes_pro'] = np.where(is_home, df_t['H_Chute'], df_t['A_Chute']).mean()
    stats['chutes_gol_pro'] = np.where(is_home, df_t['H_Chute_Gol'], df_t['A_Chute_Gol']).mean()

    stats['chutes_contra'] = np.where(is_home, df_t['A_Chute'], df_t['H_Chute']).mean()
    stats['chutes_gol_contra'] = np.where(is_home, df_t['A_Chute_Gol'], df_t['H_Chute_Gol']).mean()

    stats['ataques_pro'] = np.where(is_home, df_t['H_Ataques'], df_t['A_Ataques']).mean()
    stats['ataques_contra'] = np.where(is_home, df_t['A_Ataques'], df_t['H_Ataques']).mean()

    stats['escanteios_pro'] = np.where(is_home, df_t['H_Escanteios'], df_t['A_Escanteios']).mean()
    stats['escanteios_contra'] = np.where(is_home, df_t['A_Escanteios'], df_t['H_Escanteios']).mean()

    stats['eficiencia_ofensiva'] = stats['chutes_gol_pro'] / (stats['chutes_pro'] + 0.001)
    stats['perigo_defensivo'] = stats['chutes_gol_contra'] / (stats['chutes_contra'] + 0.001)

    return stats


def calcular_media_liga(df, liga, data):
    df_l = df[(df['Liga'] == liga) & (df['Data'] < data)].sort_values('Data').tail(WINDOW)

    if len(df_l) < WINDOW:
        return None

    return {
        'gols': ((df_l['H_Gols_FT'] + df_l['A_Gols_FT']) / 2).mean(),
        'chutes': ((df_l['H_Chute'] + df_l['A_Chute']) / 2).mean(),
        'chutes_gol': ((df_l['H_Chute_Gol'] + df_l['A_Chute_Gol']) / 2).mean(),
        'ataques': ((df_l['H_Ataques'] + df_l['A_Ataques']) / 2).mean(),
        'escanteios': ((df_l['H_Escanteios'] + df_l['A_Escanteios']) / 2).mean()
    }
