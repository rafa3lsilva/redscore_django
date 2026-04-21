import pandas as pd
import numpy as np
from scipy.stats import poisson, nbinom
from django.core.cache import cache
import hashlib
import json

def _get_cache_key(func_name, *args, **kwargs):
    # Create a unique key based on function name and arguments
    # Filter out large objects like DataFrames from the key, use their shape/last date instead
    processed_args = []
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            processed_args.append(f"df_{arg.shape}_{arg['Data'].max() if not arg.empty else 'empty'}")
        else:
            processed_args.append(str(arg))
    
    key_str = f"{func_name}:{processed_args}:{kwargs}"
    return f"stats_cache_{hashlib.md5(key_str.encode()).hexdigest()}"

# --- Funções Auxiliares ---
def obter_historico_recente(df, team, is_home, num_jogos, cenario):
    if df.empty: return df
    if cenario == 'geral':
        # Optimized filtering: use boolean indexing instead of .query or multiple steps
        mask = (df['Home'] == team) | (df['Away'] == team)
        return df[mask].tail(num_jogos)
    else:
        if is_home:
            return df[df['Home'] == team].tail(num_jogos)
        else:
            return df[df['Away'] == team].tail(num_jogos)

def _fit_nb_params(mu, var, eps=1e-9):
    if np.isnan(mu) or np.isnan(var) or var <= mu + eps:
        return None
    r = (mu * mu) / (var - mu)
    p = r / (r + mu)
    return r, p

def _pmf_nb_or_poisson(k_max, mu, var):
    params = _fit_nb_params(mu, var)
    if params is None or mu == 0:
        return poisson.pmf(np.arange(k_max + 1), mu)
    r, p = params
    return nbinom.pmf(np.arange(k_max + 1), r, p)

# --- Previsão de Gols (FT) ---
def prever_gols_ft(home, away, liga, df, num_jogos=5, cenario='casa_fora'):
    cache_key = _get_cache_key('prever_gols_ft', home, away, liga, num_jogos, cenario)
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result

    try:
        num_jogos = int(num_jogos)
    except (ValueError, TypeError):
        num_jogos = 5
        
    df_home = obter_historico_recente(df, home, True, num_jogos, cenario)
    df_away = obter_historico_recente(df, away, False, num_jogos, cenario)
    if df_home.empty or df_away.empty: return {}
    
    if cenario == 'geral':
        lambda_home = np.where(df_home['Home'] == home, df_home['H_Gols_FT'], df_home['A_Gols_FT']).mean()
        lambda_away = np.where(df_away['Home'] == away, df_away['A_Gols_FT'], df_away['H_Gols_FT']).mean()
    else:
        lambda_home = df_home['H_Gols_FT'].mean()
        lambda_away = df_away['A_Gols_FT'].mean()
    
    # Matriz Poisson - Vectorized
    max_gols = 6
    probs_h = poisson.pmf(np.arange(max_gols), lambda_home)
    probs_a = poisson.pmf(np.arange(max_gols), lambda_away)
    matriz = np.outer(probs_h, probs_a)
    
    # --- Ajuste Dixon-Coles (Inflação de Empates 0x0 e 1x1) ---
    rho = -0.10
    matriz[0, 0] *= max(0, 1 - rho * lambda_home * lambda_away)
    matriz[1, 0] *= max(0, 1 + rho * lambda_home)
    matriz[0, 1] *= max(0, 1 + rho * lambda_away)
    matriz[1, 1] *= max(0, 1 - rho)
    matriz /= matriz.sum() # Normaliza novamente

    # Probabilidades combinadas usando NumPy
    h_idx, a_idx = np.indices((max_gols, max_gols))
    p_over_25 = matriz[h_idx + a_idx > 2.5].sum()
    p_over_15 = matriz[h_idx + a_idx > 1.5].sum()
    
    # BTTS: 1 - P(Home=0) - P(Away=0) + P(0-0)
    p_btts = 1 - matriz[0, :].sum() - matriz[:, 0].sum() + matriz[0, 0]

    # Extrair Placares Mais Prováveis
    placares = []
    threshold = 0.005 # 0.5%
    mask = matriz > threshold
    matching_indices = np.argwhere(mask)
    for hg, ag in matching_indices:
        prob = matriz[hg, ag] * 100
        placares.append({'hg': int(hg), 'ag': int(ag), 'prob': round(prob, 1)})
    
    placares = sorted(placares, key=lambda x: x['prob'], reverse=True)[:5]

    result = {
        'lambda_home': round(float(lambda_home), 2),
        'lambda_away': round(float(lambda_away), 2),
        'prob_over_15': round(float(p_over_15 * 100), 1),
        'odd_over_15': round(float(1/p_over_15), 2) if p_over_15 > 0 else 0.0,
        'prob_under_15': round(float((1-p_over_15) * 100), 1),
        'odd_under_15': round(float(1/(1-p_over_15)), 2) if (1-p_over_15) > 0 else 0.0,
        'prob_over_25': round(float(p_over_25 * 100), 1),
        'odd_over_25': round(float(1/p_over_25), 2) if p_over_25 > 0 else 0.0,
        'prob_under_25': round(float((1-p_over_25) * 100), 1),
        'odd_under_25': round(float(1/(1-p_over_25)), 2) if (1-p_over_25) > 0 else 0.0,
        'prob_btts': round(float(p_btts * 100), 1),
        'odd_btts': round(float(1/p_btts), 2) if p_btts > 0 else 0.0,
        'prob_btts_nao': round(float((1 - p_btts) * 100), 1),
        'odd_btts_nao': round(float(1/(1-p_btts)), 2) if (1-p_btts) > 0 else 0.0,
        'placares_provaveis': placares
    }
    
    cache.set(cache_key, result, timeout=600)
    return result

# --- Previsão de Escanteios ---
def prever_escanteios(home, away, df, num_jogos=5, cenario='casa_fora'):
    cache_key = _get_cache_key('prever_escanteios', home, away, num_jogos, cenario)
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result

    try:
        num_jogos = int(num_jogos)
    except (ValueError, TypeError):
        num_jogos = 5
        
    df_h = obter_historico_recente(df, home, True, num_jogos, cenario)
    df_a = obter_historico_recente(df, away, False, num_jogos, cenario)
    if df_h.empty or df_a.empty: return {}

    if cenario == 'geral':
        cantos_h_arr = np.where(df_h['Home'] == home, df_h['H_Escanteios'], df_h['A_Escanteios']) if 'H_Escanteios' in df_h.columns else np.zeros(0)
        cantos_a_arr = np.where(df_a['Home'] == away, df_a['A_Escanteios'], df_a['H_Escanteios']) if 'H_Escanteios' in df_a.columns else np.zeros(0)
    else:
        cantos_h_arr = df_h['H_Escanteios'].values if 'H_Escanteios' in df_h.columns else np.zeros(0)
        cantos_a_arr = df_a['A_Escanteios'].values if 'A_Escanteios' in df_a.columns else np.zeros(0)

    # Limpando NaNs
    cantos_h_arr = cantos_h_arr[~pd.isna(cantos_h_arr)]
    cantos_a_arr = cantos_a_arr[~pd.isna(cantos_a_arr)]

    cantos_h = np.mean(cantos_h_arr) if len(cantos_h_arr) > 0 else 0
    cantos_a = np.mean(cantos_a_arr) if len(cantos_a_arr) > 0 else 0
    var_h = np.var(cantos_h_arr) if len(cantos_h_arr) > 0 else 0
    var_a = np.var(cantos_a_arr) if len(cantos_a_arr) > 0 else 0

    # Modelagem de Eventos em Cascata - Vectorized
    max_c = 15
    probs_h = _pmf_nb_or_poisson(max_c, cantos_h, var_h)
    probs_a = _pmf_nb_or_poisson(max_c, cantos_a, var_a)
    matriz = np.outer(probs_h, probs_a)
    matriz /= matriz.sum()
    
    h_idx, a_idx = np.indices((max_c + 1, max_c + 1))
    p_hm = matriz[h_idx > a_idx].sum()
    p_emp = matriz[h_idx == a_idx].sum()
    p_am = matriz[h_idx < a_idx].sum()
    
    # Encontrar a "Main Line" - Vectorized
    linhas = np.array([6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5])
    p_overs = np.array([matriz[h_idx + a_idx > l].sum() for l in linhas])
    best_idx = np.argmin(np.abs(p_overs - 0.5))
    
    best_linha = float(linhas[best_idx])
    best_p_over = float(p_overs[best_idx])

    result = {
        'media_home': round(float(cantos_h), 2),
        'media_away': round(float(cantos_a), 2),
        'total_esperado': round(float(cantos_h + cantos_a), 2),
        'prob_home_mais': round(float(p_hm * 100), 1),
        'odd_home_mais': round(float(1/p_hm), 2) if p_hm > 0 else 0.0,
        'prob_empate': round(float(p_emp * 100), 1),
        'odd_empate': round(float(1/p_emp), 2) if p_emp > 0 else 0.0,
        'prob_away_mais': round(float(p_am * 100), 1),
        'odd_away_mais': round(float(1/p_am), 2) if p_am > 0 else 0.0,
        'linha_principal': best_linha,
        'prob_over_main': round(float(best_p_over * 100), 1),
        'odd_over_main': round(float(1/best_p_over), 2) if best_p_over > 0 else 0.0
    }
    
    cache.set(cache_key, result, timeout=600)
    return result

# --- Previsão HT ---
def prever_gols_ht(home, away, df, num_jogos=5, cenario='casa_fora'):
    cache_key = _get_cache_key('prever_gols_ht', home, away, num_jogos, cenario)
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result

    try:
        num_jogos = int(num_jogos)
    except (ValueError, TypeError):
        num_jogos = 5
        
    df_h = obter_historico_recente(df, home, True, num_jogos, cenario)
    df_a = obter_historico_recente(df, away, False, num_jogos, cenario)

    if cenario == 'geral':
        arr_h = np.where(df_h['Home'] == home, df_h['H_Gols_HT'], df_h['A_Gols_HT']) if 'H_Gols_HT' in df_h.columns else np.zeros(0)
        arr_a = np.where(df_a['Home'] == away, df_a['A_Gols_HT'], df_a['H_Gols_HT']) if 'A_Gols_HT' in df_a.columns else np.zeros(0)
    else:
        arr_h = df_h['H_Gols_HT'].values if 'H_Gols_HT' in df_h.columns else np.zeros(0)
        arr_a = df_a['A_Gols_HT'].values if 'A_Gols_HT' in df_a.columns else np.zeros(0)
        
    # Limpando NaNs
    arr_h = arr_h[~np.isnan(arr_h)] if len(arr_h) > 0 else arr_h
    arr_a = arr_a[~np.isnan(arr_a)] if len(arr_a) > 0 else arr_a

    gols_ht_h = np.mean(arr_h) if len(arr_h) > 0 else 0
    gols_ht_a = np.mean(arr_a) if len(arr_a) > 0 else 0
    dp_h = np.std(arr_h) if len(arr_h) > 0 else 0
    dp_a = np.std(arr_a) if len(arr_a) > 0 else 0
    
    cv_h = (dp_h / gols_ht_h * 100) if gols_ht_h > 0 else 0
    cv_a = (dp_a / gols_ht_a * 100) if gols_ht_a > 0 else 0
    
    lambda_ht = gols_ht_h + gols_ht_a
    prob_gol_ht = 1 - poisson.pmf(0, lambda_ht)
    
    result = {
        'prob_gol_ht': round(float(prob_gol_ht * 100), 1),
        'odd_gol_ht': round(float(1/prob_gol_ht), 2) if prob_gol_ht > 0 else 0.0,
        'media_h': round(float(gols_ht_h), 2),
        'dp_h': round(float(dp_h), 2),
        'cv_h': round(float(cv_h), 1),
        'media_a': round(float(gols_ht_a), 2),
        'dp_a': round(float(dp_a), 2),
        'cv_a': round(float(cv_a), 1)
    }
    
    cache.set(cache_key, result, timeout=600)
    return result

# --- Confronto Direto Histórico Recente ---
def analisar_confronto_direto(home, away, df, num_jogos=5, cenario='casa_fora'):
    cache_key = _get_cache_key('analisar_confronto_direto', home, away, num_jogos, cenario)
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result

    try:
        num_jogos = int(num_jogos)
    except (ValueError, TypeError):
        num_jogos = 5

    df_h = obter_historico_recente(df, home, True, num_jogos, cenario)
    df_a = obter_historico_recente(df, away, False, num_jogos, cenario)

    def calc_stats(df_team, team_name):
        if df_team.empty:
            return {
                'gols_marcados': 0, 'gols_sofridos': 0, 'escanteios': 0, 'chutes_gol': 0,
                'btts_sim': 0, 'btts_nao': 0, 'clean_sheet': 0, 'over_25': 0
            }
        
        if cenario == 'geral':
            is_home_mask = (df_team['Home'] == team_name)
            gf_s = np.where(is_home_mask, df_team['H_Gols_FT'], df_team['A_Gols_FT'])
            gs_s = np.where(is_home_mask, df_team['A_Gols_FT'], df_team['H_Gols_FT'])
            esc_s = np.where(is_home_mask, df_team.get('H_Escanteios', 0), df_team.get('A_Escanteios', 0))
            cg_s = np.where(is_home_mask, df_team.get('H_Chute_Gol', 0), df_team.get('A_Chute_Gol', 0))
        else:
            is_home = (team_name == home)
            gf_col, gs_col = ('H_Gols_FT', 'A_Gols_FT') if is_home else ('A_Gols_FT', 'H_Gols_FT')
            esc_col = 'H_Escanteios' if is_home else 'A_Escanteios'
            cg_col = 'H_Chute_Gol' if is_home else 'A_Chute_Gol'

            gf_s = df_team[gf_col].values
            gs_s = df_team[gs_col].values
            esc_s = df_team[esc_col].values if esc_col in df_team.columns else np.zeros(len(df_team))
            cg_s = df_team[cg_col].values if cg_col in df_team.columns else np.zeros(len(df_team))

        # Vectorized stats
        gf = np.nanmean(gf_s)
        gs = np.nanmean(gs_s)
        esc = np.nanmean(esc_s)
        cg = np.nanmean(cg_s)

        # Booleans - Vectorized
        btts_mask = (gf_s > 0) & (gs_s > 0)
        btts = np.nanmean(btts_mask) * 100
        cs = np.nanmean(gs_s == 0) * 100
        ov25 = np.nanmean((gf_s + gs_s) > 2.5) * 100

        return {
            'gols_marcados': round(float(gf), 2),
            'gols_sofridos': round(float(gs), 2),
            'escanteios': round(float(esc), 2),
            'chutes_gol': round(float(cg), 2),
            'btts_sim': round(float(btts), 1),
            'btts_nao': round(float(100 - btts), 1),
            'clean_sheet': round(float(cs), 1),
            'over_25': round(float(ov25), 1)
        }

    result = {
        'home': calc_stats(df_h, home),
        'away': calc_stats(df_a, away),
        'num_jogos': num_jogos,
        'cenario': cenario
    }
    
    cache.set(cache_key, result, timeout=600)
    return result

# --- Placares Históricos Comuns ---
def analisar_placares_comuns(home, away, df, num_jogos=5, cenario='casa_fora'):
    cache_key = _get_cache_key('analisar_placares_comuns', home, away, num_jogos, cenario)
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result

    from collections import Counter
    try:
        num_jogos = int(num_jogos)
    except (ValueError, TypeError):
        num_jogos = 5

    df_h = obter_historico_recente(df, home, True, num_jogos, cenario)
    df_a = obter_historico_recente(df, away, False, num_jogos, cenario)

    def get_top(df_team, team_name):
        if df_team.empty: return []
        
        # Vectorized string creation for scores
        if cenario == 'geral':
            is_home = (df_team['Home'] == team_name)
            h_gols = np.where(is_home, df_team['H_Gols_FT'], df_team['A_Gols_FT']).astype(int)
            a_gols = np.where(is_home, df_team['A_Gols_FT'], df_team['H_Gols_FT']).astype(int)
        else:
            is_team_home = (team_name == home)
            h_gols = df_team['H_Gols_FT' if is_team_home else 'A_Gols_FT'].astype(int)
            a_gols = df_team['A_Gols_FT' if is_team_home else 'H_Gols_FT'].astype(int)
            
        scores = [f"{hg}-{ag}" for hg, ag in zip(h_gols, a_gols)]
        counts = Counter(scores)
        total = len(scores)
        return [{'score': s, 'count': c, 'pct': round((c/total)*100)} for s, c in counts.most_common(3)]

    result = {
        'home_scores': get_top(df_h, home),
        'away_scores': get_top(df_a, away),
        'num_jogos': num_jogos
    }
    
    cache.set(cache_key, result, timeout=600)
    return result

# --- Backtest de Desempenho por Odd ---
def analisar_desempenho_odd(team, odd_alvo, df, tolerancia=0.10):
    cache_key = _get_cache_key('analisar_desempenho_odd', team, odd_alvo, tolerancia)
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result

    if df.empty or not odd_alvo:
        return {'esperada': 0, 'real': 0, 'amostra': 0}

    try:
        odd_alvo = float(odd_alvo)
        if odd_alvo <= 1:
            return {'esperada': 0, 'real': 0, 'amostra': 0}
            
        esperada = (1 / odd_alvo) * 100
        min_odd = odd_alvo * (1 - tolerancia)
        max_odd = odd_alvo * (1 + tolerancia)
        
        # Jogos como mandante
        df_home = df[df['Home'] == team]
        if 'Odd_H' in df_home.columns:
            df_home_faixa = df_home[(df_home['Odd_H'] >= min_odd) & (df_home['Odd_H'] <= max_odd)]
            vitorias_home = len(df_home_faixa[df_home_faixa['H_Gols_FT'] > df_home_faixa['A_Gols_FT']])
            total_home = len(df_home_faixa)
        else:
            vitorias_home, total_home = 0, 0
            
        # Jogos como visitante
        df_away = df[df['Away'] == team]
        if 'Odd_A' in df_away.columns:
            df_away_faixa = df_away[(df_away['Odd_A'] >= min_odd) & (df_away['Odd_A'] <= max_odd)]
            vitorias_away = len(df_away_faixa[df_away_faixa['A_Gols_FT'] > df_away_faixa['H_Gols_FT']])
            total_away = len(df_away_faixa)
        else:
            vitorias_away, total_away = 0, 0
            
        total_jogos = total_home + total_away
        vitorias = vitorias_home + vitorias_away
        
        real = (vitorias / total_jogos * 100) if total_jogos > 0 else 0
        
        result = {
            'esperada': round(esperada, 1),
            'real': round(real, 1),
            'amostra': total_jogos
        }
        cache.set(cache_key, result, timeout=1800)
        return result
        
    except Exception as e:
        print(f"Erro em analisar_desempenho_odd: {e}")
        return {'esperada': 0, 'real': 0, 'amostra': 0}