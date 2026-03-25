import pandas as pd
import numpy as np
from scipy.stats import poisson, nbinom

# --- Funções Auxiliares ---
def obter_historico_recente(df, team, is_home, num_jogos, cenario):
    if df.empty: return df
    if cenario == 'geral':
        return df[(df['Home'] == team) | (df['Away'] == team)].tail(num_jogos)
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
        return [poisson.pmf(k, mu) for k in range(k_max + 1)]
    r, p = params
    return [nbinom.pmf(k, r, p) for k in range(k_max + 1)]

# --- Previsão de Gols (FT) ---
def prever_gols_ft(home, away, liga, df, num_jogos=5, cenario='casa_fora'):
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
    
    # Matriz Poisson
    max_gols = 6
    probs_h = [poisson.pmf(i, lambda_home) for i in range(max_gols)]
    probs_a = [poisson.pmf(i, lambda_away) for i in range(max_gols)]
    matriz = np.outer(probs_h, probs_a)
    
    # --- Ajuste Dixon-Coles (Inflação de Empates 0x0 e 1x1) ---
    rho = -0.10
    matriz[0, 0] *= max(0, 1 - rho * lambda_home * lambda_away)
    matriz[1, 0] *= max(0, 1 + rho * lambda_home)
    matriz[0, 1] *= max(0, 1 + rho * lambda_away)
    matriz[1, 1] *= max(0, 1 - rho)
    matriz /= matriz.sum() # Normaliza novamente

    p_over_25 = sum(matriz[i, j] for i in range(max_gols) for j in range(max_gols) if i+j > 2.5)
    p_over_15 = sum(matriz[i, j] for i in range(max_gols) for j in range(max_gols) if i+j > 1.5)
    p_btts = 1 - matriz[0, :].sum() - matriz[:, 0].sum() + matriz[0,0]

    # Extrair Placares Mais Prováveis diretamente da Matriz
    placares = []
    for hg in range(max_gols):
        for ag in range(max_gols):
            prob = matriz[hg, ag] * 100
            if prob > 0.5:
                placares.append({'hg': hg, 'ag': ag, 'prob': round(prob, 1)})
    placares = sorted(placares, key=lambda x: x['prob'], reverse=True)[:5]

    return {
        'lambda_home': round(lambda_home, 2),
        'lambda_away': round(lambda_away, 2),
        'prob_over_15': round(p_over_15 * 100, 1),
        'odd_over_15': round(1/p_over_15, 2) if p_over_15 > 0 else 0.0,
        'prob_under_15': round((1-p_over_15) * 100, 1),
        'odd_under_15': round(1/(1-p_over_15), 2) if (1-p_over_15) > 0 else 0.0,
        'prob_over_25': round(p_over_25 * 100, 1),
        'odd_over_25': round(1/p_over_25, 2) if p_over_25 > 0 else 0.0,
        'prob_under_25': round((1-p_over_25) * 100, 1),
        'odd_under_25': round(1/(1-p_over_25), 2) if (1-p_over_25) > 0 else 0.0,
        'prob_btts': round(p_btts * 100, 1),
        'odd_btts': round(1/p_btts, 2) if p_btts > 0 else 0.0,
        'prob_btts_nao': round((1 - p_btts) * 100, 1),
        'odd_btts_nao': round(1/(1-p_btts), 2) if (1-p_btts) > 0 else 0.0,
        'placares_provaveis': placares
    }

# --- Previsão de Escanteios ---
def prever_escanteios(home, away, df, num_jogos=5, cenario='casa_fora'):
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
    try:
        cantos_h_arr = cantos_h_arr[~pd.isna(cantos_h_arr)]
        cantos_a_arr = cantos_a_arr[~pd.isna(cantos_a_arr)]
    except:
        pass

    cantos_h = np.mean(cantos_h_arr) if len(cantos_h_arr) > 0 else 0
    cantos_a = np.mean(cantos_a_arr) if len(cantos_a_arr) > 0 else 0
    var_h = np.var(cantos_h_arr) if len(cantos_h_arr) > 0 else 0
    var_a = np.var(cantos_a_arr) if len(cantos_a_arr) > 0 else 0

    # Modelagem de Eventos em Cascata (Binomial Negativa) / Poisson
    max_c = 15
    probs_h = _pmf_nb_or_poisson(max_c, cantos_h, var_h)
    probs_a = _pmf_nb_or_poisson(max_c, cantos_a, var_a)
    matriz = np.outer(probs_h, probs_a)
    matriz /= matriz.sum()
    
    p_hm = sum(matriz[i, j] for i in range(max_c) for j in range(max_c) if i > j)
    p_emp = sum(matriz[i, j] for i in range(max_c) for j in range(max_c) if i == j)
    p_am = sum(matriz[i, j] for i in range(max_c) for j in range(max_c) if i < j)
    
    # Encontrar a "Main Line" (linha com odd mais próxima de 2.0 / prob de 50%)
    best_linha = 9.5
    best_diff = 1.0
    best_p_over = 0.0
    for linha in [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]:
        p_over = sum(matriz[i, j] for i in range(max_c) for j in range(max_c) if i + j > linha)
        diff = abs(p_over - 0.5)
        if diff < best_diff:
            best_diff = diff
            best_linha = linha
            best_p_over = p_over

    return {
        'media_home': round(cantos_h, 2),
        'media_away': round(cantos_a, 2),
        'total_esperado': round(cantos_h + cantos_a, 2),
        'prob_home_mais': round(p_hm * 100, 1),
        'odd_home_mais': round(1/p_hm, 2) if p_hm > 0 else 0.0,
        'prob_empate': round(p_emp * 100, 1),
        'odd_empate': round(1/p_emp, 2) if p_emp > 0 else 0.0,
        'prob_away_mais': round(p_am * 100, 1),
        'odd_away_mais': round(1/p_am, 2) if p_am > 0 else 0.0,
        'linha_principal': best_linha,
        'prob_over_main': round(best_p_over * 100, 1),
        'odd_over_main': round(1/best_p_over, 2) if best_p_over > 0 else 0.0
    }

# --- Previsão HT ---
def prever_gols_ht(home, away, df, num_jogos=5, cenario='casa_fora'):
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
    
    return {
        'prob_gol_ht': round(prob_gol_ht * 100, 1),
        'odd_gol_ht': round(1/prob_gol_ht, 2) if prob_gol_ht > 0 else 0.0,
        'media_h': round(gols_ht_h, 2),
        'dp_h': round(dp_h, 2),
        'cv_h': round(cv_h, 1),
        'media_a': round(gols_ht_a, 2),
        'dp_a': round(dp_a, 2),
        'cv_a': round(cv_a, 1)
    }

# --- Confronto Direto Histórico Recente ---
def analisar_confronto_direto(home, away, df, num_jogos=5, cenario='casa_fora'):
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
            gf_s = np.where(df_team['Home'] == team_name, df_team['H_Gols_FT'], df_team['A_Gols_FT'])
            gs_s = np.where(df_team['Home'] == team_name, df_team['A_Gols_FT'], df_team['H_Gols_FT'])
            esc_s = np.where(df_team['Home'] == team_name, df_team['H_Escanteios'], df_team['A_Escanteios']) if 'H_Escanteios' in df_team.columns else np.zeros(len(df_team))
            cg_s = np.where(df_team['Home'] == team_name, df_team['H_Chute_Gol'], df_team['A_Chute_Gol']) if 'H_Chute_Gol' in df_team.columns else np.zeros(len(df_team))
        else:
            is_home = (team_name == home)
            gf_col = 'H_Gols_FT' if is_home else 'A_Gols_FT'
            gs_col = 'A_Gols_FT' if is_home else 'H_Gols_FT'
            esc_col = 'H_Escanteios' if is_home else 'A_Escanteios'
            cg_col = 'H_Chute_Gol' if is_home else 'A_Chute_Gol'

            gf_s = df_team[gf_col].values
            gs_s = df_team[gs_col].values
            esc_s = df_team[esc_col].values if esc_col in df_team.columns else np.zeros(len(df_team))
            cg_s = df_team[cg_col].values if cg_col in df_team.columns else np.zeros(len(df_team))

        gf = np.mean(gf_s)
        gs = np.mean(gs_s)
        esc = np.mean(esc_s)
        cg = np.mean(cg_s)

        # Booleans
        btts = np.mean((gf_s > 0) & (gs_s > 0)) * 100
        cs = np.mean(gs_s == 0) * 100
        ov25 = np.mean((gf_s + gs_s) > 2.5) * 100

        return {
            'gols_marcados': round(gf, 2),
            'gols_sofridos': round(gs, 2),
            'escanteios': round(esc, 2),
            'chutes_gol': round(cg, 2),
            'btts_sim': round(btts, 1),
            'btts_nao': round(100 - btts, 1),
            'clean_sheet': round(cs, 1),
            'over_25': round(ov25, 1)
        }

    return {
        'home': calc_stats(df_h, home),
        'away': calc_stats(df_a, away),
        'num_jogos': num_jogos,
        'cenario': cenario
    }

# --- Placares Históricos Comuns ---
def analisar_placares_comuns(home, away, df, num_jogos=5, cenario='casa_fora'):
    from collections import Counter
    try:
        num_jogos = int(num_jogos)
    except (ValueError, TypeError):
        num_jogos = 5

    df_h = obter_historico_recente(df, home, True, num_jogos, cenario)
    df_a = obter_historico_recente(df, away, False, num_jogos, cenario)

    def get_top(df_team, team_name):
        if df_team.empty: return []
        scores = []
        for _, row in df_team.iterrows():
            if cenario == 'geral':
                if row['Home'] == team_name:
                    scores.append(f"{int(row['H_Gols_FT'])}-{int(row['A_Gols_FT'])}")
                else:
                    scores.append(f"{int(row['A_Gols_FT'])}-{int(row['H_Gols_FT'])}")
            else:
                if team_name == home:
                    scores.append(f"{int(row['H_Gols_FT'])}-{int(row['A_Gols_FT'])}")
                else:
                    scores.append(f"{int(row['A_Gols_FT'])}-{int(row['H_Gols_FT'])}")
        counts = Counter(scores)
        total = len(scores)
        if total == 0: return []
        return [{'score': s, 'count': c, 'pct': round((c/total)*100)} for s, c in counts.most_common(3)]

    return {
        'home_scores': get_top(df_h, home),
        'away_scores': get_top(df_a, away),
        'num_jogos': num_jogos
    }