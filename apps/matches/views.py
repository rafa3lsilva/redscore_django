from typing import Any, Dict
from django.shortcuts import render
from django.conf import settings
from datetime import datetime
import pandas as pd
from .services import estatisticas, ia_predictor
from .services.data_service import parse_odd, get_historico, carregar_jogos_do_dia

# --- VIEWS ---

def pagina_inicial(request):
    data_filtro = request.GET.get('data', datetime.now().strftime('%Y-%m-%d'))
    
    df_jogos = carregar_jogos_do_dia(data_filtro)
    jogos_por_liga = {}

    if not df_jogos.empty:
        # A pedido do usuário, removemos a ordenação global ('hora', 'liga') 
        # para preservar exatamente a ordem e o agrupamento nativos do CSV fonte.
        
        # Itera linha a linha preservando a ordem do CSV
        for _, row in df_jogos.iterrows():
            liga = row.get('liga', 'Outros')

            # Tratamento da hora para ficar bonita (10:30 em vez de 10:30:00)
            hora_raw = row.get('hora') or row.get('time') or '--:--'
            hora_str_upper = str(hora_raw).strip().upper()
            
            # Ignorar jogos adiados (POSTP) ou sem horário válido ('NAN' gerado pelo pandas)
            if 'POSTP' in hora_str_upper or hora_str_upper == 'NAN' or hora_str_upper == '--:--':
                continue
                
            hora = str(hora_raw).replace('.0', '')[0:5]
            
            # Como o Python 3.7+ mantém a ordem de inserção nos dicionários:
            # A primeira vez que a liga aparecer (no jogo mais cedo), 
            # ela cria a "Caixa" da liga na posição correta.
            if liga not in jogos_por_liga: 
                jogos_por_liga[liga] = []
            
            jogos_por_liga[liga].append({
                'home': row.get('home'),
                'away': row.get('away'),
                'liga': liga,
                'hora': hora,
                'Odd_H': row.get('odd_h', 0),
                'Odd_D': row.get('odd_d', 0),
                'Odd_A': row.get('odd_a', 0),
            })

    return render(request, 'matches/index.html', {
        'jogos_por_liga': jogos_por_liga,
        'data_selecionada': data_filtro
    })

def analise_jogo(request):
    # 1. Captura PARÂMETROS da URL
    home = request.GET.get('home')
    away = request.GET.get('away')
    liga = request.GET.get('liga', 'Liga Desconhecida')
    hora = request.GET.get('hora', '--:--') # <--- Capturando a hora
    data_jogo = request.GET.get('data', datetime.now().strftime('%Y-%m-%d'))
    
    # IA Peso Slider
    peso_str = request.GET.get('peso', '50')
    try:
        peso_atual = int(peso_str)
    except ValueError:
        peso_atual = 50

    # Filtros Confronto Direto
    num_jogos = request.GET.get('num_jogos', 5)
    cenario = request.GET.get('cenario', 'casa_fora')

    # Se faltar time, volta para home
    if not home or not away:
        print("❌ Times não informados na URL")
        return pagina_inicial(request)

    # 2. Tratamento das odds (robusto)
    odd_h = parse_odd(request.GET.get('odd_h'))
    odd_d = parse_odd(request.GET.get('odd_d'))
    odd_a = parse_odd(request.GET.get('odd_a'))

    odds_validas = all([odd_h, odd_d, odd_a])

    print(f"🔍 Analisando: {home} vs {away} | Odds: {odd_h}/{odd_d}/{odd_a}")

    # 3. Carrega Histórico
    df_historico = get_historico()
    
    # 4. Monta o Contexto (Variáveis para o HTML)
    context: Dict[str, Any] = {
        'home_team': home,   # Nome simples para usar {{ home_team }}
        'away_team': away,   # Nome simples para usar {{ away_team }}
        'liga': liga,
        'hora_jogo': hora,   # <--- Passando a hora
        'data_jogo': data_jogo,
        'peso_atual': peso_atual,
        'analise': {}
    }

    if df_historico.empty:
        print("❌ Histórico vazio")
        return render(request, 'matches/analise.html', context)

    # 5. Estatísticas matemáticas (sempre)
    context['analise']['ft'] = estatisticas.prever_gols_ft(
        home, away, liga, df_historico, num_jogos=num_jogos, cenario=cenario
    )
    context['analise']['ht'] = estatisticas.prever_gols_ht(
        home, away, df_historico, num_jogos=num_jogos, cenario=cenario
    )
    context['analise']['cantos'] = estatisticas.prever_escanteios(
        home, away, df_historico, num_jogos=num_jogos, cenario=cenario
    )
    context['analise']['confronto'] = estatisticas.analisar_confronto_direto(
        home, away, df_historico, num_jogos=num_jogos, cenario=cenario
    )
    context['analise']['placares_comuns'] = estatisticas.analisar_placares_comuns(
        home, away, df_historico, num_jogos=num_jogos, cenario=cenario
    )

    # 5.1 Extração Inteligente do Placar Exato via Dixon-Coles
    context['analise']['placares'] = context['analise']['ft'].get('placares_provaveis', [])

    # Histórico de Jogos
    try:
        num_j_hist = int(num_jogos)
    except:
        num_j_hist = 5

    def get_hist(team, is_home_view):
        if cenario == 'geral':
            df_t = df_historico[(df_historico['Home'] == team) | (df_historico['Away'] == team)].tail(num_j_hist)
        else:
            if is_home_view:
                df_t = df_historico[df_historico['Home'] == team].tail(num_j_hist)
            else:
                df_t = df_historico[df_historico['Away'] == team].tail(num_j_hist)
                
        h_list = []
        for _, row in df_t.iterrows():
            dt = str(row.get('Data', ''))[:10]
            hg = int(row.get('H_Gols_FT', 0)) if not pd.isna(row.get('H_Gols_FT')) else 0
            ag = int(row.get('A_Gols_FT', 0)) if not pd.isna(row.get('A_Gols_FT')) else 0
            h_ht = int(row.get('H_Gols_HT', 0)) if 'H_Gols_HT' in row and not pd.isna(row.get('H_Gols_HT')) else 0
            a_ht = int(row.get('A_Gols_HT', 0)) if 'A_Gols_HT' in row and not pd.isna(row.get('A_Gols_HT')) else 0
            h_cantos = int(row.get('H_Escanteios', 0)) if 'H_Escanteios' in row and not pd.isna(row.get('H_Escanteios')) else 0
            a_cantos = int(row.get('A_Escanteios', 0)) if 'A_Escanteios' in row and not pd.isna(row.get('A_Escanteios')) else 0
            
            r_home = row.get('Home', '')
            r_away = row.get('Away', '')
            
            if hg == ag:
                res = 'E'
            elif (r_home == team and hg > ag) or (r_away == team and ag > hg):
                res = 'V'
            else:
                res = 'D'

            liga = row.get('League', row.get('Liga', ''))

            h_list.append({
                'data': dt,
                'liga': liga,
                'home': r_home,
                'away': r_away,
                'hg': hg, 'ag': ag,
                'h_ht': h_ht, 'a_ht': a_ht,
                'h_cantos': h_cantos, 'a_cantos': a_cantos,
                'resultado': res
            })
        return list(reversed(h_list))

    context['analise']['hist_home'] = get_hist(home, True)
    context['analise']['hist_away'] = get_hist(away, False)

    # 6. IA (somente se odds forem válidas)
    if odds_validas:
        try:
            print(f"🤖 Rodando IA... Com Peso Recente = {peso_atual}%")
            preds = ia_predictor.calcular_probabilidades_ia(
                home, away, liga, odd_h, odd_d, odd_a, df_historico, peso_atual
            )
            context['analise']['ia'] = preds
            print(f"✅ Resultado IA: {preds}")
        except Exception as e:
            msg = f"Erro ao processar IA: {e}"
            print(f"❌ {msg}")
            context['analise']['ia'] = {'erro': msg}
            context['mostrar_form_odds'] = True
    else:
        print("⚠️ Odds ausentes ou inválidas")
        context['analise']['ia'] = {'erro': 'Odds ausentes'}
        context['mostrar_form_odds'] = True

    return render(request, 'matches/analise.html', context)