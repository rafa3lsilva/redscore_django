import pandas as pd
import os
from datetime import datetime
from django.conf import settings
from django.core.cache import cache

def parse_odd(val):
    try:
        val = float(str(val).replace(',', '.'))
        return val if val > 1 else None
    except Exception:
        return None

def normalizar_colunas(df):
    if df.empty: return df
    df.columns = df.columns.str.strip().str.lower()
    return df

def get_historico():
    cached_df = cache.get('historico_df')
    if cached_df is not None:
        return cached_df

    # Ler diretamente da tabela do banco de dados invés do arquivo CSV
    from apps.matches.models import Historico
    try:
        qs = Historico.objects.all().values()
        df = pd.DataFrame.from_records(qs)
        if df.empty:
            return df
            
        # Renomear as colunas para o padrão antigo (com iniciais maiúsculas) que as views.py e estatisticas.py esperam
        rename_map = {
            'data': 'Data',
            'home': 'Home',
            'away': 'Away',
            'liga': 'League',
            'h_gols_ft': 'H_Gols_FT',
            'a_gols_ft': 'A_Gols_FT',
            'h_gols_ht': 'H_Gols_HT',
            'a_gols_ht': 'A_Gols_HT',
            'h_escanteios': 'H_Escanteios',
            'a_escanteios': 'A_Escanteios'
        }
        df = df.rename(columns=rename_map)
        
        # Manter compatibilidade do tipo Data de datetime local
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        df = df.dropna(subset=['Data'])
        
        cache.set('historico_df', df, timeout=1800)
        return df

    except Exception as e:
        print(f"❌ Erro ao ler histórico do Supabase: {e}")

    return pd.DataFrame()


def carregar_jogos_do_dia(data_selecionada_str=None):
    if not data_selecionada_str:
        data_selecionada_str = datetime.now().strftime('%Y-%m-%d')

    cache_key = f'jogos_dia_{data_selecionada_str}'
    cached_df = cache.get(cache_key)
    if cached_df is not None:
        return cached_df

    from apps.matches.models import JogoDoDia
    
    try:
        qs = JogoDoDia.objects.filter(data=data_selecionada_str).values()
        df = pd.DataFrame.from_records(qs)
        
        if df.empty:
            return df
            
        # As views.py já esperavam tudo minusculo graças ao normalizador antigo,
        # O DataFrame do model (que já é lowercase nos fields) será perfeitamente compatível!
        cache.set(cache_key, df, timeout=300)
        return df
        
    except Exception as e:
        print(f"❌ Erro lendo jogos do dia do banco de dados: {e}")
        
    return pd.DataFrame()
