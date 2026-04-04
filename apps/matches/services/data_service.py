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

    from django.db import connection
    try:
        with connection.cursor() as cursor:
            # Consulta na tabela manual 'dados_redscore' (histórico)
            query = """
                SELECT "Data" as data, "Home" as home, "Away" as away, 
                       "Liga" as liga, "H_Gols_FT" as h_gols_ft, 
                       "A_Gols_FT" as a_gols_ft, "H_Gols_HT" as h_gols_ht, 
                       "A_Gols_HT" as a_gols_ht, "H_Escanteios" as h_escanteios, 
                       "A_Escanteios" as a_escanteios,
                       "H_Chute" as h_chute, "A_Chute" as a_chute,
                       "H_Chute_Gol" as h_chute_gol, "A_Chute_Gol" as a_chute_gol,
                       "H_Ataques" as h_ataques, "A_Ataques" as a_ataques
                FROM dados_redscore
            """
            cursor.execute(query)
            columns = [col[0].lower() for col in cursor.description]
            records = cursor.fetchall()
            
        df = pd.DataFrame.from_records(records, columns=columns)
        
        if df.empty:
            return df
            
        # Renomear as colunas para o padrão antigo
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
            'a_escanteios': 'A_Escanteios',
            'h_chute': 'H_Chute',
            'a_chute': 'A_Chute',
            'h_chute_gol': 'H_Chute_Gol',
            'a_chute_gol': 'A_Chute_Gol',
            'h_ataques': 'H_Ataques',
            'a_ataques': 'A_Ataques'
        }
        df = df.rename(columns=rename_map)
        
        # Manter compatibilidade do tipo Data de datetime local
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        df = df.dropna(subset=['Data'])
        
        # Sort data once for efficiency in later filtering
        df = df.sort_values('Data')
        
        # Pre-calculate common stats to avoid repeating in ia_features
        # This is a small optimization but adds up
        
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

    from django.db import connection
    
    try:
        with connection.cursor() as cursor:
            # Query the user's custom table exactly as it is in Supabase
            query = """
                SELECT data, liga, hora, home, away, 
                       "Odd_H" as odd_h, "Odd_D" as odd_d, "Odd_A" as odd_a, 
                       link_confronto 
                FROM jogos_do_dia 
                WHERE data = %s
                ORDER BY liga, hora, home
            """
            cursor.execute(query, [data_selecionada_str])
            columns = [col[0].lower() for col in cursor.description]
            records = cursor.fetchall()
            
        df = pd.DataFrame.from_records(records, columns=columns)
        
        if df.empty:
            return pd.DataFrame()
            
        cache.set(cache_key, df, timeout=300)
        return df
        
    except Exception as e:
        print(f"❌ Erro lendo jogos do dia do banco de dados: {e}")
        return pd.DataFrame()
        
    return pd.DataFrame()
