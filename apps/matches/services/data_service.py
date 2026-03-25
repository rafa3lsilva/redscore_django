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

    caminho = getattr(settings, 'CAMINHO_HISTORICO', '')
    if os.path.exists(caminho):
        try:
            df = pd.read_csv(caminho)
            df.columns = df.columns.str.strip()

            # 🔥 CORREÇÃO ESSENCIAL
            df['Data'] = (
                df['Data']
                .astype(str)
                .str.strip()
                .str[:10]
            )
            df['Data'] = pd.to_datetime(
                df['Data'],
                format='%Y-%m-%d',
                errors='coerce'
            )

            df = df.dropna(subset=['Data'])
            cache.set('historico_df', df, timeout=3600)
            return df

        except Exception as e:
            print(f"❌ Erro ao ler histórico: {e}")

    return pd.DataFrame()


def carregar_jogos_do_dia(data_selecionada_str=None):
    if not data_selecionada_str:
        data_selecionada_str = datetime.now().strftime('%Y-%m-%d')

    cache_key = f'jogos_dia_{data_selecionada_str}'
    cached_df = cache.get(cache_key)
    if cached_df is not None:
        return cached_df

    nomes_possiveis = [
        f"Jogos_do_Dia_RedScore_{data_selecionada_str}.csv",
        f"jogos_do_dia_{data_selecionada_str}.csv",
        f"Jogos_do_Dia_{data_selecionada_str}.csv"
    ]

    pasta_diaria = getattr(settings, 'PASTA_JOGOS_DO_DIA', '')

    for nome in nomes_possiveis:
        caminho = os.path.join(pasta_diaria, nome)
        if os.path.exists(caminho):
            try:
                df = pd.read_csv(caminho)
                df = normalizar_colunas(df)
                cache.set(cache_key, df, timeout=1800)
                return df
            except Exception as e:
                print(f"❌ Erro lendo {nome}: {e}")
    return pd.DataFrame()
