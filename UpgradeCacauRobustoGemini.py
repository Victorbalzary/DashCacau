import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
import numpy as np
import streamlit as st
import mplfinance as mpf
from io import BytesIO
from matplotlib.lines import Line2D

# --- Configuração da Página do Streamlit ---
st.set_page_config(layout="wide", page_title="Dashboard de Análise de Cacau")

# --- Funções utilitárias ---
@st.cache_data
def obter_historico_anual(ticker: str, ano: int) -> pd.DataFrame | None:
    """Busca o histórico de preços para o ano (retorna None se vazio/erro)."""
    try:
        data_inicial = f"{ano}-01-01"
        data_final = f"{ano}-12-31"
        dados = yf.download(ticker, start=data_inicial, end=data_final, progress=False, auto_adjust=True)
        if dados is None or dados.empty:
            return None
        # garante índice datetime
        if not isinstance(dados.index, pd.DatetimeIndex):
            dados.index = pd.to_datetime(dados.index, errors='coerce')
        # remove timezone se houver
        try:
            if hasattr(dados.index, 'tz') and dados.index.tz is not None:
                dados.index = dados.index.tz_convert(None)
        except Exception:
            try:
                dados.index = dados.index.tz_localize(None)
            except Exception:
                pass
        return dados
    except Exception:
        return None

def extrair_close_series(dados: pd.DataFrame) -> pd.Series | None:
    """Retorna uma Series do preço de fechamento, tentando vários caminhos seguros."""
    if dados is None or dados.empty:
        return None

    # tenta nomes comuns
    candidates = ['Close', 'Adj Close', 'Adj_Close', 'close', 'adj close', 'adj_close']
    for c in candidates:
        if c in dados.columns:
            s = dados[c]
            if isinstance(s, pd.Series):
                return s

    # se tiver apenas uma coluna, squeeze para Series
    if dados.shape[1] == 1:
        s = dados.squeeze()
        if isinstance(s, pd.Series):
            return s

    # procura coluna cujo nome contenha 'close'
    for c in dados.columns:
        if 'close' in str(c).lower():
            s = dados[c]
            if isinstance(s, pd.Series):
                return s

    return None

def gerar_relatorio_pdf(df_analise, df_stats, df_projecao, fig, ticker, ano):
    """Gera o relatório em PDF e retorna bytes."""
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

        fig_text = plt.figure(figsize=(11.69, 8.27))
        texto_analise = df_analise.round(2).to_string()
        texto_stats = df_stats.round(2).to_string()
        texto_projecao = df_projecao.round(2).to_string() if df_projecao is not None else "Sem projeção"
        texto_final = f"""
Relatório de Análise de Preços do Cacau ({ticker})

Data de Emissão: {datetime.now().strftime('%d/%m/%Y %H:%M')}
--------------------------------------------------------------------

1. Análise Mensal (Preço Real vs. Meta)
--------------------------------------------------------------------
{texto_analise}


2. Estatísticas Descritivas (Baseado na Média Mensal)
--------------------------------------------------------------------
{texto_stats}


3. Projeção de Preços (Próximos Meses via Regressão Linear)
--------------------------------------------------------------------
{texto_projecao}
"""
        fig_text.text(0.01, 0.99, texto_final, transform=fig_text.transFigure, family='monospace',
                      verticalalignment='top', fontsize=9)
        pdf.savefig(fig_text, bbox_inches='tight')
        plt.close(fig_text)
    return pdf_buffer.getvalue()

# --- Interface do Dashboard ---
st.title("📈 Dashboard de Análise de Preços do Cacau")

with st.sidebar:
    st.header("⚙️ Parâmetros de Análise")
    ticker = st.text_input("Ticker do Contrato", "CCU25.NYB")
    ano_corrente = datetime.now().year
    ano_analise = st.number_input("Ano da Análise", min_value=2000, max_value=ano_corrente, value=ano_corrente)
    st.subheader("Fatores da Meta")
    fator_ratio = st.number_input("Fator 1 (ratio)", value=1.18, format="%.2f")
    fator_imposto = st.number_input("Fator 2 (imposto)", value=1.10, format="%.2f")
    debug_cols = st.checkbox("Mostrar colunas retornadas (debug)", value=False)

# --- Lógica Principal da Análise ---
dados_anuais = obter_historico_anual(ticker, ano_analise)

if dados_anuais is None:
    st.error(f"Não foi possível obter dados para o ticker '{ticker}' no ano {ano_analise}. Verifique ticker / conexão.")
else:
    if debug_cols:
        st.write("Colunas retornadas:", list(dados_anuais.columns))
        st.write("Index exemplo:", dados_anuais.index[:3])

    close_series = extrair_close_series(dados_anuais)

    if close_series is None:
        st.error("Não foi possível localizar uma coluna de fechamento (Close) nos dados retornados.")
    else:
        # Garantir que é Series e sem NA no índice
        close_series = close_series.dropna()
        if close_series.empty:
            st.warning("Série de fechamento vazia após remover NA.")
        else:
            # Resample mensal (M = month end); trabalhar com Series para usar to_frame()
            try:
                media_mensal = close_series.resample('M').mean().to_frame(name='Média Mensal')
            except AttributeError:
                # caso (improvável) close_series seja DataFrame, forçar squeeze
                close_series = close_series.squeeze()
                media_mensal = close_series.resample('M').mean().to_frame(name='Média Mensal')

            media_mensal.dropna(inplace=True)

            if media_mensal.empty or len(media_mensal) < 1:
                st.warning("Não há dados mensais suficientes para gerar a análise.")
            else:
                # calcula metas e estatísticas
                media_mensal['Meta'] = media_mensal['Média Mensal'] * fator_ratio * fator_imposto
                media_mensal['Diferença %'] = ((media_mensal['Média Mensal'] / media_mensal['Meta']) - 1) * 100
                estatisticas = media_mensal[['Média Mensal', 'Meta']].describe()

                # Projeção: só se tivermos pelo menos 2 pontos para ajustar regressão linear
                projecao_df = None
                if len(media_mensal) >= 2:
                    try:
                        X = np.arange(len(media_mensal)).reshape(-1, 1)
                        y = media_mensal['Média Mensal'].values
                        modelo = LinearRegression().fit(X, y)
                        futuros_idx = np.arange(len(media_mensal), len(media_mensal) + 3).reshape(-1, 1)
                        predicoes = modelo.predict(futuros_idx)
                        last_date = media_mensal.index[-1]
                        datas_futuras = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=3, freq='M')
                        projecao_df = pd.DataFrame({'Projeção': predicoes}, index=datas_futuras)
                    except Exception as e:
                        st.warning(f"Não foi possível gerar projeção: {e}")

                # --- Exibição de métricas ---
                st.header("Resumo do Mês Atual")
                col1, col2, col3 = st.columns(3)
                mes_atual = media_mensal.iloc[-1]
                col1.metric("Média Mensal Atual", f"${mes_atual['Média Mensal']:.2f}")
                col2.metric("Meta para o Mês", f"${mes_atual['Meta']:.2f}")
                col3.metric("Diferença vs. Meta", f"{mes_atual['Diferença %']:.2f}%", delta_color="inverse")

                # --- Gráfico principal ---
                st.header("Análise Gráfica: Média Mensal vs. Meta e Projeção")
                fig_principal, ax = plt.subplots(figsize=(12, 6))
                ax.plot(media_mensal.index, media_mensal['Meta'], marker='x', linestyle='--', label='Meta', color='purple')
                # desenha segmentos coloridos entre pontos
                for i in range(len(media_mensal) - 1):
                    y_segment = media_mensal['Média Mensal'].iloc[i:i+2]
                    x_segment = media_mensal.index[i:i+2]
                    cor = 'green' if media_mensal['Média Mensal'].iloc[i] <= media_mensal['Meta'].iloc[i] else 'red'
                    ax.plot(x_segment, y_segment, color=cor, marker='o', markersize=5)
                if projecao_df is not None:
                    ax.plot(projecao_df.index, projecao_df['Projeção'], color='orange', linestyle=':', marker='*', markersize=8, label='Projeção (3 Meses)')
                legend_elements = [
                    Line2D([0], [0], color='purple', lw=2, linestyle='--', marker='x', label='Meta'),
                    Line2D([0], [0], color='green', lw=2, marker='o', label='Média ≤ Meta'),
                    Line2D([0], [0], color='red', lw=2, marker='o', label='Média > Meta'),
                ]
                if projecao_df is not None:
                    legend_elements.append(Line2D([0], [0], color='orange', lw=2, linestyle=':', marker='*', label='Projeção'))
                ax.legend(handles=legend_elements, loc='best')
                ax.set_title(f'Análise de Preço do Cacau ({ticker}) - {ano_analise}', fontsize=16)
                ax.set_ylabel('Preço (USD)')
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
                st.pyplot(fig_principal)

                # --- Candlestick últimos 60 dias ---
                st.header("Análise Diária (Últimos 60 dias)")
                try:
                    dados_recentes = dados_anuais.last("60D")
                except Exception:
                    # fallback caso .last("60D") falhe
                    dados_recentes = dados_anuais.iloc[-60:]
                if not dados_recentes.empty and set(['Open','High','Low','Close']).issubset(set(dados_recentes.columns)):
                    fig_candle, axes = mpf.plot(
                        dados_recentes,
                        type='candle',
                        style='yahoo',
                        title="Preços Diários e Médias Móveis",
                        ylabel='Preço (USD)',
                        mav=(10, 20),
                        volume=True,
                        panel_ratios=(3, 1),
                        figsize=(15, 8),
                        returnfig=True
                    )
                    st.pyplot(fig_candle)
                else:
                    st.warning("Dados diários insuficientes para candlestick (falta Open/High/Low/Close ou menos de 1 dia).")

                # --- Tabelas e export ---
                with st.expander("Ver Tabelas Detalhadas"):
                    st.dataframe(media_mensal.style.format("${:.2f}", subset=['Média Mensal', 'Meta'])
                                 .format("{:.2f}%", subset=['Diferença %']))
                    st.dataframe(estatisticas.style.format("{:.2f}"))
                    if projecao_df is not None:
                        st.dataframe(projecao_df.style.format("${:.2f}"))

                st.sidebar.header("Exportar Relatório")
                pdf_data = gerar_relatorio_pdf(media_mensal, estatisticas, projecao_df if projecao_df is not None else pd.DataFrame(), fig_principal, ticker, ano_analise)
                st.sidebar.download_button(
                    label="📄 Baixar Relatório em PDF",
                    data=pdf_data,
                    file_name=f"relatorio_cacau_{ticker}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
