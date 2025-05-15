# 📦 Import des librairies nécessaires
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ============================
# 📌 Paramètres de configuration
# ============================

indices_tickers = {
    "S&P 500": "^GSPC",
    "DAX": "^GDAXI",
    "CAC 40": "^FCHI",
    "FTSE 100": "^FTSE"
}

tradingview_tickers = {
    "S&P 500": "OANDA:SPX500USD",
    "DAX": "OANDA:DE30EUR",
    "CAC 40": "OANDA:FR40EUR",
    "FTSE 100": "OANDA:UK100GBP"
}

# ================================
# Fonction utilitaire : Calcul RSI
# ================================
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# =========================
# Interface de l'application
# =========================

st.title("📈 Prédiction des indices boursiers")

st.sidebar.header("Paramètres")

selected_index_name = st.sidebar.selectbox("Sélectionner un indice", list(indices_tickers.keys()))
selected_index_ticker = indices_tickers[selected_index_name]
selected_tradingview_ticker = tradingview_tickers[selected_index_name]

start_date = st.sidebar.date_input('Date de début', pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input('Date de fin', pd.to_datetime('today'))

realtime_analysis_choice = st.sidebar.selectbox(
    "Analyse en temps réel",
    ("Aperçu", "Analyse graphique")
)

selected_category = st.sidebar.selectbox(
    "Sélectionner un modèle de prédiction",
    ("Standard", "Équilibré", "Signal")
)

# ==============================
# Récupération des données boursières
# ==============================

index_data = yf.download(selected_index_ticker, start=start_date, end=end_date)
index_data['RSI'] = compute_rsi(index_data)

# ===========================
# Affichage selon le choix utilisateur
# ===========================

if realtime_analysis_choice == "Aperçu":
    latest_rsi = index_data['RSI'].iloc[-1]

    if latest_rsi > 70:
        sentiment = "🔴 Surachat (chercher la vente)"
    elif latest_rsi < 30:
        sentiment = "🟢 Survente (chercher l'achat)"
    else:
        sentiment = "🟡 Neutre"

    st.subheader(f"Sentiment actuel du marché : {sentiment}")
    st.write(f"RSI actuel : {latest_rsi:.2f}")

    st.subheader(f"Données historiques — {selected_index_name}")
    st.dataframe(index_data.iloc[::-1])

    # Graphique matplotlib du prix de clôture
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(index_data.index, index_data['Close'], label='Prix de clôture')
    ax.set_title(f"Évolution du prix : {selected_index_name}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Prix')
    ax.legend()
    st.pyplot(fig)

    # Graphique interactif RSI (Plotly)
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=index_data.index,
        y=index_data['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='purple')
    ))

    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",)
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",)

    fig_rsi.update_layout(
        title=f"RSI interactif : {selected_index_name}",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=450,
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig_rsi, use_container_width=True)

elif realtime_analysis_choice == "Analyse graphique":
    st.subheader(f"Graphique interactif en temps réel — {selected_index_name}")

    st.write("""
    ⚠️ **Note importante :**  
    Les graphiques interactifs ci-dessous utilisent les tickers de TradingView via **OANDA CFD**.  
    Les prix peuvent différer de ceux récupérés via Yahoo Finance car :
    - **Yahoo Finance** fournit des données différées (officielles)
    - **OANDA CFD** reflète le marché dérivé en continu.

    👉 Utilisez cet outil pour une **analyse graphique en direct**.
    """)

    tradingview_widget = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
      "width": "100%",
      "height": 620,
      "symbol": "{selected_tradingview_ticker}",
      "interval": "D",
      "timezone": "Etc/UTC",
      "theme": "light",
      "style": "1",
      "locale": "fr",
      "toolbar_bg": "#f1f3f6",
      "enable_publishing": false,
      "hide_side_toolbar": false,
      "allow_symbol_change": true,
      "container_id": "tradingview_widget"
      }});
      </script>
    </div>
    """
    components.html(tradingview_widget, height=640)

# ================================
# Affichage du modèle de prédiction sélectionné
# ================================

st.subheader(f"Page {selected_category}")
st.write(f"📌 Ici tu pourras afficher ce que tu veux pour la catégorie **{selected_category}**.")
