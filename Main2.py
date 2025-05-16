# Import des librairies nécessaires
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit.components.v1 as components
import os
import joblib
import time
import subprocess

start_time = time.time()

if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

os.chdir(script_dir)

if not os.path.exists("data/model/classifier_model_ftse.joblib"):
    subprocess.run(["python", "training_ftse.py"])
subprocess.run(["python", "Main_2.py"])

y_pred = joblib.load("data/model/y_pred.joblib")
model_accuracy_ftse = joblib.load("data/model/model_accuracy_ftse.joblib")
classification_report_ftse = joblib.load("data/model/classification_report_ftse.joblib")


# ============================
# Paramètres de configuration
# ============================

indices_tickers = {
    "FTSE 100": "^FTSE"
}

tradingview_tickers = {
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

st.title(" Prédiction des indices boursiers")

st.sidebar.header("Paramètres")

# Affichage des prédictions et performances EN PREMIER
st.sidebar.markdown("### Modèle de prédiction")
selected_category = st.sidebar.selectbox(
    "Sélectionner un modèle de prédiction",
    ("Standard", "Équilibré", "Signal")
)

# Mapping pour les clés
model_key_map = {
    "Standard": "standard",
    "Équilibré": "balanced",
    "Signal": "signal"
}
key = model_key_map[selected_category]

# Prédiction du modèle
prediction_series = y_pred[key]
last_prediction = prediction_series[-1]

if last_prediction == 1:
    forecast_message = " Le modèle prévoit une journée **haussière** pour la session d'aujourd'hui."
else:
    forecast_message = " Le modèle prévoit une journée **baissière** pour la session d'aujourd'hui."

st.success(forecast_message)

# Explication des modèles
explanations = {
    "Standard": "**Modèle standard** : Le modèle standard est intéressant car il présente un taux de réussite global élevé,"
    " atteignant 60 %. Cela en fait un bon outil pour capter la tendance dominante du marché.",
    "Équilibré": "**Modèle équilibré** :Le modèle équilibré mérite également une attention particulière : "
    "bien qu'il affiche une performance globale similaire, il parvient à rééquilibrer le rappel (recall) "
    "entre les journées haussières et baissières. Cette caractéristique est précieuse car elle limite les biais du modèle "
    "en évitant de privilégier une seule direction du marché, ce qui permet d'obtenir des prédictions plus fiables "
    "et mieux réparties dans le temps, indépendamment du contexte de marché.",
    "Signal": "**Modèle signal** :  Le modèle signal se distingue par une précision de 67"
    " % sur la classe 1 (journées haussières), contre 50 % sur la classe 0 (journées baissières)."
    " Cela en fait un modèle particulièrement pertinent pour les stratégies axées sur la détection "
    "des mouvements haussiers ou l'optimisation des prises de position longues."
}
st.info(explanations[selected_category])

# Performances
st.markdown("### Performance du modèle")
st.write(f"**Précision :** {round(model_accuracy_ftse[key]*100, 2)}% sur ces 7 dernières années")
report_text = classification_report_ftse[key]

# Création de la figure
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')  # pas d'axe

# Affichage du texte du rapport avec police monospace
ax.text(0, 1, report_text, fontsize=10, fontfamily='monospace', verticalalignment='top')

# Affichage dans Streamlit
st.markdown("### Rapport de classification")
st.pyplot(fig)

# ==== Paramètres secondaires ====
selected_index_name = st.sidebar.selectbox("Sélectionner un indice", list(indices_tickers.keys()))
selected_index_ticker = indices_tickers[selected_index_name]
selected_tradingview_ticker = tradingview_tickers[selected_index_name]

start_date = st.sidebar.date_input(
                                    'Date de début', pd.to_datetime('2015-01-01'),
                                    min_value=pd.to_datetime("2015-01-01").date(),
                                    max_value=pd.to_datetime("today").date()
                                                                                )
end_date = st.sidebar.date_input(
                                    'Date de fin', pd.to_datetime('today'),
                                    min_value=pd.to_datetime("2015-01-01").date(),
                                    max_value=pd.to_datetime("today").date()
                                                                                )

realtime_analysis_choice = st.sidebar.selectbox(
    "Analyse en temps réel",
    ("Aperçu", "Analyse graphique")
)

# ==============================
# Récupération des données boursières
# ==============================
with open("data/history/ftse100.txt","r") as ftse:
    row = ftse.readlines()
    last_line = int(row[-1].strip())

if last_line//86400 != start_time//86400: 
    data = yf.download(selected_index_ticker, start="2015-01-01")
    data.to_csv("data/data/web_app_ftse_data.csv")

try:
    data = pd.read_csv("data/data/web_app_ftse_data.csv",index_col=[0],header=[0,1])
    data.index = pd.to_datetime(data.index)
except:
    data = yf.download(selected_index_ticker, start="2015-01-01")
    data.to_csv("data/data/web_app_ftse_data.csv")

index_data = data.copy()
index_data = index_data.loc[start_date:end_date,:]
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
    if end_date >= pd.to_datetime("today").date():
        st.write(f"RSI actuel = {latest_rsi:.2f}")
    else:
        st.write(f"RSI à la date du {end_date.strftime('%Y-%m-%d')} = {latest_rsi:.2f}")

    st.subheader(f"Données historiques — {selected_index_name}")
    st.dataframe(index_data.iloc[::-1])

    # Moyennes des 20 et 5 derniers jours
    mean_20 = index_data['Close'].iloc[-20:].mean()
    mean_5 = index_data['Close'].iloc[-5:].mean()

    st.info(
        f"**Prix moyen sur 20 jours (1 mois de cotation) :** {round(mean_20.values[0],2)}\n\n"
        f"**Prix moyen sur 5 jours (1 semaine de cotation) :** {round(mean_5.values[0],2)}"
    )

    # Graphique RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=index_data.index,
        y=index_data['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='purple')
    ))

    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")

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
