# Import des librairies nécessaires
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit.components.v1 as components
import os
import joblib
import subprocess
import sys
import datetime
from zoneinfo import ZoneInfo

start_time = datetime.datetime.now(ZoneInfo("Europe/London")).timestamp()  #temps en timestamp du moment de lancement du programme

#vérification de si on est en script ou environement intéractif
if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__)) #récupération du chemin du dossier parent de ce fichier
else:
    script_dir = os.getcwd() #récupération du répertoire de travail actuel

os.chdir(script_dir) #change le répertoire de travail courant vers le dossier du script

#véification de l'existance des modèles
if not os.path.exists("data/model/classifier_model_ftse.joblib"):
    print("Premier lancement : entrainement des modèles (~1min)")
    subprocess.run([sys.executable, "training_ftse.py"]) #si ils n'existent pas on lance le fichier d'entrainement des modèles
    print(  "Les modèles ont été entrainés avec succès et stockés à l'adresse suivante :"
            " ./data/model/classifier_model_ftse.joblib "   )

#paramètres de configuration
indices_tickers = {"FTSE 100": "^FTSE"}

tradingview_tickers = {"FTSE 100": "OANDA:UK100GBP"}

#configuration de la sidebar
st.sidebar.header("Paramètres")
selected_index_name = st.sidebar.selectbox("Sélectionner un indice", list(indices_tickers.keys()))
selected_index_ticker = indices_tickers[selected_index_name]
selected_tradingview_ticker = tradingview_tickers[selected_index_name]

five_years_ago = (pd.to_datetime("today") - pd.DateOffset(years=5)).date()
five_years_ago_plus_14 = (five_years_ago + pd.Timedelta(days=14))
start_date = st.sidebar.date_input( 'Date de début', five_years_ago,
                                    min_value=five_years_ago,
                                    max_value=pd.to_datetime("today").date()    )

end_date = st.sidebar.date_input(   'Date de fin', pd.to_datetime('today'),
                                    min_value=five_years_ago_plus_14,
                                    max_value=pd.to_datetime("today").date()    )

realtime_analysis_choice = st.sidebar.selectbox(    "Analyse",
                                                    ("Relative Strength Index", "Graphique TradingView")    )

#fonction pour éviter les week-ends
def adjust_to_last_friday(date):
    weekday = date.weekday()
    if weekday == 5:      # samedi
        return date - pd.Timedelta(days=1)
    elif weekday == 6:    # dimanche
        return date - pd.Timedelta(days=2)
    return date
end_date = adjust_to_last_friday(end_date)

#stokage des inputs utilisateurs influant sur les features du modèle
user_input = {}
user_input["start_date"] = start_date
user_input["end_date"] = end_date
joblib.dump(user_input, "data/data/user_input.joblib")

#optimisation de l'experience utilisateur en lançant le programme de prédiction seulement si les inputs utilisateurs changent
if not os.path.exists("data/history/web_app_user_input.txt"): #si le fichier .txt d'historique des inputs n'existe pas encore
    with open("data/history/web_app_user_input.txt","w") as ipt: #on le crée et on stock en string la date de fin
        ipt.write(str(end_date))
    subprocess.run([sys.executable, "prediction.py"]) #lancement du programme de prediction
else:
    with open("data/history/web_app_user_input.txt","r") as ipt: #si le fichier .txt d'historique des inputs existe
        row = ipt.readlines()
        last_line = row[-1].strip() #on récupère la dernière date de fin en passé en input
        if not (last_line == str(end_date)): #si elle n'est pas équivalente à celle actuellement en input
            with open("data/history/web_app_user_input.txt", "w") as ipt:
                ipt.write(str(end_date)) #on actualise le fichier d'historique
            subprocess.run([sys.executable, "prediction.py"]) #puis on lance le programme de prédiction

y_pred = joblib.load("data/model/y_pred.joblib") #on récupère les prédictions correspondantes aux inputs actuels 
model_accuracy_ftse = joblib.load("data/model/model_accuracy_ftse.joblib")
classification_report_ftse = joblib.load("data/model/classification_report_ftse.joblib")

#fonction calcul du rsi:
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

#interface de la webapp
st.info("Ajuster la date de fin dans la side bar afin d'obtenir les résultats de prévision sur une séance passé")
st.warning( "Si la date de fin correspond à un jour du week-end la prédiction correspond à celle du vendredi précédent "
            "étant données l'abscence de cotation les week-ends. Similairement pour les jours fériés anglais." )
st.warning("Une journée commence à 8h utc et se termine à 16h30 utc")
st.title(" Prédiction de l'indice boursier")

#affichage des prédictions et performances
st.sidebar.markdown("### Modèle de prédiction")
selected_category = st.sidebar.selectbox(   "Sélectionner un modèle de prédiction",
                                            ("Standard", "Équilibré", "Signal") )

#mapping pour les clés
model_key_map = {   "Standard": "standard",
                    "Équilibré": "balanced",
                    "Signal": "signal"  }

key = model_key_map[selected_category]

#prédiction du modèle
prediction_series = y_pred[key]
last_prediction = prediction_series[-1]
#comparaison temporelle
now_uk = datetime.datetime.now(ZoneInfo("Europe/London"))
if end_date == pd.to_datetime("today").date(): #si la date de fin dans la sidebar correspond à aujourd'hui sinon
    if not now_uk.hour < 8: #si il n'est pas moins de 8h (heure de Londres) on affiche la prédiction
        if last_prediction == 1:
            forecast_message = " Le modèle prévoit une journée **haussière** pour la session d'aujourd'hui."
        else:
            forecast_message = " Le modèle prévoit une journée **baissière** pour la session d'aujourd'hui."
        st.success(forecast_message)
    else: #sinon message d'erreur
        st.error(   "Il faut attendre l'ouverture de la place boursière de Londres à 8h (heure de Londres) "
                    "pour obtenir une prédiction pour la journée d'aujourd'hui" )
else:
    if last_prediction == 1:
        forecast_message = f" Le modèle prévoyait une journée **haussière** pour la session du {end_date}."
    else:
        forecast_message = f" Le modèle prévoyait une journée **baissière** pour la session du {end_date}"  
    st.success(forecast_message)

#explication des modèles
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
    "des mouvements haussiers ou l'optimisation des prises de position long."
}
st.info(explanations[selected_category])

#performances
st.markdown("### Performance du modèle")
st.write(f"**Précision :** {round(model_accuracy_ftse[key]*100, 2)}% sur ces 7 dernières années")
report_text = classification_report_ftse[key]

#création de la figure faisant apparaitre le report
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off') 

ax.text(0, 1, report_text, fontsize=10, fontfamily='monospace', verticalalignment='top')
#affichage dans Streamlit
st.markdown("### Rapport de classification")
st.pyplot(fig)

#optimisation de la récupération des données boursières
with open("data/history/ftse100.txt","r") as ftse: #ouverture du fichier .txt d'historique des données
    row = ftse.readlines()
    last_line = int(row[-1].strip())

#si la dernière fois qu'on à téléchargé les données n'était pas aujourd'hui de à partir de 8h utc alors on actualise 
eight_hours = int(3600*8)
if ((last_line-eight_hours)//86400 != (start_time-eight_hours)//86400):  
    data = yf.download(selected_index_ticker, period="5y",interval="1d")
    data.to_csv("data/data/web_app_ftse_data.csv")
else:
    data = pd.read_csv("data/data/web_app_ftse_data.csv",index_col=[0],header=[0,1])
    data.index = pd.to_datetime(data.index)

#ajout d'une colonne RSI en utilisant la fonction compute_rsi crée précédemment
index_data = data.copy()
index_data = index_data.loc[start_date:end_date,:]
index_data['RSI'] = compute_rsi(index_data)

# Affichage selon le choix utilisateur

#si "Relative Strength Index" est choisi dans la side bar
if realtime_analysis_choice == "Relative Strength Index":
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

    st.info(    f"**Prix moyen sur 20 jours (1 mois de cotation) :** {round(mean_20.values[0],2)}\n\n"
                f"**Prix moyen sur 5 jours (1 semaine de cotation) :** {round(mean_5.values[0],2)}" )

    #graphique du RSI
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

#si "Graphique TradingView" est choisi dans la side bar
elif realtime_analysis_choice == "Graphique TradingView":
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