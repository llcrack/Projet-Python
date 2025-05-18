import yfinance as yf
import numpy as np
import pandas as pd
import os
import joblib
import datetime
from zoneinfo import ZoneInfo

#vérification de si on est en script ou environement intéractif
if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

os.chdir(script_dir) #change le répertoire de travail courant vers le dossier du script

start_time = datetime.datetime.now(ZoneInfo("Europe/London")).timestamp()  #temps en timestamp du moment de lancement du programme

#vérification de l'existance des dossiers de stockage sinon création
if not os.path.exists("data"):    
    os.makedirs("data")
if not os.path.exists("data/history"):           
    os.makedirs("data/history")
if not os.path.exists("data/data"):   
    os.makedirs("data/data")

if not os.path.exists("data/history/ftse100.txt"):
    with open("data/history/ftse100.txt","w") as ftse:
        initialisation = "2000-01-01 00:00:00"
        initialisation = np.datetime64(initialisation,'s')
        initialisation = str(initialisation.astype("int64"))
        ftse.write(initialisation)

with open("data/history/ftse100.txt","r") as ftse:
    row = ftse.readlines()
    last_line = int(row[-1].strip())

#si la dernière fois qu'on à téléchargé les données n'était pas aujourd'hui de à partir de 8h utc alors on actualise 
eight_hours = int(3600*8)
if ((last_line-eight_hours)//86400 != (start_time-eight_hours)//86400): 
    tickers = ["AZN.L","HSBA.L","ULVR.L"]                
    ftse_data = yf.download(tickers,period="5y",interval="1d")[["Open","Close"]]
    ftse_indice_data = yf.download("^FTSE", period="5y",interval="1d")[["Open","Close"]]
    sp_indice_vol_data = yf.download("^VIX",period="5y",interval="1d")[["Open","Close"]]
    ftse_data.to_csv("data/data/ftse_data.csv")                   
    ftse_indice_data.to_csv("data/data/ftse_indice_data.csv")
    sp_indice_vol_data.to_csv("data/data/sp_indice_vol_data.csv") 
    with open("data/history/ftse100.txt","a") as ftse:
        note = str(np.int64(start_time))
        ftse.write(f"\n{note}")
else:                                                              
    ftse_data = pd.read_csv("data/data/ftse_data.csv",index_col=[0],header=[0,1])
    ftse_indice_data = pd.read_csv("data/data/ftse_indice_data.csv",index_col=[0],header=[0,1])
    sp_indice_vol_data = pd.read_csv("data/data/sp_indice_vol_data.csv",index_col=[0],header=[0,1])
    ftse_data.index = pd.to_datetime(ftse_data.index)
    ftse_indice_data.index = pd.to_datetime(ftse_indice_data.index)
    sp_indice_vol_data.index = pd.to_datetime(sp_indice_vol_data.index)

#importation des inputs utilisateurs de la web app et adaptation des dataframes en conséquence
user_input = joblib.load("data/data/user_input.joblib")
end_date = user_input["end_date"]
end_date = pd.to_datetime(end_date)
ftse_data = ftse_data.loc[:end_date,:]
ftse_indice_data = ftse_indice_data.loc[:end_date,:]
sp_indice_vol_data = sp_indice_vol_data.loc[:end_date,:]

#création de x_classifier_ftse (dataframe des features) de la même façon que dans le fichier training_ftse.py
ftse_data, sp_indice_vol_data = ftse_data.align(sp_indice_vol_data, join="inner", axis=0)
ftse_data, ftse_indice_data = ftse_data.align(ftse_indice_data, join="inner",axis=0)

ftse_data = ftse_data.iloc[-6:,:]
ftse_indice_data = ftse_indice_data.iloc[-6:,:]
sp_indice_vol_data = sp_indice_vol_data.iloc[-6:,:]

tickers = sorted(set(ftse_data.columns.get_level_values(1)))
for ticker in tickers:
    ftse_data[("log_return",ticker)] = np.log(ftse_data[("Close",ticker)]/
                                                ftse_data[("Open",ticker)])
for ticker in tickers:
    ftse_data[("open_gap_up",ticker)] = np.where(
        ftse_data[("Open",ticker)].shift(-1)>ftse_data[("Close",ticker)]*1.01,1,0
        )
for ticker in tickers:
    ftse_data[("open_gap_down",ticker)] = np.where(
        ftse_data[("Open",ticker)].shift(-1)<ftse_data[("Close",ticker)]*0.99,1,0
        )
ftse_data = ftse_data.iloc[:-1,-9:]

sp_indice_vol_data[("log_return","^VIX")] = np.log(sp_indice_vol_data[("Close","^VIX")]/
                                                sp_indice_vol_data[("Open","^VIX")])
sp_indice_vol_data = sp_indice_vol_data.iloc[:-1,-1]

x_classifier_ftse = ftse_data.merge(sp_indice_vol_data,how="inner",left_index=True,right_index=True)

previous_indice_day = {}
for i in range(0,5):
    previous_indice_day[("shift",f"shift {str(i)}")] = np.log(
        ftse_indice_data[("Close","^FTSE")].shift(i)/ftse_indice_data[("Open","^FTSE")].shift(i)
        )
previous_indice_day[("gap","open_gap_up")] = np.where(
    ftse_indice_data[("Open","^FTSE")].shift(-1)>ftse_indice_data[("Close","^FTSE")]*1.01,1,0
    )
previous_indice_day[("gap","open_gap_down")] = np.where(
    ftse_indice_data[("Open","^FTSE")].shift(-1)<ftse_indice_data[("Close","^FTSE")]*0.99,1,0
    )
previous_indice_day = pd.DataFrame(previous_indice_day, index=ftse_indice_data.index)
previous_indice_day = previous_indice_day.iloc[i:-1,:]

x_classifier_ftse = x_classifier_ftse.iloc[i:,:]
x_classifier_ftse = x_classifier_ftse.to_numpy()
previous_indice_day = previous_indice_day.to_numpy()
x_classifier_ftse = np.concatenate((x_classifier_ftse,previous_indice_day), axis=1)

classifier_model_ftse = joblib.load("data/model/classifier_model_ftse.joblib") #importation des modèles

y_pred = {} #création d'un dictionnaire pour stocker les prédictions de chaque modèle 
for model in classifier_model_ftse.keys():
    y_pred[model] = classifier_model_ftse[model].predict(x_classifier_ftse)

joblib.dump(y_pred,"data/model/y_pred.joblib") #exportation des prédictions qui seront réutilisées dans la web app