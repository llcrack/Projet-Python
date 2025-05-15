import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import time
import os
from io import StringIO
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.ensemble import RandomForestClassifier

start_time = time.time()
if not os.path.exists("data"):    ### Création d'un dossier data et de sous dossiers s'ils n'existent ###
    os.makedirs("data")           ### pas encore afin de stocker les données de chaque actif et       ###
    os.makedirs("data/history")   ### indices ainsi que la dernière fois que le programme a été lancé ###  
    os.makedirs("data/data")
#Téléchargement des données des 500 actions du SP500 ainsi que l'indice lui même
if not os.path.exists("data/history/sp500.txt"):
    with open("data/history/sp500.txt","w") as sp:
        initialisation = "2000-01-01 00:00:00"
        initialisation = np.datetime64(initialisation,'s')
        initialisation = str(initialisation.astype("int64"))
        sp.write(initialisation)
with open("data/history/sp500.txt","r") as sp:
    row = sp.readlines()
    last_line = int(row[-1].strip())
if last_line//86400 != start_time//86400:                 ### On actualise les données via l'API Yahoo Finance ###
    url = "https://stockanalysis.com/list/sp-500-stocks/" ### si la dernière fois que le programme à été lancé ###
    headers = {"user-agent":"Mozilla/5.0"}                ### était un jour antérieur à aujourd'hui            ###
    reponse = requests.get(url, headers=headers)
    tickers = pd.read_html(StringIO(reponse.text))
    tickers = tickers[0]["Symbol"]
    tickers = tickers.to_list()
    if "GOOG" in tickers:                                     ### Sur le site où on récupère les 500 tickers des actions ### 
        tickers.remove("GOOG")                                ### qui composent le S&P500 GOOGLE est présent 2 fois      ### 
    tickers = [ticker.replace(".","-") for ticker in tickers] 
    sp_data = yf.download(tickers,period="3mo",interval="1d")[["Open","Close"]]
    sp_indice_data = yf.download("ES=F", period="3mo",interval="1d")[["Open","Close"]]
    sp_data.to_csv("data/data/sp_data.csv")                   ### Enregistrement des données en csv pour ne pas avoir à  ### 
    sp_indice_data.to_csv("data/data/sp_indice_data.csv")     ### les retélécherger plusieurs fois par jour              ###
    with open("data/history/sp500.txt","a") as sp:
        note = str(np.int64(time.time()))
        sp.write(f"\n{note}")
    print("Données actualisées depuis Yfinance")
else:                                                             ### Dans le cas où la dernière fois qu'on à lancé le   ###
    sp_data = pd.read_csv(                                        ### programme est dans la même journée qu'aujourd'hui, ###
        "data/data/sp_data.csv",                                  ### on importe les données depuis le dossier data/data ###
        index_col=[0],header=[0,1]
        )                
    sp_indice_data = pd.read_csv("data/data/sp_indice_data.csv",index_col=[0],header=[0,1])                                  
    sp_data.index = pd.to_datetime(sp_data.index)                 
    sp_indice_data.index = pd.to_datetime(sp_indice_data.index)
#Téléchargement des données des 40 actions fraançaise au plus grande capitalisation ainsi que du cac40
if not os.path.exists("data/history/cac40.txt"):
    with open("data/history/cac40.txt","w") as cac:
        initialisation = "2000-01-01 00:00:00"
        initialisation = np.datetime64(initialisation,'s')
        initialisation = str(initialisation.astype("int64"))
        cac.write(initialisation)
with open("data/history/cac40.txt","r") as cac:
    row = cac.readlines()
    last_line = int(row[-1].strip())
if last_line//86400 != start_time//86400:                  ### On actualise les données via l'API Yahoo Finance ###
    url = "https://stockanalysis.com/list/euronext-paris/" ### si la dernière fois que le programme à été lancé ###
    headers = {"user-agent":"Mozilla/5.0"}                 ### était un jour antérieur à aujourd'hui            ###
    reponse = requests.get(url, headers=headers)
    tickers = pd.read_html(StringIO(reponse.text))
    tickers = tickers[0]["Symbol"]
    tickers = tickers[0:40]
    tickers = tickers.to_list()
    tickers = [(ticker+".PA") for ticker in tickers]
    cac_data = yf.download(tickers,period="3mo",interval="1d")[["Open","Close"]]
    cac_indice_data = yf.download("^FCHI", period="3mo",interval="1d")[["Open","Close"]]
    cac_data.to_csv("data/data/cac_data.csv")                   ### Enregistrement des données en csv pour ne pas avoir à  ### 
    cac_indice_data.to_csv("data/data/cac_indice_data.csv")     ### les retélécherger plusieurs fois par jour              ###
    with open("data/history/cac40.txt","a") as cac:
        note = str(np.int64(start_time))
        cac.write(f"\n{note}")
    print("Données actualisées depuis Yfinance")
else:                                                              ### Dans le cas où la dernière fois qu'on à lancé le   ###
    cac_data = pd.read_csv(                                        ### programme est dans la même journée qu'aujourd'hui, ###
        "data/data/cac_data.csv",                                  ### on importe les données depuis le dossier data/data ###
        index_col=[0],header=[0,1]
        )                
    cac_indice_data = pd.read_csv("data/data/cac_indice_data.csv",index_col=[0],header=[0,1])                                  
    cac_data.index = pd.to_datetime(cac_data.index)                 
    cac_indice_data.index = pd.to_datetime(cac_indice_data.index)