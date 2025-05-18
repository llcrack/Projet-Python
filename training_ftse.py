import yfinance as yf
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#vérification de si on est en script ou environement intéractif
if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

os.chdir(script_dir) #change le répertoire de travail courant vers le dossier du script

#vérification de l'existance du modèle sinon entrainement de celui-ci
if not (    os.path.exists("data/model/classifier_model_ftse.joblib") and
            os.path.exists("data/model/model_accuracy_ftse.joblib") and
            os.path.exists("data/model/classification_report_ftse.joblib") and
            os.path.exists("data/training_data_x_classifier/x_classifier_ftse.csv")   ):

    #vérification de l'existance des dossiers de stockage sinon création
    if not os.path.exists("data"):    
        os.makedirs("data")
    if not os.path.exists("data/history"):           
        os.makedirs("data/history")
    if not os.path.exists("data/data"):   
        os.makedirs("data/data")
    if not os.path.exists("data/training_data"):
        os.makedirs("data/training_data")
    if not os.path.exists("data/model"):
        os.makedirs("data/model")
    if not os.path.exists("data/training_data_x_classifier"):
        os.makedirs("data/training_data_x_classifier")

    #vérification de l'existance des données d'entrainements sinon on les télécharge via l'api yfinance puis on les stocks
    if not (    os.path.exists("data/training_data/training_ftse.csv") and
                os.path.exists("data/training_data/training_indice_ftse.csv")   ):
        
        tickers = ["AZN.L","HSBA.L","ULVR.L"]                
        training_ftse = yf.download(tickers,start="2001-01-01",end="2024-12-31",interval="1d")[["Open","Close"]]
        training_indice_ftse = yf.download("^FTSE",start="2001-01-01",end="2024-12-31",interval="1d")[["Open","Close"]]
        training_ftse.to_csv("data/training_data/training_ftse.csv")
        training_indice_ftse.to_csv("data/training_data/training_indice_ftse.csv")

    training_ftse = pd.read_csv("data/training_data/training_ftse.csv",index_col=[0],header=[0,1])
    training_indice_ftse = pd.read_csv("data/training_data/training_indice_ftse.csv",index_col=[0],header=[0,1])
    training_ftse.index = pd.to_datetime(training_ftse.index)
    training_indice_ftse.index = pd.to_datetime(training_indice_ftse.index)


    if not os.path.exists("data/training_data/training_indice_sp_vol.csv"):
        training_indice_sp_vol = yf.download("^VIX",start="2001-01-01",end="2024-12-31",interval="1d")[["Open","Close"]]
        training_indice_sp_vol.to_csv("data/training_data/training_indice_sp_vol.csv")
    training_indice_sp_vol = pd.read_csv( 
        "data/training_data/training_indice_sp_vol.csv",index_col=[0],header=[0,1])
    training_indice_sp_vol.index = pd.to_datetime(training_indice_sp_vol.index)

    #alignement des 3 dataframes
    training_ftse, training_indice_sp_vol = training_ftse.align(training_indice_sp_vol, join="inner", axis=0)
    training_ftse, training_indice_ftse = training_ftse.align(training_indice_ftse, join="inner",axis=0)

    #créations des features à partir des 3 dataframes précédent
    tickers = sorted(set(training_ftse.columns.get_level_values(1))) #pour chacun des 3 actifs création de colonnes rendements logarithmique
    for ticker in tickers:
        training_ftse[("log_return",ticker)] = np.log(training_ftse[("Close",ticker)]/
                                                    training_ftse[("Open",ticker)])
    for ticker in tickers:                                                    #détection de gap haussier à l'ouverture
        training_ftse[("open_gap_up",ticker)] = np.where(
            training_ftse[("Open",ticker)].shift(-1)>training_ftse[("Close",ticker)]*1.01,1,0
            )
    for ticker in tickers:                                                     #détection de gap baissier à l'ouverture
        training_ftse[("open_gap_down",ticker)] = np.where(
            training_ftse[("Open",ticker)].shift(-1)<training_ftse[("Close",ticker)]*0.99,1,0
            )
    training_ftse = training_ftse.iloc[:,-9:]                 #on ne garde que les 9 colonnes nouvellement crée en feature

    training_indice_sp_vol[("log_return","^VIX")] = np.log(training_indice_sp_vol[("Close","^VIX")]/ #création d'un colonne de rendement            #
                                                    training_indice_sp_vol[("Open","^VIX")])         #logarithmique sur le VIX (volatilité du sp500)#
    training_indice_sp_vol = training_indice_sp_vol.iloc[:,-1] #on ne garde que cette colonne en feature pour le VIX

    x_classifier_ftse = training_ftse.merge(training_indice_sp_vol,how="inner",left_index=True,right_index=True) #création du dataframe X qui contiendra l'ensemble des observation et feature nécéssaire à l'entrainement du modèle
    previous_indice_day = {} #création d'un dictionnaire qui contiendra les rendements logarithmiques du footsie100 sur 5 jours#
    for i in range(0,5):     #ainsi qu'une détection des gap haussier et baissier à l'ouverture                                #
        previous_indice_day[("shift",f"shift {str(i)}")] = np.log(
            training_indice_ftse[("Close","^FTSE")].shift(i)/training_indice_ftse[("Open","^FTSE")].shift(i)
            )
    previous_indice_day[("gap","open_gap_up")] = np.where(
        training_indice_ftse[("Open","^FTSE")].shift(-1)>training_indice_ftse[("Close","^FTSE")]*1.01,1,0
        )
    previous_indice_day[("gap","open_gap_down")] = np.where(
        training_indice_ftse[("Open","^FTSE")].shift(-1)<training_indice_ftse[("Close","^FTSE")]*0.99,1,0
        )
    previous_indice_day = pd.DataFrame(previous_indice_day, index=training_indice_ftse.index)
    x_classifier_ftse_csv = x_classifier_ftse.merge(previous_indice_day,how="inner",left_index=True,right_index=True) #Création et exportation en csv du dataframe X qui sera sur Drive
    x_classifier_ftse_csv.to_csv("data/training_data_x_classifier/x_classifier_ftse.csv")
    previous_indice_day = previous_indice_day.iloc[i:-2,:]
    x_classifier_ftse = x_classifier_ftse.iloc[i:-2,:]
    x_classifier_ftse = x_classifier_ftse.to_numpy()
    previous_indice_day = previous_indice_day.to_numpy()
    x_classifier_ftse = np.concatenate((x_classifier_ftse,previous_indice_day), axis=1) #Complétation du Dataframe des features
    y_classifier_ftse = np.where(training_indice_ftse[("Close", "^FTSE")].shift(-1)>
    training_indice_ftse[("Open", "^FTSE")].shift(-1),1,0) #création de l'array Y target
    y_classifier_ftse = y_classifier_ftse[i:-2]


    class_weight_dict = {   "standard":None,
                            "balanced":"balanced",
                            "signal":{0:1.7,1:1.0}  } #entrainement de 3 modèle selon le paramètre class_weight de RandomForestClassifier
    classifier_model_ftse = {} #Dictionnaire pour stocker les 3 modèles
    model_accuracy_ftse = {} #pour stocker la précisions des 3 modèles
    classification_report_ftse = {} #pour stocker les 3 rapports
    x_train, x_test, y_train, y_test = train_test_split(x_classifier_ftse, y_classifier_ftse, test_size=0.3, shuffle=False) #on réserve 30% des observations pour tester les performance des modèles
    for name,weight in class_weight_dict.items():
        model = RandomForestClassifier( n_estimators=4000,
                                        max_depth=10,
                                        min_samples_split=10,
                                        min_samples_leaf=4,
                                        max_features='sqrt',
                                        class_weight=weight,
                                        n_jobs=-1   )
        model = model.fit(x_train,y_train)
        classifier_model_ftse[name] = model
        y_pred = classifier_model_ftse[name].predict(x_test)
        model_accuracy_ftse[name] = accuracy_score(y_test, y_pred)
        classification_report_ftse[name] = classification_report(y_test, y_pred)
    joblib.dump(classifier_model_ftse, "data/model/classifier_model_ftse.joblib") #stockage des modèles et de leurs performance
    joblib.dump(model_accuracy_ftse, "data/model/model_accuracy_ftse.joblib")
    joblib.dump(classification_report_ftse, "data/model/classification_report_ftse.joblib")