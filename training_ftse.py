import yfinance as yf
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

os.chdir(script_dir)

if not (    os.path.exists("data/model/classifier_model_ftse.joblib") and
            os.path.exists("data/model/model_accuracy_ftse.joblib") and
            os.path.exists("data/model/classification_report_ftse.joblib") and
            os.path.exists("data/training_data_x_classifier/x_classifier_ftse.csv")   ):


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
    print(  "Données d'entrainement importé depuis ./data/training_data/training_ftse.csv"
            " et ./data/training_data/training_indice_ftse.csv" )

    if not os.path.exists("data/training_data/training_indice_sp_vol.csv"):
        training_indice_sp_vol = yf.download("^VIX",start="2001-01-01",end="2024-12-31",interval="1d")[["Open","Close"]]
        training_indice_sp_vol.to_csv("data/training_data/training_indice_sp_vol.csv")
    training_indice_sp_vol = pd.read_csv( 
        "data/training_data/training_indice_sp_vol.csv",index_col=[0],header=[0,1])
    training_indice_sp_vol.index = pd.to_datetime(training_indice_sp_vol.index)

    training_ftse, training_indice_sp_vol = training_ftse.align(training_indice_sp_vol, join="inner", axis=0)
    training_ftse, training_indice_ftse = training_ftse.align(training_indice_ftse, join="inner",axis=0)

    tickers = sorted(set(training_ftse.columns.get_level_values(1)))
    for ticker in tickers:
        training_ftse[("log_return",ticker)] = np.log(training_ftse[("Close",ticker)]/
                                                    training_ftse[("Open",ticker)])
    for ticker in tickers:
        training_ftse[("open_gap_up",ticker)] = np.where(
            training_ftse[("Open",ticker)].shift(-1)>training_ftse[("Close",ticker)]*1.01,1,0
            )
    for ticker in tickers:
        training_ftse[("open_gap_down",ticker)] = np.where(
            training_ftse[("Open",ticker)].shift(-1)<training_ftse[("Close",ticker)]*0.99,1,0
            )
    training_ftse = training_ftse.iloc[:,-9:]

    training_indice_sp_vol[("log_return","^VIX")] = np.log(training_indice_sp_vol[("Close","^VIX")]/
                                                    training_indice_sp_vol[("Open","^VIX")])
    training_indice_sp_vol = training_indice_sp_vol.iloc[:,-1]

    x_classifier_ftse = training_ftse.merge(training_indice_sp_vol,how="inner",left_index=True,right_index=True)
    previous_indice_day = {}
    for i in range(0,5):
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
    x_classifier_ftse_csv = x_classifier_ftse.merge(previous_indice_day,how="inner",left_index=True,right_index=True)
    x_classifier_ftse_csv.to_csv("data/training_data_x_classifier/x_classifier_ftse.csv")
    previous_indice_day = previous_indice_day.iloc[i:-2,:]
    x_classifier_ftse = x_classifier_ftse.iloc[i:-2,:]
    x_classifier_ftse = x_classifier_ftse.to_numpy()
    previous_indice_day = previous_indice_day.to_numpy()
    x_classifier_ftse = np.concatenate((x_classifier_ftse,previous_indice_day), axis=1)
    y_classifier_ftse = np.where(training_indice_ftse[("Close", "^FTSE")].shift(-1)>
    training_indice_ftse[("Open", "^FTSE")].shift(-1),1,0)
    y_classifier_ftse = y_classifier_ftse[i:-2]


    class_weight_dict = {   "standard":None,
                            "balanced":"balanced",
                            "signal":{0:1.7,1:1.0}  }
    classifier_model_ftse = {}
    model_accuracy_ftse = {}
    classification_report_ftse = {}
    x_train, x_test, y_train, y_test = train_test_split(x_classifier_ftse, y_classifier_ftse, test_size=0.3, shuffle=False)
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
        print(f"La précision du modèle {name} est de {round(model_accuracy_ftse[name]*100,3)} %")
        classification_report_ftse[name] = classification_report(y_test, y_pred)
        print(classification_report_ftse[name])
    joblib.dump(classifier_model_ftse, "data/model/classifier_model_ftse.joblib")
    joblib.dump(model_accuracy_ftse, "data/model/model_accuracy_ftse.joblib")
    joblib.dump(classification_report_ftse, "data/model/classification_report_ftse.joblib")