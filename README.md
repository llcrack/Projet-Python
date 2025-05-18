# 📈 Prédiction quotidienne du FTSE 100 avec Streamlit et Machine Learning

## 🎯 Objectif du projet

Ce projet a pour but de développer une application web en Python (via Streamlit) capable de :

- Prédire la direction journalière de l’indice boursier **FTSE 100** (hausse ou baisse),
- Dépasser un taux de réussite naïf de 50 % (tirage aléatoire) qui constitue un enjeu majeur en finance de marché,
- En se basant sur un modèle de **Machine Learning** entraîné en amont,
- Tout en proposant une **interface utilisateur interactive et informative**.

L'utilisateur peut sélectionner des paramètres temporels, consulter le **RSI** de l’indice sur une période choisie, et afficher le **graphique temps réel de TradingView**.

---

## 📊 Choix du dataset

Le projet utilise des données financières accessibles librement via **Yahoo Finance (yfinance)**. Les sources incluent :

- 3 des 100 actions du FTSE 100 : **AstraZeneca (AZN.L), HSBC (HSBA.L), Unilever (ULVR.L)**, composantes majeures du FTSE 100,
- L’indice FTSE 100 lui-même (`^FTSE`),
- L’indice de volatilité **VIX (`^VIX`)**, en tant qu’indicateur exogène global.    
- 👉 Ces données permettent de construire des features à la fois micro (comportement de titres individuels) et macro (volatilité, tendance de l’indice) pour améliorer la précision des prédictions.

---

## 🧠 Choix du modèle

Nous avons utilisé un **RandomForestClassifier** de la bibliothèque `scikit-learn` pour plusieurs raisons :

- Il **gère bien les données tabulaires** avec des relations non linéaires.
- Il est **robuste au surapprentissage**, surtout avec des hyperparamètres bien choisis.
- Il fournit une **bonne interprétabilité** (via l’importance des features).
- Il s'adapte bien à des datasets de taille moyenne comme ici (moins de 10 000 lignes après traitement).

Nous avons entraîné **trois variantes du modèle** selon des objectifs différents :

- `standard` : modèle classique sans pondération des classes.  
- `balanced` : pondération automatique pour gérer un éventuel déséquilibre haussiers/baissiers.  
- `signal` : favorise la détection des journées baissières et augmente la précision des signaux haussiers.

👉 Ce choix permet à l’utilisateur de sélectionner le modèle le plus adapté à sa stratégie : **prédiction globale** ou **détection d'opportunité long/short**.

## 🖥️ Fonctionnement de l’application

L'application Streamlit est organisée en **3 modules** :

### `training_ftse.py`

- Télécharge les données brutes historiques,
- Prépare les features et labels,
- Entraîne le modèle avec `train_test_split`,
- Sauvegarde le modèle et les performances avec `joblib`.

### `prediction.py`

- Reçoit les inputs utilisateurs de la web app (date de fin),
- Effectue les prédictions pour chaque modèle et renvoie les résultats à la web app.

### `web_app.py`

Propose une interface utilisateur avec **sidebar** :

- Choix du modèle (`standard`, `équilibré`, `signal`),
- Sélection de l’intervalle temporel,
- Choix du type d’analyse (**RSI** ou **graphique interactif**),
- Charge les modèles et affiche la prédiction du jour choisi.

Affiche également :

- Une carte **RSI dynamique** avec seuils d’achat/vente,
- Un résumé des performances (**classification report**),
- Un **graphique interactif** via TradingView.

---

## 📌 L’application est optimisée pour :

- Éviter les rechargements inutiles si l’input utilisateur n’a pas changé,
- Ne faire la prédiction d'aujourd'hui qu’après **8h (heure de Londres)**, correspondant à l'ouverture de la place boursière londonienne,
- Adapter automatiquement les dates si un **week-end** ou **jour férié** est sélectionné.

---

## ✅ Fonctionnalités

- ✔️ Prédiction boursière FTSE avec **Random Forest**
- ✔️ Choix du modèle et période d’analyse
- ✔️ RSI **graphique interactif**
- ✔️ **TradingView** intégré
- ✔️ Optimisation du chargement des données
- ✔️ Contrôle temporel intelligent (**UTC, jours fériés**)

---

## 🚫 Limitations connues

- L'application suppose que le marché britannique est fermé les **week-ends** et certains **jours fériés UK**.
- Les données sont récupérées via **yfinance** : en cas de surcharge API ou d’erreur réseau, une exception peut survenir.

---

## 📁 Structure du projet

```
Projet-Python/
│
├── web_app.py                      # Interface Streamlit
├── prediction.py                   # Script de prédiction
├── training_ftse.py                # Entraînement des modèles
│
├── requirements.txt                # Liste des dépendances
├── README.md                       # Présentation du projet
│
└── data/
    ├── data/                       # Données FTSE actuelles
    ├── model/                      # Modèles et résultats sauvegardés
    ├── history/                    # Timestamps et historiques
    ├── training_data/              # Données brutes d'entraînement
    └── training_data_x_classifier/ # Features finales utilisées pour le modèle
```

---

## 📹 Démonstration vidéo

La vidéo de démonstration est fournie dans le dossier Google Drive :  
👉 https://drive.google.com/drive/folders/1mEXAjKg-vCgXeArxqmXDLxjJt7pAo05m?usp=drive_link

---

## ⚙️ Lancer le projet

```bash
git clone https://github.com/yao-trz/Projet-Python.git
cd Projet-Python
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
streamlit run web_app.py
```
### 👨‍💻 Auteurs

Ce projet a été réalisé en autonomie par :

- **Yao TREZISE**  
- **Komi TREZISE**  
- **Mamadou Cherif DIALLO**

Dans le cadre d’un projet académique de prédiction supervisée avec **Streamlit**,  
portant sur l’analyse et la prévision de l’indice boursier **FTSE 100**.
