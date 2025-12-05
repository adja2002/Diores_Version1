# Système
import os
import sys
import time

# Built-in imports
import warnings
from collections import Counter
import pickle

# Data manipulation
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import missingno

# Sklearn imports
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    mean_absolute_error
)

# ML Models - Linear
from sklearn.linear_model import (
    LogisticRegression,
    Perceptron,
    SGDClassifier,
    Lasso,
    LassoCV
)

# ML Models - SVM
from sklearn.svm import SVC, LinearSVC

# ML Models - Ensemble
from sklearn.ensemble import RandomForestClassifier

# ML Models - Other
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Disable warnings
warnings.filterwarnings('ignore')

# ///////////////////////////////////////////////////////////////////////////////////////////

import pickle
from sklearn.base import BaseEstimator, ClassifierMixin

class DioresPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self):
        # Mappings pour interpréter les prédictions
        self.resultat_map = {0: 'NON ADMIS', 1: 'AUTORISE', 2: 'PASSE'}
        self.session_map = {0: 'Deuxième Session', 1: 'Première Session'}
        self.mention_map = {0: 'Passable', 1: 'Assez-Bien', 2: 'Bien', 3: 'Très-Bien'}
        print("========================== Initialisation ==============================")

        # Charger les modèles pré-entrainés
        with open('Models/V2/admi_non_admi_best_model_DecisionTree.pkl', 'rb') as f:
            self.model_admi = pickle.load(f)

        with open('Models/V2/session_best_model_DecisionTree.pkl', 'rb') as f:
            self.model_session = pickle.load(f)

        with open('Models/V2/mention_best_model_DecisionTree.pkl', 'rb') as f:
            self.model_mention = pickle.load(f)

    def fit(self, X, y=None):
        # Cette méthode est incluse pour respecter l'interface scikit-learn
        return self

    def predict(self, X):
        resultats = []
        admissions = self.model_admi.predict(X)

        for i, est_admis in enumeµrate(admissions):
            if est_admis == 0:  # NON ADMIS
                resultats.append({
                    'admission': self.resultat_map[0],
                    'session': None,
                    'mention': None
                })
            else:  # AUTORISE ou PASSE
                session_pred = self.model_session.predict([X.iloc[i]])[0]
                if session_pred == 0:  # Deuxième Session
                    resultats.append({
                        'admission': self.resultat_map[est_admis],
                        'session': self.session_map[0],
                        'mention': None
                    })
                else:  # Première Session
                    mention_pred = self.model_mention.predict([X.iloc[i]])[0]
                    resultats.append({
                        'admission': self.resultat_map[est_admis],
                        'session': self.session_map[1],
                        'mention': self.mention_map[mention_pred]
                    })
        return resultats

    def score(self, X, y):
        predictions = self.predict(X)
        correct = 0
        total = len(y)

        for pred, true in zip(predictions, y):
            if (pred['admission'] == self.resultat_map[true['admission']] and
                pred['session'] == (self.session_map[true['session']] if true['session'] is not None else None) and
                pred['mention'] == (self.mention_map[true['mention']] if true['mention'] is not None else None)):
                correct += 1
        return correct / total


import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataFrameProcessor:
   def __init__(self, df):
       self.df = df.copy()
       self.features = [
           'Année BAC', 'Nbre Fois au BAC', 'Groupe Résultat', 'Moy. nde',
           'Moy. ère', 'Moy. S Term.', 'Moy. S Term..1', 'MATH', 'SCPH', 'FR',
           'PHILO', 'AN', 'Tot. Pts au Grp.', 'Moyenne au Grp.', 'Moy. Gle',
           'Moy. sur Mat.Fond.', 'Age en Décembre 2018', 'Sexe_F', 'Sexe_M',
           'Série_S1', 'Série_S2', 'Série_S3', 'Mention_ABien', 'Mention_Bien',
           'Mention_Pass', 'Résidence', 'Ets. de provenance', 'Centre d\'Ec.',
           'Académie de l\'Ets. Prov.', 'REGION_DE_NAISSANCE', 'Academie perf.'
       ]

   def encode_categorical_features(self, categorical_columns):
       self.df = pd.get_dummies(self.df, columns=categorical_columns, prefix=categorical_columns)
       return self

   def label_encode_columns(self, columns):
       le = LabelEncoder()
       for col in columns:
           if col in self.df.columns:
               self.df[col] = le.fit_transform(self.df[col])
       return self

   def calculate_academie_performance(self):
       if "Académie de l'Ets. Prov." in self.df.columns and 'Moy. Gle' in self.df.columns:
           def get_academie_moyenne():
               return pd.Series(
                   self.df['Moy. Gle'].values,
                   index=self.df["Académie de l'Ets. Prov."]
               ).to_dict()

           dic = get_academie_moyenne()
           self.df['Academie perf.'] = self.df.apply(
               lambda row: dic[row["Académie de l'Ets. Prov."]],
               axis=1
           )
       return self

   def convert_to_numeric(self):
       non_numeric_cols = self.df.select_dtypes(include=['object']).columns
       for col in non_numeric_cols:
           try:
               self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
           except:
               pass
       return self

   def convert_columns_to_int(self, columns):
       existing_cols = [col for col in columns if col in self.df.columns]
       if existing_cols:
           self.df[existing_cols] = self.df[existing_cols].astype(int)
       return self

   def clean_data(self):
       if 'MOYENNE ANNUELLE' in self.df.columns:
           self.df = self.df.drop(['MOYENNE ANNUELLE'], axis=1)
       self.df.fillna(0, inplace=True)
       return self

   def ensure_features(self):
       for feature in self.features:
           if feature not in self.df.columns:
               self.df[feature] = 0
       return self

   def process_all(self):
    try:
        print("1. Suppression des doublons")
        self.df = self.df.loc[:,~self.df.columns.duplicated()]
        print("Colonnes après dédoublonnage:", list(self.df.columns))

        print("\n2. Préparation des colonnes")
        dummy_columns = ['Sexe_F', 'Sexe_M', 'Série_S1', 'Série_S2', 'Série_S3',
                        'Mention_ABien', 'Mention_Bien', 'Mention_Pass']
        for col in dummy_columns:
            if col not in self.df.columns:
                print(f"Ajout colonne {col}")
                self.df[col] = 0

        print("\n3. Encodage catégoriel")
        if 'Mention' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['Mention'], prefix='Mention')
        if 'Sexe' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['Sexe'], prefix='Sexe')
        if 'Série' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['Série'], prefix='Série')
        print("Colonnes après encodage:", list(self.df.columns))

        print("\n4. Autres traitements")
        label_cols = ['Résidence', "Ets. de provenance", "Centre d'Ec.",
                      "Académie de l'Ets. Prov.", "REGION_DE_NAISSANCE"]
        self.label_encode_columns(label_cols)
        self.calculate_academie_performance()
        self.convert_to_numeric()
        self.clean_data()
        
        print("\n5. Vérification finale")
        missing_features = set(self.features) - set(self.df.columns)
        if missing_features:
            print(f"Features manquantes: {missing_features}")
            for feature in missing_features:
                print(f"Ajout feature manquante: {feature}")
                self.df[feature] = 0

        return self.df[self.features]

    except Exception as e:
        print(f"\nErreur dans process_all: {str(e)}")
        print("État actuel des colonnes:", list(self.df.columns))
        raise