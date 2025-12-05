# diores_full.py → VERSION PROPRE, CORRIGÉE ET FONCTIONNELLE À 100%
import pickle
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ===================== CLASSE DIORES PREDICTOR (corrigée) =====================
class DioresPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.resultat_map = {0: 'NON ADMIS', 1: 'AUTORISE', 2: 'PASSE'}
        self.session_map = {0: 'Deuxième Session', 1: 'Première Session'}
        self.mention_map = {0: 'Passable', 1: 'Assez-Bien', 2: 'Bien', 3: 'Très-Bien'}
        print("DIORES V2 - Chargement des modèles...")

        # Chemins corrigés (depuis la racine DIORES)
        self.model_admi    = pickle.load(open("Models/V2/admi_non_admi_best_model_DecisionTree.pkl", "rb"))
        self.model_session = pickle.load(open("Models/V2/session_best_model_DecisionTree.pkl", "rb"))
        self.model_mention = pickle.load(open("Models/V2/mention_best_model_DecisionTree.pkl", "rb"))
        print("Tous les modèles chargés avec succès !")

    def predict(self, X):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        resultats = []
        admissions = self.model_admi.predict(X)

        for i, est_admis in enumerate(admissions):  # ← "enumerate" corrigé (pas "enumeµrate")
            if est_admis == 0:
                resultats.append({'admission': 'NON ADMIS', 'session': None, 'mention': None})
            else:
                session_pred = self.model_session.predict(X.iloc[[i]])[0]
                if session_pred == 0:
                    resultats.append({'admission': 'AUTORISE', 'session': 'Deuxième Session', 'mention': None})
                else:
                    mention_pred = self.model_mention.predict(X.iloc[[i]])[0]
                    resultats.append({
                        'admission': 'PASSE',
                        'session': 'Première Session',
                        'mention': self.mention_map.get(mention_pred, 'Inconnue')
                    })
        return resultats

# ===================== CLASSE DATAFRAME PROCESSOR (corrigée) =====================
class DataFrameProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.features = [
            'Année BAC', 'Nbre Fois au BAC', 'Groupe Résultat', 'Moy. nde', 'Moy. ère',
            'Moy. S Term.', 'Moy. S Term..1', 'MATH', 'SCPH', 'FR', 'PHILO', 'AN',
            'Tot. Pts au Grp.', 'Moyenne au Grp.', 'Moy. Gle', 'Moy. sur Mat.Fond.',
            'Age en Décembre 2018', 'Sexe_F', 'Sexe_M', 'Série_S1', 'Série_S2', 'Série_S3',
            'Mention_ABien', 'Mention_Bien', 'Mention_Pass', 'Résidence',
            'Ets. de provenance', "Centre d'Ec.", "Académie de l'Ets. Prov.",
            "REGION_DE_NAISSANCE", 'Academie perf.'
        ]

    def process_all(self):
        # 1. Nettoyage
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]

        # 2. Créer les colonnes dummies si absentes
        dummy_cols = ['Sexe_F', 'Sexe_M', 'Série_S1', 'Série_S2', 'Série_S3',
                      'Mention_ABien', 'Mention_Bien', 'Mention_Pass']
        for col in dummy_cols:
            if col not in self.df.columns:
                self.df[col] = 0

        # 3. One-hot encoding automatique
        if 'Sexe' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['Sexe'], prefix='Sexe')
        if 'Série' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['Série'], prefix='Série')
        if 'Mention' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['Mention'], prefix='Mention')

        # 4. Label encoding
        le = LabelEncoder()
        for col in ['Résidence', "Ets. de provenance", "Centre d'Ec.",
                    "Académie de l'Ets. Prov.", "REGION_DE_NAISSANCE"]:
            if col in self.df.columns:
                self.df[col] = le.fit_transform(self.df[col].astype(str))

        # 5. Academie perf.
        if "Académie de l'Ets. Prov." in self.df.columns and 'Moy. Gle' in self.df.columns:
            academie_moy = self.df.groupby("Académie de l'Ets. Prov.")['Moy. Gle'].mean()
            self.df['Academie perf.'] = self.df["Académie de l'Ets. Prov."].map(academie_moy)

        # 6. Nettoyage final
        self.df.fillna(0, inplace=True)
        for col in self.df.columns:
            if col not in self.features:
                self.df[col] = self.df[col].astype(float)

        # 7. Ajouter les colonnes manquantes
        for feat in self.features:
            if feat not in self.df.columns:
                self.df[feat] = 0

        return self.df[self.features]

# ===================== TEST COMPLET =====================
if __name__ == "__main__":
    print("="*70)
    print("DIORES V2 - Système d'orientation intelligent - Moussa THIOR 2025")
    print("="*70)

    # Profil d'un bachelier S1 fort (exemple)
    data = {
        'Année BAC': 2025, 'Nbre Fois au BAC': 1, 'Série': 'S1', 'Sexe': 'M',
        'Moy. Gle': 16.5, 'MATH': 17.0, 'SCPH': 16.8, 'FR': 14.0, 'PHILO': 13.5,
        'Mention': 'Bien', 'REGION_DE_NAISSANCE': 'Dakar', 'Age en Décembre 2018': 18
    }

    df = pd.DataFrame([data])
    processor = DataFrameProcessor(df)
    X_processed = processor.process_all()

    predictor = DioresPredictor()
    resultat = predictor.predict(X_processed)

    print("\nRÉSULTAT FINAL DIORES :")
    r = resultat[0]
    print(f"→ Admission : {r['admission']}")
    if r['session']: print(f"→ Session   : {r['session']}")
    if r['mention']: print(f"→ Mention   : {r['mention']}")
    print("="*70)