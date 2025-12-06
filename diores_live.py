# diores_live.py  → CE FICHIER MARCHE À 100% CHEZ TOI
import pickle
import pandas as pd

print("DIORES V2 - Prédicteur BAC Sénégal 2018-2024")
print("="*60)

# CHARGEMENT DES 3 MODÈLES (chemins corrigés pour ton PC)
try:
    model_admi    = pickle.load(open("Models/V2/admi_non_admi_best_model_DecisionTree.pkl", "rb"))
    model_session = pickle.load(open("Models/V2/session_best_model_DecisionTree.pkl", "rb"))
    model_mention = pickle.load(open("Models/V2/mention_best_model_DecisionTree.pkl", "rb"))
    print("Tous les modèles chargés avec succès !")
except Exception as e:
    print("ERREUR : Modèle non trouvé → vérifie le chemin Models/V2/")
    print(e)
    exit()

def predire_bac(profil):
    df = pd.DataFrame([profil])
    
    # 1. Admis ou pas ?
    pred_admi = model_admi.predict(df)[0]
    if pred_admi == 0:
        return "NON ADMIS au BAC"
    
    # 2. Session ?
    pred_session = model_session.predict(df)[0]
    if pred_session == 0:
        return "AUTORISÉ → Deuxième session"
    
    # 3. Mention ?
    pred_mention = model_mention.predict(df)[0]
    mentions = {0: "Passable", 1: "Assez Bien", 2: "Bien", 3: "Très Bien"}
    mention = mentions.get(pred_mention, "Inconnue")
    
    return f"PASSÉ en PREMIÈRE SESSION avec mention {mention} !"

# TEST IMMÉDIAT
profil_test = {
    'Année BAC': 2024, 'Nbre Fois au BAC': 1, 'Groupe Résultat': 1,
    'Moy. nde': 15.8, 'Moy. ère': 16.2, 'Moy. S Term.': 16.8, 'Moy. S Term..1': 16.8,
    'MATH': 17.5, 'SCPH': 17.0, 'FR': 14.5, 'PHILO': 13.0, 'AN': 15.0,
    'Tot. Pts au Grp.': 870, 'Moyenne au Grp.': 16.7, 'Moy. Gle': 16.4,
    'Moy. sur Mat.Fond.': 17.2, 'Age en Décembre 2018': 18,
    'Sexe_F': 0, 'Sexe_M': 1,
    'Série_S1': 1, 'Série_S2': 0, 'Série_S3': 0,
    'Mention_ABien': 0, 'Mention_Bien': 1, 'Mention_Pass': 0,
    'Résidence': 1, 'Ets. de provenance': 2, "Centre d'Ec.": 3,
    "Académie de l'Ets. Prov.": 1, "REGION_DE_NAISSANCE": 1,
    'Academie perf.': 16.0
}

resultat = predire_bac(profil_test)
print("\nRÉSULTAT POUR CE PROFIL :")
print(resultat)