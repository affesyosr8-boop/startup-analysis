import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Configuration de la page
st.set_page_config(page_title="Analyse de Performance Startups", layout="wide")

st.title("Outil d'Analyse de Performance des Startups")
st.write("Cette plateforme permet d'identifier les leviers de rentabilité et de simuler des prévisions de profit.")

# --- SECTION 1 : CHARGEMENT DES DONNÉES ---
st.markdown("---")
st.header("1. Importation du fichier de données")
uploaded_file = st.file_uploader("Veuillez charger votre fichier 50_Startups.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    with st.expander("Afficher un aperçu du jeu de données"):
        st.dataframe(df.head(10))
    
    # Préparation des variables (Preprocessing)
    # On utilise columns=None pour que pandas trouve les colonnes catégorielles automatiquement
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # On s'assure que toutes les colonnes sont des nombres
    X_data = df_encoded.drop('Profit', axis=1, errors='ignore').astype(float)
    y_data = df_encoded['Profit'].astype(float)

    # --- SECTION 2 : SIMULATION ---
    st.markdown("---")
    st.header("2. Simulation de profit prévisionnel")
    st.write("Ajustez les paramètres disponibles pour calculer une estimation du profit.")
    
    # On boucle sur les colonnes RÉELLES qui existent après le encodage
    actual_columns = X_data.columns.tolist()
    user_inputs = {}
    
    # Création dynamique des colonnes dans l'interface
    cols = st.columns(min(len(actual_columns), 5))
    
    for i, col_name in enumerate(actual_columns):
        with cols[i % len(cols)]:
            label_affichage = col_name.replace('_', ' ')
            valeur_moyenne = float(X_data[col_name].mean())
            user_inputs[col_name] = st.number_input(label_affichage, value=valeur_moyenne)

    # --- SECTION 3 : ANALYSE STATISTIQUE ---
    st.markdown("---")
    st.header("3. Analyse de Régression et Optimisation")
    
    zone_gauche, zone_droite = st.columns([1, 1])
    
    with zone_gauche:
        if st.button("Calculer le modèle optimal"):
            X_opt = sm.add_constant(X_data).astype(float)
            while True:
                modele = sm.OLS(y_data, X_opt).fit()
                # On vérifie si on doit supprimer une variable (P-value > 0.05)
                p_values = modele.pvalues
                if p_values.max() > 0.05:
                    variable_max = p_values.idxmax()
                    # On ne supprime pas la constante
                    if variable_max == 'const' and len(p_values) > 1:
                        # Si c'est la constante qui a la plus grande p-value mais d'autres restent
                        temp_p = p_values.drop('const')
                        if temp_p.max() > 0.05:
                            variable_max = temp_p.idxmax()
                            X_opt = X_opt.drop(columns=[variable_max])
                        else: break
                    else:
                        X_opt = X_opt.drop(columns=[variable_max])
                else:
                    break
            
            st.success("Analyse terminée.")
            
            # Prédiction finale sécurisée
            input_df = pd.DataFrame([user_inputs])
            input_df = sm.add_constant(input_df, has_constant='add')
            # Aligner les colonnes avec le modèle final
            for col in X_opt.columns:
                if col not in input_df.columns:
                    input_df[col] = 1.0 if col == 'const' else 0.0
            
            input_final = input_df[X_opt.columns]
            prediction = modele.predict(input_final)[0]
            
            st.metric("Profit Estimé", f"{prediction:,.2f} $")
            st.text("Résumé détaillé du modèle :")
            st.text(model_summary := modele.summary())

    with zone_droite:
        if st.button("Consulter l'historique d'élimination"):
            X_step = sm.add_constant(X_data).astype(float)
            etape = 1
            while True:
                model_step = sm.OLS(y_data, X_step).fit()
                with st.expander(f"Étape {etape} : {len(X_step.columns)} variables"):
                    st.text(model_step.summary())
                
                p_max = model_step.pvalues.max()
                if p_max > 0.05:
                    var_del = model_step.pvalues.idxmax()
                    X_step = X_step.drop(columns=[var_del])
                    etape += 1
                else: break
else:
    st.info("Veuillez charger un fichier CSV pour activer l'analyse.")
