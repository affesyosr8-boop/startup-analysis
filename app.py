import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 1. Configuration mta3 el Page
st.set_page_config(page_title="Startup Analysis", layout="wide")
st.title("💰 Startup Profit Predictor")

# 2. Sidebar Upload
st.sidebar.header("📥 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload 50_Startups.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("## 📊 Data Preview", df.head())
    
    # Preprocessing automatique (Convert en float pour éviter les erreurs)
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
    X_data = df_encoded.drop('Profit', axis=1).astype(float)
    y_data = df_encoded['Profit'].astype(float)

    # 3. Inputs mta3 el utilisateur (Dima dhohrin)
    st.write("---")
    st.write("## ✍️ Entrez les valeurs pour la prédiction")
    cols = st.columns(5)
    user_inputs = {}
    
    # List fixed mta3 el features
    manual_features = ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New_York']
    
    for i, feat in enumerate(manual_features):
        with cols[i % 5]:
            val_default = float(X_data[feat].mean()) if feat in X_data.columns else 0.0
            user_inputs[feat] = st.number_input(f"{feat}", value=val_default)

    st.write("---")
    
    # 4. Boutounat el Action
    col_btn1, col_btn2 = st.columns([1, 4])
    
    if col_btn1.button("🚀 Lancer Backward Elimination"):
        st.subheader("🎯 Résultat Optimum")
        X_opt = sm.add_constant(X_data).astype(float)
        
        while True:
            model = sm.OLS(y_data, X_opt).fit()
            if model.pvalues.max() > 0.05:
                var = model.pvalues.idxmax()
                X_opt = X_opt.drop(columns=[var])
            else:
                break
        
        st.success(f"Modèle optimisé avec : {list(X_opt.columns)}")
        st.text(model.summary())
        
        # Résultat de la prédiction
        input_df = pd.DataFrame([user_inputs])
        input_df = sm.add_constant(input_df, has_constant='add')
        input_final = input_df[X_opt.columns]
        prediction = model.predict(input_final)
        
        st.metric("Profit Estimé", f"${prediction[0]:,.2f}")

    if col_btn2.button("📜 Tous les Résultats"):
        st.subheader("Historique de l'élimination")
        X_all = sm.add_constant(X_data).astype(float)
        iteration = 1
        while True:
            model_step = sm.OLS(y_data, X_all).fit()
            with st.expander(f"Étape {iteration} - Variables: {len(X_all.columns)}"):
                st.text(model_step.summary())
            
            if model_step.pvalues.max() > 0.05:
                var = model_step.pvalues.idxmax()
                X_all = X_all.drop(columns=[var])
                iteration += 1
            else:
                st.success("🎯 Modèle Optimum atteint !")
                break
else:
    st.info("👋 Veuillez uploader le fichier CSV pour commencer.")
