import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

st.title("🚀 Interface d'Analyse des Startups")

# L'upload tawwa iwali fi wast el page bech ma tatla3ch ValueError
uploaded_file = st.file_uploader("Veuillez choisir le fichier 50_Startups.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    
    # Prétraitement
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
    X = df_encoded.drop('Profit', axis=1)
    y = df_encoded['Profit']

    # Inputs lel prediction
    st.header("Entrez les prix pour prédire")
    user_inputs = {}
    cols = st.columns(len(X.columns))
    for i, col in enumerate(X.columns):
        user_inputs[col] = cols[i].number_input(f"{col}", value=float(df_encoded[col].mean()))

    if st.button("Lancer Backward Elimination"):
        X_pd = sm.add_constant(X)
        X_pd = X_pd.astype(float)
        
        while True:
            model = sm.OLS(y, X_pd).fit()
            p_values = model.pvalues
            if p_values.max() > 0.05:
                X_pd = X_pd.drop(columns=[p_values.idxmax()])
            else:
                break
        
        st.success("Analyse terminée!")
        st.text(model.summary())
else:
    st.info("👋 Mar7ba bik! Veuillez uploader le fichier CSV pour commencer l'analyse.")
    
