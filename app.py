import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 1. Configuration
st.set_page_config(page_title="Startup Analysis", layout="wide")
st.title("🚀 Interface d'Analyse des Startups (Optimisée)")

# 2. Upload (FEL WEST MOUSH FEL SIDEBAR)
st.subheader("📁 1. Charger les données")
uploaded_file = st.file_uploader("Choisissez le fichier 50_Startups.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Data Preview")
    st.write(df.head())
    
    # Preprocessing
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
    X_data = df_encoded.drop('Profit', axis=1).astype(float)
    y_data = df_encoded['Profit'].astype(float)

    # 3. Inputs
    st.write("---")
    st.subheader("✍️ 2. Entrez les valeurs pour prédire")
    col_in = st.columns(5)
    feature_names = ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New_York']
    user_inputs = {}

    for i, col_name in enumerate(feature_names):
        with col_in[i % 5]:
            default_val = float(X_data[col_name].mean()) if col_name in X_data.columns else 0.0
            user_inputs[col_name] = st.number_input(f"{col_name}", value=default_val)

    # 4. Boutons
    st.write("")
    col_btn1, col_btn2 = st.columns([1, 4])

    if col_btn1.button("🚀 Lancer Backward Elimination"):
        st.subheader("🎯 Résultat Optimum")
        X_pd = sm.add_constant(X_data).astype(float)
        while True:
            model = sm.OLS(y_data, X_pd).fit()
            if model.pvalues.max() > 0.05:
                var = model.pvalues.idxmax()
                X_pd = X_pd.drop(columns=[var])
            else:
                break
        st.success(f"Variables retenues: {list(X_pd.columns)}")
        st.text(model.summary())
        
        # Prédiction
        input_df = pd.DataFrame([user_inputs])
        input_df = sm.add_constant(input_df, has_constant='add')
        input_final = input_df[X_pd.columns]
        prediction = model.predict(input_final)
        st.metric("Profit Estimé", f"${prediction[0]:,.2f}")

    if col_btn2.button("📜 Tous les Résultats"):
        st.subheader("Historique")
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
                break
else:
    st.warning("👈 Veuillez uploader le fichier CSV ci-dessus pour activer l'analyse.")
