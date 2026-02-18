import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 1. Configuration de la page (Wide layout bech yjiw jnab b3adhhom)
st.set_page_config(page_title="Startup Analysis", layout="wide")
st.title("💰 Startup Profit Predictor")

# 2. Sidebar - Upload Data
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload 50_Startups.csv", type="csv")

if uploaded_file is not None:
    # Read Data
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Preprocessing (Encoding)
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
    X = df_encoded.drop('Profit', axis=1)
    y = df_encoded['Profit']

    # User Inputs pour la prédiction finale
    st.sidebar.header("2. Predict for new Startup")
    user_data = {}
    for col in X.columns:
        user_data[col] = st.sidebar.number_input(f"Enter {col}", value=float(df_encoded[col].mean()))

    # 3. Affichage des deux méthodes (Numpy vs Pandas)
    st.write("---")
    st.subheader("Choisir la méthode de Backward Elimination")
    
    col1, col2 = st.columns(2)

    with col1:
        # --- Méthode NUMPY ---
        if st.button("Run Backward (Numpy)"):
            st.markdown("#### 🔢 Résultat Numpy (Indices)")
            X_numpy = sm.add_constant(np.array(X))
            SL = 0.05
            while True:
                model_np = sm.OLS(y, X_numpy).fit()
                max_p = max(model_np.pvalues)
                if max_p > SL:
                    X_numpy = np.delete(X_numpy, model_np.pvalues.argmax(), axis=1)
                else:
                    break
            st.text(model_np.summary())

    with col2:
        # --- Méthode PANDAS ---
        if st.button("Run Backward (Pandas)"):
            st.markdown("#### 🐼 Résultat Pandas (Column Names)")
            X_pd = sm.add_constant(X)
            SL = 0.05
            while True:
                model_pd = sm.OLS(y, X_pd).fit()
                max_p = model_pd.pvalues.max()
                if max_p > SL:
                    var = model_pd.pvalues.idxmax()
                    st.write(f"Suppression: `{var}`")
                    X_pd = X_pd.drop(columns=[var])
                else:
                    break
            
            st.success(f"Variables finales: {list(X_pd.columns)}")
            st.text(model_pd.summary())

            # 4. Prediction finale (basée sur le modèle Pandas)
            st.write("---")
            st.subheader("🔮 Prediction Result")
            new_x = pd.DataFrame([user_data])
            # On garde seulement les colonnes que le modèle a retenu
            new_x_const = sm.add_constant(new_x, has_constant='add')[X_pd.columns]
            pred = model_pd.predict(new_x_const)
            st.metric("Estimated Profit", f"${pred[0]:,.2f}")

else:
    # Message d'accueil pour éviter le ValueError au début
    st.info("👋 Mar7ba bik! Please upload the '50_Startups.csv' file from the sidebar to start.")
