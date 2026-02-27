# Streamlit app for running and comparing models from TPML notebook
# Usage: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

st.set_page_config(page_title="TP ML - Compare Models", layout="wide")

# --- Helper functions ---
@st.cache_data
def load_data(path='Dataset1.csv'):
    df = pd.read_csv(path)
    return df


def encode_features(df):
    df_encoded = df.copy()
    encoders = {}
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            # convert NaN to string before encoding then restore NaN if present
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
    return df_encoded, encoders


def knn_impute(df_num, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    arr = imputer.fit_transform(df_num)
    return pd.DataFrame(arr, columns=df_num.columns)


# --- UI ---
st.title("TP ML — Interface d'évaluation des modèles")
st.markdown(
    "Application Streamlit pour charger automatiquement `Dataset1.csv`, prétraiter, entraîner plusieurs algorithmes et comparer leurs résultats en un seul graphique.")

# Load dataset
with st.spinner("Chargement de la dataset..."):
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Impossible de charger Dataset1.csv : {e}")
        st.stop()

# Display dataset
st.subheader("Aperçu de la dataset")
with st.expander("Voir le tableau (format structuré)", expanded=True):
    st.dataframe(df)

col1, col2 = st.columns([1, 2])
with col1:
    st.write("### Résumé rapide")
    st.write(df.describe(include='all').T)
    st.write("Valeurs manquantes par colonne:")
    st.write(df.isnull().sum())

with col2:
    st.write("### Choix de l'utilisateur")
    task = st.radio("Tâche", ("Classification", "Regression"))
    # default targets based on notebook
    default_class_target = 'Precip Type' if 'Precip Type' in df.columns else df.columns[-1]
    default_reg_target = 'Humidity' if 'Humidity' in df.columns else df.columns[-1]
    if task == 'Classification':
        target = st.selectbox("Choisir la variable cible (classification)", df.columns.tolist(), index=list(df.columns).index(default_class_target))
    else:
        target = st.selectbox("Choisir la variable cible (régression)", df.columns.tolist(), index=list(df.columns).index(default_reg_target))

    st.write("---")
    st.write("Prétraitement:")
    impute_method = st.selectbox("Méthode d'imputation pour les valeurs manquantes", ('KNN (recommandé)', 'Mode (valeur la plus fréquente)', 'Supprimer lignes'))
    knn_neighbors_imputer = st.slider("KNN imputer - k (si utilisé)", min_value=1, max_value=20, value=5)
    standardize = st.checkbox("Standardiser les features (StandardScaler)", value=True)

    st.write("---")
    st.write("Paramètres des modèles:")
    knn_k = st.slider("K pour KNN classifier/regressor", min_value=1, max_value=30, value=5)
    svm_kernel = st.selectbox("Kernel SVM", ('rbf', 'linear', 'poly', 'sigmoid'))
    test_size = st.slider("Taille du test (fraction)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

    run_button = st.button("Exécuter les modèles")

# When user clicks run
if run_button:
    st.session_state['last_run'] = pd.Timestamp.now().isoformat()
    with st.spinner("Prétraitement et entraînement des modèles..."):
        # Prepare dataset copy
        data = df.copy()

        # Keep only numeric for imputation (but encode categoricals first to numeric via temporary label encoding for KNN imputer)
        # We'll encode strings to numbers but store mapping back where possible
        # For simplicity with KNNImputer, convert object columns temporarily to label-encoded numeric values
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Save original target if categorical
        target_is_object = data[target].dtype == 'object'

        # Encode categoricals for features and possibly target (for classification)
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            # Replace NaN with a placeholder string to allow LabelEncoder to fit
            data[col] = data[col].astype(str)
            data[col] = le.fit_transform(data[col])
            encoders[col] = le

        # Imputation
        if impute_method == 'KNN (recommandé)':
            try:
                data = knn_impute(data, n_neighbors=knn_neighbors_imputer)
            except Exception as e:
                st.error(f"Erreur KNN imputer: {e}")
                st.stop()
        elif impute_method == 'Mode (valeur la plus fréquente)':
            for c in data.columns:
                data[c] = data[c].fillna(data[c].mode()[0])
        else:
            data = data.dropna()

        # After imputation, restore proper dtypes for target when possible
        # If target originally categorical (object), and we had an encoder, inverse transform back to original labels for display, but for modeling keep numeric
        # Split X/y
        X = data.drop(columns=[target])
        y = data[target]

        # If standardize selected
        scaler = None
        if standardize:
            scaler = StandardScaler()
            try:
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            except Exception as e:
                st.warning(f"Standardisation échouée (peut contenir non-numériques): {e}")

        # Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if task=='Classification' and len(np.unique(y))>1 else None)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Model lists (no duplicates)
        results = []

        if task == 'Classification':
            # Models: RandomForest, KNN, SVC, DecisionTree
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=knn_k),
                'SVC': SVC(kernel=svm_kernel, probability=False),
                'DecisionTree': DecisionTreeClassifier(random_state=42)
            }

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    rep = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
                except Exception as e:
                    st.warning(f"Le modèle {name} a échoué: {e}")
                    acc = np.nan
                    rep = {}
                results.append({'model': name, 'accuracy': acc, 'report': rep})

            # Build results df
            res_df = pd.DataFrame([{'Model': r['model'], 'Accuracy': r['accuracy']} for r in results])
            best_idx = res_df['Accuracy'].idxmax()
            best_model_name = res_df.loc[best_idx, 'Model'] if not res_df['Accuracy'].isnull().all() else 'Aucun'

            # Display
            st.subheader("Résultats Classification")
            st.write(res_df.sort_values('Accuracy', ascending=False).reset_index(drop=True))

            # Single consolidated graph (accuracies)
            fig = px.bar(res_df, x='Model', y='Accuracy', text='Accuracy', title='Comparaison des modèles (Accuracy)')
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"Meilleur modèle classification: {best_model_name} (Accuracy = {res_df['Accuracy'].max():.3f})")

            # Show classification reports in expandable sections
            for r in results:
                with st.expander(f"Classification report — {r['model']}"):
                    if r['report']:
                        rep_df = pd.DataFrame(r['report']).transpose()
                        st.dataframe(rep_df)
                    else:
                        st.write("Pas de rapport disponible")

        else:
            # Regression models: Linear, DecisionTreeRegressor, Lasso, Ridge
            models = {
                'LinearRegression': LinearRegression(),
                'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
                'Lasso': Lasso(alpha=0.1, max_iter=10000),
                'Ridge': Ridge(alpha=0.1)
            }

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                except Exception as e:
                    st.warning(f"Le modèle {name} a échoué: {e}")
                    mse = np.nan
                    r2 = np.nan
                results.append({'model': name, 'mse': mse, 'r2': r2})

            # Results df and best model by R2
            res_df = pd.DataFrame([{'Model': r['model'], 'R2': r['r2'], 'MSE': r['mse']} for r in results])
            best_idx = res_df['R2'].idxmax()
            best_model_name = res_df.loc[best_idx, 'Model'] if not res_df['R2'].isnull().all() else 'Aucun'

            st.subheader("Résultats Régression")
            st.write(res_df.sort_values('R2', ascending=False).reset_index(drop=True))

            # Consolidated graph (R2 scores)
            fig = px.bar(res_df, x='Model', y='R2', text='R2', title='Comparaison des modèles (R²)')
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"Meilleur modèle régression: {best_model_name} (R² = {res_df['R2'].max():.3f})")

            # Show MSE table
            with st.expander("Métriques détaillées (MSE et R²)"):
                st.dataframe(res_df)

    st.balloons()
    st.write("---")
    st.write("Exécution terminée. Vous pouvez ajuster les paramètres et relancer.")

else:
    st.info("Sélectionnez des paramètres puis cliquez sur 'Exécuter les modèles' pour commencer.")

# Footer
st.markdown("---")
st.caption("Basé sur le notebook TPML. Choisissez le kernel SVM et le K pour KNN puis exécutez.")
