# app.py - VERSION FINALE 100% FONCTIONNELLE (Dataset change bien dans ML)
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    r2_score, mean_squared_error, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


# ===================== CONFIG =====================
DATASETS_DIR = "datasets"
os.makedirs(DATASETS_DIR, exist_ok=True)

datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]
if not datasets:
    st.error("Ajoute des CSV dans le dossier 'datasets/'")
    st.stop()
datasets = [os.path.splitext(f)[0] for f in datasets]

# ===================== APP =====================
st.set_page_config(page_title="ML Dashboard", layout="wide")
st.title("Dashboard ML Interactif - Classification & Régression")

# Sidebar
st.sidebar.title("Datasets")
# Sélection par boutons dans la sidebar
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = datasets[0]
for ds in datasets:
    if st.sidebar.button(ds, key=f"ds_btn_{ds}"):
        st.session_state.current_dataset = ds

selected = st.session_state.current_dataset

# Chargement du dataset (force reload à chaque changement)
@st.cache_data(show_spinner=False)
def load_dataset(name):
    return pd.read_csv(os.path.join(DATASETS_DIR, f"{name}.csv"))

df_raw = load_dataset(selected)

# Règles spécifiques pour Dataset1
if selected.lower() == 'dataset1':
    cols_to_drop = [c for c in ['Daily Summary', 'Formatted Date'] if c in df_raw.columns]
    if cols_to_drop:
        df_raw = df_raw.drop(columns=cols_to_drop)
    if 'Humidity' in df_raw.columns:
        df_raw['Humidity'] = pd.to_numeric(df_raw['Humidity'], errors='coerce')

# Règles spécifiques pour T1 : supprimer les colonnes de type date/time
if selected.lower() == 't1':
    date_like_cols = [c for c in df_raw.columns if ('date' in c.lower()) or ('time' in c.lower())]
    dt_cols = list(df_raw.select_dtypes(include=['datetime64[ns]']).columns)
    cols_to_drop = list(dict.fromkeys(date_like_cols + dt_cols))  # unique order-preserving
    if cols_to_drop:
        df_raw = df_raw.drop(columns=cols_to_drop)

# Règles spécifiques pour Active_power : supprimer 'date' mais garder Wind Speed et Wind Direction
if selected.lower() == 'active_power':
    if 'Date/Time' in df_raw.columns:
        df_raw = df_raw.drop(columns=['Date/Time'])

# Nettoyage du cache si dataset change
if st.session_state.get('previous_dataset') != selected:
    # Supprimer seulement les données liées au dataset précédent
    keys_to_remove = [k for k in st.session_state.keys() if k not in ['current_dataset', 'previous_dataset']]
    for key in keys_to_remove:
        del st.session_state[key]
    st.session_state.previous_dataset = selected

tab_eda, tab_ml = st.tabs(["EDA", "ML"])

# --------------------- EDA ---------------------
with tab_eda:
    st.header(f"EDA - {selected}")
    df_eda = df_raw.copy()

    st.dataframe(df_eda.head(10), use_container_width=True)
    st.write(f"**{df_eda.shape[0]} lignes × {df_eda.shape[1]} colonnes**")

    # Description du dataset
    st.subheader("Description du dataset")
    with st.expander("Statistiques descriptives"):
        st.write(df_eda.describe(include='all').transpose())
    with st.expander("Types de colonnes"):
        dtypes_df = pd.DataFrame({
            'colonne': df_eda.columns,
            'type': df_eda.dtypes.astype(str),
            'nunique': [df_eda[c].nunique(dropna=False) for c in df_eda.columns]
        })
        st.dataframe(dtypes_df, use_container_width=True)

    # Valeurs manquantes
    missing = df_eda.isnull().sum()
    if missing.sum() > 0:
        st.subheader("Valeurs manquantes")
        st.write(missing[missing > 0])
        fig = px.bar(missing[missing > 0] / len(df_eda) * 100, title="% Manquantes")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Imputation")
        method = st.selectbox("Méthode", ['mean', 'median', 'most_frequent', 'KNN'], key="imp_eda")
        missing_cols = missing[missing > 0].index.tolist()

        # Avant
        for col in missing_cols:
            with st.expander(f"{col} - Avant"):
                if pd.api.types.is_numeric_dtype(df_eda[col]):
                    st.write(df_eda[col].describe())
                    fig = px.histogram(df_eda, x=col, title=f"{col} - Avant")
                    st.plotly_chart(fig)

        if st.button("Appliquer imputation", key="apply_imp"):
            with st.spinner("Imputation..."):
                num_cols = df_eda.select_dtypes(include=['number']).columns.tolist()
                cat_cols = df_eda.select_dtypes(include=['object']).columns.tolist()

                if method == 'KNN':
                    imputer_num = KNNImputer()
                else:
                    imputer_num = SimpleImputer(strategy=method)

                imputer_cat = SimpleImputer(strategy='most_frequent')

                preprocessor = ColumnTransformer([
                    ('num', imputer_num, num_cols),
                    ('cat', imputer_cat, cat_cols)
                ])

                imputed_data = preprocessor.fit_transform(df_eda)
                df_imputed = pd.DataFrame(imputed_data, columns=num_cols + cat_cols)
                # Réorganiser les colonnes dans l'ordre original
                df_imputed = df_imputed[df_eda.columns]
                st.session_state.df_imputed = df_imputed
                st.session_state.imputation_applied = True
                st.success("✅ Imputation appliquée ! Le dataset imputé sera utilisé pour le ML.")

                # Afficher les valeurs manquantes avant/après
                st.subheader("Comparaison des valeurs manquantes (Avant / Après)")
                missing_after = df_imputed.isnull().sum()
                comp_missing = pd.DataFrame({
                    'Avant': missing[missing > 0],
                    'Après': missing_after[missing.index]
                }).fillna(0).astype(int)
                st.dataframe(comp_missing)

                # Après
                for col in missing_cols:
                    with st.expander(f"{col} - Après"):
                        if pd.api.types.is_numeric_dtype(df_imputed[col]):
                            st.write(df_imputed[col].describe())
                            fig = px.histogram(df_imputed, x=col, title=f"{col} - Après", color_discrete_sequence=['#27ae60'])
                            st.plotly_chart(fig)

    # Distribution d'une colonne choisie
    st.subheader("Distribution d'une colonne")
    col_choice = st.selectbox("Choisir une colonne", options=df_eda.columns, key="eda_col_choice")
    if pd.api.types.is_numeric_dtype(df_eda[col_choice]):
        fig = px.histogram(df_eda, x=col_choice, title=col_choice, nbins=30)
    else:
        counts = df_eda[col_choice].astype(str).value_counts(dropna=False).reset_index()
        counts.columns = [col_choice, 'count']
        fig = px.bar(counts, x=col_choice, y='count', title=col_choice)
    st.plotly_chart(fig, use_container_width=True)

    # Corrélation
    st.subheader("Corrélation")
    df_corr = df_eda.copy()
    for col in df_corr.select_dtypes(include=['object']).columns:
        df_corr[col] = LabelEncoder().fit_transform(df_corr[col].astype(str))
    fig = px.imshow(df_corr.corr(), text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

# --------------------- ML ---------------------
with tab_ml:
    st.header(f"ML - {selected}")

    # Utilise df imputé ou brut
    df = st.session_state.get('df_imputed', df_raw.copy())
    
    # Indicateur du dataset utilisé
    if st.session_state.get('imputation_applied', False):
        st.success("✅ Utilisation du dataset IMPUTÉ pour l'entraînement")
    else:
        st.info("ℹ️ Utilisation du dataset BRUT (allez dans l'onglet EDA pour appliquer une imputation si nécessaire)")

    # Sélection de la tâche via boutons
    st.subheader("Tâche")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    if col_btn1.button("Classification", key="btn_task_clf"):
        st.session_state.task = "Classification"
    if col_btn2.button("Régression", key="btn_task_reg"):
        st.session_state.task = "Régression"
    if col_btn3.button("Détection d'anomalie", key="btn_task_anomaly"):
        st.session_state.task = "Détection d'anomalie"
    task = st.session_state.get("task", "Classification")
    is_clf = task == "Classification"
    is_anomaly = task == "Détection d'anomalie"
    st.caption(f"Tâche sélectionnée : {task}")

    # ==================== DÉTECTION D'ANOMALIE ====================
    if is_anomaly:
        st.subheader("Configuration - Détection d'anomalie")
        
        # Choix de la colonne cible
        target = st.selectbox("Colonne cible", options=df.columns.tolist(), key="target_anomaly")
        
        X = df.drop(columns=[target]).copy()
        y = df[target]
        
        # Suppression colonnes inutiles
        drop_cols = []
        for col in X.columns:
            name = col.lower()
            if any(k in name for k in ['date', 'time', 'id', 'index', 'row', 'formatted', 'daily']):
                drop_cols.append(col)
            elif X[col].nunique() > 0.95 * len(X):
                drop_cols.append(col)
        drop_cols = [c for c in drop_cols if c.lower() != 'humidity']
        
        # Ne jamais supprimer Wind Speed et Wind Direction pour Active_power
        if selected.lower() == 'active_power':
            drop_cols = [c for c in drop_cols if c not in ['Wind Speed (m/s)', 'Wind Direction (°)']]
        
        if drop_cols:
            st.warning(f"Supprimées : {', '.join(drop_cols)}")
            X = X.drop(columns=drop_cols)
        
        if X.empty:
            st.error("Aucune feature !")
            st.stop()
        
        # Encoder les colonnes catégorielles
        num_cols = X.select_dtypes(include='number').columns.tolist()
        cat_cols = X.select_dtypes(include='object').columns.tolist()
        
        if cat_cols:
            for col in cat_cols:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Choix de la classe normale
        unique_classes = y.unique()
        st.write(f"Classes disponibles : {list(unique_classes)}")
        normal_class = st.selectbox("Classe NORMALE (les autres = anomalies)", options=unique_classes, key="normal_class")
        
        # Standardisation
        standardize = st.checkbox("Standardiser ?", True, key="std_anomaly")
        
        # Test size
        test_size = st.slider("Test %", 10, 50, 20, key="test_anomaly") / 100
        
        # Choix du modèle
        anomaly_model = st.radio("Méthode", ["One-Class SVM", "Local Outlier Factor (LOF)"], horizontal=True, key="anomaly_method")
        
        # Paramètres
        if anomaly_model == "One-Class SVM":
            nu = st.slider("nu (proportion d'anomalies attendues)", 0.01, 0.5, 0.1, key="nu_param")
        else:
            n_neighbors = st.slider("n_neighbors", 5, 50, 20, key="lof_neighbors")
            contamination = st.slider("contamination", 0.01, 0.5, 0.1, key="lof_contam")
        
        if st.button("Lancer détection", key="launch_anomaly"):
            with st.spinner("Entraînement..."):
                # Standardisation
                if standardize:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = X.values
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, stratify=y, random_state=42
                )
                
                # Construction du train (90% normal, 10% anomalies)
                X_train_normal = X_train[y_train == normal_class]
                X_train_anomaly = X_train[y_train != normal_class]
                
                n_normal = int(0.9 * len(X_train_normal))
                X_train_normal = X_train_normal[:n_normal]
                
                n_anomaly = int(0.1 * len(X_train_anomaly))
                X_train_anomaly = X_train_anomaly[:n_anomaly]
                
                X_train_final = np.vstack([X_train_normal, X_train_anomaly])
                
                # Labels binaires (Normal=1, Anomalie=-1)
                y_test_binary = np.where(y_test == normal_class, 1, -1)
                
                # Entraînement
                if anomaly_model == "One-Class SVM":
                    model = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
                    model.fit(X_train_final)
                    y_pred = model.predict(X_test)
                    scores = model.decision_function(X_test)
                else:
                    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)
                    model.fit(X_train_final)
                    y_pred = model.predict(X_test)
                    scores = model.decision_function(X_test)
                
                # Métriques
                st.success("✅ Terminé !")
                st.subheader(f"Résultats - {anomaly_model}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy_score(y_test_binary, y_pred):.4f}")
                col2.metric("Precision", f"{precision_score(y_test_binary, y_pred, pos_label=1, zero_division=0):.4f}")
                col3.metric("Recall", f"{recall_score(y_test_binary, y_pred, pos_label=1, zero_division=0):.4f}")
                col4.metric("F1-Score", f"{f1_score(y_test_binary, y_pred, pos_label=1, zero_division=0):.4f}")
                
                # Courbe ROC
                st.subheader("Courbe ROC")
                fpr, tpr, _ = roc_curve(y_test_binary, scores)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{anomaly_model} (AUC={roc_auc:.2f})'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random'))
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    title="ROC Curve"
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                
                # Courbe Precision-Recall
                st.subheader("Courbe Precision-Recall")
                prec, rec, _ = precision_recall_curve(y_test_binary, scores)
                
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name=anomaly_model))
                fig_pr.update_layout(
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    title="Precision-Recall Curve"
                )
                st.plotly_chart(fig_pr, use_container_width=True)
        
        st.stop()  # Arrêter ici pour ne pas exécuter le code classification/régression

    # ==================== CLASSIFICATION / RÉGRESSION ====================

    # Cible
    st.subheader("Cible")
    if is_clf:
        candidates = [c for c in df.columns if df[c].nunique() <= 20]
    else:
        candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 10]
    if not candidates:
        candidates = df.columns.tolist()

    target = st.selectbox("Choisir cible", options=candidates, key="target")

    X = df.drop(columns=[target]).copy()
    y = df[target]

    # Suppression colonnes inutiles
    drop_cols = []
    for col in X.columns:
        name = col.lower()
        if any(k in name for k in ['date', 'time', 'id', 'index', 'row', 'formatted', 'daily']):
            drop_cols.append(col)
        elif X[col].nunique() > 0.95 * len(X):
            drop_cols.append(col)

    # Ne jamais supprimer Humidity pour Dataset1
    drop_cols = [c for c in drop_cols if c.lower() != 'humidity']
    
    # Ne jamais supprimer Wind Speed et Wind Direction pour Active_power
    if selected.lower() == 'active_power':
        drop_cols = [c for c in drop_cols if c not in ['Wind Speed (m/s)', 'Wind Direction (°)']]

    if drop_cols:
        st.warning(f"Supprimées : {', '.join(drop_cols)}")
        X = X.drop(columns=drop_cols)

    if X.empty:
        st.error("Aucune feature !")
        st.stop()

    if is_clf and y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y))

    # Preprocessing
    standardize = st.checkbox("Standardiser ?", True)

    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    # Attention mémoire: encoder en sparse pour éviter les matrices denses énormes
    num_pipe = Pipeline([('scaler', StandardScaler())]) if standardize else 'passthrough'
    cat_pipe = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))]) if cat_cols else 'passthrough'

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    # Split
    test_size = st.slider("Test %", 10, 50, 20) / 100
    shuffle = st.checkbox("Shuffle ?", True)

    # Modèles
    models = (
        ['KNN', 'Random Forest', 'Decision Tree', 'SVM', 'XGBoost', 'Logistic Regression', 'Gaussian NB']
        if is_clf else
        ['XGBoost', 'Linear Regression', 'Ridge', 'Lasso', 'AdaBoost', 'Gradient Boosting']
    )

    # Fonction pour obtenir les hyperparamètres selon le modèle
    def get_hyperparameters(model_name, prefix, is_classification):
        params = {}
        st.markdown(f"**Hyperparamètres - {model_name}**")
        
        if model_name == 'KNN':
            params['n_neighbors'] = st.slider("n_neighbors (K)", 1, 30, 5, key=f"{prefix}_knn_k")
            params['weights'] = st.selectbox("weights", ['uniform', 'distance'], key=f"{prefix}_knn_w")
            params['metric'] = st.selectbox("metric", ['euclidean', 'manhattan', 'minkowski'], key=f"{prefix}_knn_m")
        
        elif model_name == 'Random Forest':
            params['n_estimators'] = st.slider("n_estimators", 10, 500, 100, 10, key=f"{prefix}_rf_n")
            params['max_depth'] = st.slider("max_depth", 1, 50, 10, key=f"{prefix}_rf_d")
            params['min_samples_split'] = st.slider("min_samples_split", 2, 20, 2, key=f"{prefix}_rf_s")
            params['min_samples_leaf'] = st.slider("min_samples_leaf", 1, 20, 1, key=f"{prefix}_rf_l")
            params['max_features'] = st.selectbox("max_features", ['sqrt', 'log2', None], key=f"{prefix}_rf_f")
        
        elif model_name == 'Decision Tree':
            params['max_depth'] = st.slider("max_depth", 1, 50, 10, key=f"{prefix}_dt_d")
            params['min_samples_split'] = st.slider("min_samples_split", 2, 20, 2, key=f"{prefix}_dt_s")
            params['min_samples_leaf'] = st.slider("min_samples_leaf", 1, 20, 1, key=f"{prefix}_dt_l")
            params['criterion'] = st.selectbox("criterion", ['gini', 'entropy'] if is_classification else ['squared_error', 'absolute_error'], key=f"{prefix}_dt_c")
        
        elif model_name == 'SVM':
            params['C'] = st.slider("C (regularization)", 0.01, 10.0, 1.0, 0.1, key=f"{prefix}_svm_c")
            params['kernel'] = st.selectbox("kernel", ['rbf', 'linear', 'poly', 'sigmoid'], key=f"{prefix}_svm_k")
            if params['kernel'] == 'poly':
                params['degree'] = st.slider("degree", 2, 5, 3, key=f"{prefix}_svm_d")
            params['gamma'] = st.selectbox("gamma", ['scale', 'auto'], key=f"{prefix}_svm_g")
        
        elif model_name == 'XGBoost':
            params['n_estimators'] = st.slider("n_estimators", 10, 500, 100, 10, key=f"{prefix}_xgb_n")
            params['max_depth'] = st.slider("max_depth", 1, 20, 6, key=f"{prefix}_xgb_d")
            params['learning_rate'] = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01, key=f"{prefix}_xgb_lr")
            params['subsample'] = st.slider("subsample", 0.5, 1.0, 0.8, 0.1, key=f"{prefix}_xgb_ss")
            params['colsample_bytree'] = st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.1, key=f"{prefix}_xgb_cs")
        
        elif model_name == 'Logistic Regression':
            params['C'] = st.slider("C (inverse regularization)", 0.01, 10.0, 1.0, 0.1, key=f"{prefix}_lr_c")
            params['penalty'] = st.selectbox("penalty", ['l2', 'l1', 'elasticnet', None], key=f"{prefix}_lr_p")
            params['solver'] = st.selectbox("solver", ['lbfgs', 'liblinear', 'saga'], key=f"{prefix}_lr_s")
            params['max_iter'] = st.slider("max_iter", 100, 1000, 100, 100, key=f"{prefix}_lr_i")
        
        elif model_name == 'Gaussian NB':
            params['var_smoothing'] = st.slider("var_smoothing", 1e-10, 1e-5, 1e-9, format="%.1e", key=f"{prefix}_gnb_v")
        
        elif model_name == 'Linear Regression':
            params['fit_intercept'] = st.checkbox("fit_intercept", True, key=f"{prefix}_linreg_fi")
        
        elif model_name == 'Ridge':
            params['alpha'] = st.slider("alpha (regularization)", 0.01, 10.0, 1.0, 0.1, key=f"{prefix}_ridge_a")
            params['solver'] = st.selectbox("solver", ['auto', 'svd', 'cholesky', 'lsqr', 'saga'], key=f"{prefix}_ridge_s")
        
        elif model_name == 'Lasso':
            params['alpha'] = st.slider("alpha (regularization)", 0.01, 10.0, 1.0, 0.1, key=f"{prefix}_lasso_a")
            params['max_iter'] = st.slider("max_iter", 100, 2000, 1000, 100, key=f"{prefix}_lasso_i")
        
        elif model_name == 'AdaBoost':
            params['n_estimators'] = st.slider("n_estimators", 10, 500, 50, 10, key=f"{prefix}_ada_n")
            params['learning_rate'] = st.slider("learning_rate", 0.01, 2.0, 1.0, 0.1, key=f"{prefix}_ada_lr")
        
        elif model_name == 'Gradient Boosting':
            params['n_estimators'] = st.slider("n_estimators", 10, 500, 100, 10, key=f"{prefix}_gb_n")
            params['learning_rate'] = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01, key=f"{prefix}_gb_lr")
            params['max_depth'] = st.slider("max_depth", 1, 20, 3, key=f"{prefix}_gb_d")
            params['subsample'] = st.slider("subsample", 0.5, 1.0, 1.0, 0.1, key=f"{prefix}_gb_ss")
        
        return params

    st.subheader("Configuration Modèle 1")
    model1 = st.selectbox("Modèle 1", models, key="m1")
    with st.expander("⚙️ Hyperparamètres Modèle 1", expanded=True):
        p1 = get_hyperparameters(model1, "m1", is_clf)

    selected = [(model1, p1)]
    
    use_model2 = st.checkbox("Ajouter Modèle 2")
    if use_model2:
        st.subheader("Configuration Modèle 2")
        model2 = st.selectbox("Modèle 2", models, index=min(1, len(models)-1), key="m2")
        with st.expander("⚙️ Hyperparamètres Modèle 2", expanded=True):
            p2 = get_hyperparameters(model2, "m2", is_clf)
        selected.append((model2, p2))

    if st.button("Lancer"):
        with st.spinner("Entraînement..."):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=42)

            X_train_p = preprocessor.fit_transform(X_train)
            X_test_p = preprocessor.transform(X_test)

            results = {}
            for name, params in selected:
                try:
                    if is_clf:
                        model_map = {
                            'KNN': KNeighborsClassifier,
                            'Random Forest': RandomForestClassifier,
                            'Decision Tree': DecisionTreeClassifier,
                            'SVM': lambda **kw: SVC(probability=True, **kw),
                            'XGBoost': XGBClassifier,
                            'Logistic Regression': LogisticRegression,
                            'Gaussian NB': GaussianNB
                        }
                    else:
                        model_map = {
                            'XGBoost': XGBRegressor,
                            'Linear Regression': LinearRegression,
                            'Ridge': Ridge,
                            'Lasso': Lasso,
                            'AdaBoost': AdaBoostRegressor,
                            'Gradient Boosting': GradientBoostingRegressor
                        }
                    
                    model = model_map[name](**params)

                    model.fit(X_train_p, y_train)
                    pred = model.predict(X_test_p)

                    if is_clf:
                        results[name] = {
                            'metrics': {'Accuracy': round(accuracy_score(y_test, pred), 4)},
                            'cm': confusion_matrix(y_test, pred)
                        }
                    else:
                        results[name] = {
                            'metrics': {'R²': round(r2_score(y_test, pred), 4)},
                            'true': y_test, 'pred': pred
                        }
                except Exception as e:
                    st.error(f"{name} → {e}")

            st.success("Terminé !")
            for name, r in results.items():
                with st.expander(name):
                    st.write(r['metrics'])
                    if is_clf:
                        fig, ax = plt.subplots()
                        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', ax=ax)
                        st.pyplot(fig)
                    else:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=r['true'], y=r['pred'], mode='markers'))
                        fig.add_trace(go.Scatter(x=[r['true'].min(), r['true'].max()], y=[r['true'].min(), r['true'].max()], line=dict(color='red', dash='dash')))
                        st.plotly_chart(fig)

            if len(results) > 1:
                if st.button("Comparer"):
                    comp = pd.DataFrame({n: r['metrics'] for n, r in results.items()}).T
                    st.bar_chart(comp)