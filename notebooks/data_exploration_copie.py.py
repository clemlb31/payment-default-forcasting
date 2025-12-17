#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:





# # Data Loading
# 

# In[2]:


df = pd.read_csv('../data/Loan_data.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# # Preprocessing

# #### Détection des valeurs manquantes:

# In[5]:


nans = df.isna().sum(axis=0)
print(nans.sum())


# Pas de valeurs manquantes, pas besoin de les traiter.

# #### Traitement des dates

# In[6]:


df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'])
df['ApplicationYear'] = df['ApplicationDate'].dt.year
df['ApplicationMonth'] = df['ApplicationDate'].dt.month
df['ApplicationDay'] = df['ApplicationDate'].dt.day
df = df.drop('ApplicationDate', axis=1)


# ### Séparation des variables numériques et catégorielles

# In[7]:


numerical_vars = df.select_dtypes(include=['number']).columns.tolist()
categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()

categorical_vars_count = {col: df[col].nunique() for col in categorical_vars}


# In[8]:


print('Variables numériques:', numerical_vars)
print('Variables catégorielles:', categorical_vars_count)


# #### Encodage des variables catégorielles (one-hot encoding)

# In[9]:


drop = 'first'
encoder = OneHotEncoder(sparse_output=False, drop=drop)
encoded = encoder.fit_transform(df[list(categorical_vars_count.keys())])
encoded_categorical_vars = encoder.get_feature_names_out(list(categorical_vars_count.keys()))
encoded_df = pd.DataFrame(encoded, columns=encoded_categorical_vars)


# In[10]:


encoded_df = df[numerical_vars].join(encoded_df)


# In[11]:


encoded_df.head()


# ### Analyse exploratoire des données

# #### Matrice de corrélation des variables numériques

# In[12]:


plt.figure(figsize=(12, 10))

sns.heatmap(encoded_df[numerical_vars].corr(), 
            linewidths=0.5, 
            xticklabels=True, 
            yticklabels=True)

plt.xticks(rotation=90, ha='right') 
plt.yticks(rotation=0)    
plt.title('Matrice de corrélation des variables numériques')          

plt.show()


# #### Matrice de corrélations des variables catégorielles (après encodage)

# In[13]:


plt.figure(figsize=(12, 10))

sns.heatmap(encoded_df[encoded_categorical_vars].corr(), 
            linewidths=0.5, 
            xticklabels=True, 
            yticklabels=True)

plt.xticks(rotation=90, ha='right') 
plt.yticks(rotation=0)              
plt.title('Matrice de corrélation des variables catégorielles')
plt.show()


# In[ ]:


numerical_vars = df.select_dtypes(include=['number']).columns.tolist()

categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()

categorical_vars_count = {col: df[col].nunique() for col in categorical_vars}
print('Variables numériques:', numerical_vars)
print('Variables catégorielles:', categorical_vars_count)


# In[14]:


plt.figure(figsize=(12, 10))

sns.heatmap(encoded_df.corr(), 
            linewidths=0.5, 
            xticklabels=True, 
            yticklabels=True)

plt.xticks(rotation=90, ha='right') 
plt.yticks(rotation=0)              
plt.title('Matrice de corrélation des variables numériques et catégorielles')
plt.show()


# In[15]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='EmploymentStatus', y='RiskScore', data=df)
plt.xticks(rotation=45)
plt.title('Distribution du Risk Score par Statut Professionnel')
plt.show()


# In[16]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='EducationLevel', y='RiskScore', data=df)
plt.xticks(rotation=45)
plt.title('Distribution du Risk Score par Education')
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(x='MaritalStatus', y='RiskScore', data=df)
plt.xticks(rotation=45)
plt.title('Distribution du Risk Score par Statut Matrimonial')
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(x='HomeOwnershipStatus', y='RiskScore', data=df)
plt.xticks(rotation=45)
plt.title('Distribution du Risk Score par Statut de Propriété')
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(x='LoanPurpose', y='RiskScore', data=df)
plt.xticks(rotation=45)
plt.title('Distribution du Risk Score par Objectif du Prêt')
plt.show()


# In[17]:


plt.figure(figsize=(10, 6))
sns.histplot(df['RiskScore'], kde=True, bins=80, color='teal')
plt.title('Distribution du Risk Score (Variable Cible)')
plt.xlabel('Risk Score')
plt.ylabel('Fréquence')
plt.show()


# # Feature engineering
# 

# In[18]:


#  Transformations Logarithmiques pour réduire l'asymétrie 
cols_to_log = ['AnnualIncome', 'LoanAmount', 'NetWorth', 'TotalAssets', 'TotalLiabilities']
for col in cols_to_log:
    # On vérifie qu'il n'y a pas de valeurs négatives problématiques
    if (df[col] >= 0).all():
        df[f'Log_{col}'] = np.log1p(df[col])



# In[19]:


# Tranches d'âge
bins = [0,18, 25, 35, 45, 55, 65, 100]
labels = ['0-18','18-25', '26-35', '36-45', '46-55', '56-65', '65+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)


# In[20]:


# Total Cash 
df['Total_Cash'] = df['SavingsAccountBalance'] + df['CheckingAccountBalance']

# Ratio de couverture du prêt par le cash
df['Cash_to_Loan_Ratio'] = df['Total_Cash'] / (df['LoanAmount'] + 1)

# Ratio Épargne / Revenu Annuel 
df['Savings_to_Income'] = df['Total_Cash'] / (df['AnnualIncome'] + 1)


#  Ratio d'endettement global
df['Global_Debt_Ratio'] = df['TotalLiabilities'] / (df['TotalAssets'] + 1)

# Poids du prêt demandé par rapport au revenu annuel
df['Loan_to_Income'] = df['LoanAmount'] / (df['AnnualIncome'] + 1)


# Revenu mensuel - Dettes mensuelles actuelles - Futur paiement du prêt
df['Free_Cash_Flow'] = df['MonthlyIncome'] - df['MonthlyDebtPayments'] - df['MonthlyLoanPayment']

# Reste à vivre par personne
df['Income_Per_Capita'] = df['Free_Cash_Flow'] / (df['NumberOfDependents'] + 1)

# Spread de taux 
# Si c'est élevé, l'analyste humain ou le système précédent a déjà vu un risque.
df['Interest_Spread'] = df['InterestRate'] - df['BaseInterestRate']


# In[21]:


#count nan
df.isna().sum()


# On refait le one-hot encodding

# In[22]:


numerical_vars = df.select_dtypes(include=['number']).columns.tolist()

categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()

categorical_vars_count = {col: df[col].nunique() for col in categorical_vars}
print('Variables numériques:', numerical_vars)
print('Variables catégorielles:', categorical_vars_count)


# In[23]:


drop = 'first'
encoder = OneHotEncoder(sparse_output=False, drop=drop)
encoded = encoder.fit_transform(df[list(categorical_vars_count.keys())])
encoded_categorical_vars = encoder.get_feature_names_out(list(categorical_vars_count.keys()))
encoded_df = pd.DataFrame(encoded, columns=encoded_categorical_vars)
encoded_df = df[numerical_vars].join(encoded_df)


# In[24]:


encoded_df.head()


# # Entraînement des modèles
# 

# ### Division train test

# In[80]:


non_train_col = ['LoanApproved', 'RiskScore']

X_train, X_test, y_train, y_test = train_test_split(encoded_df.drop(columns=non_train_col), encoded_df['RiskScore'], test_size=0.2, random_state=1)


# ### Baseline: Modèle trivial et régression linéaire

#  "Modèle" aléatoire

# In[26]:


y_pred = pd.DataFrame(data=np.random.normal(df['RiskScore'].mean(), df['RiskScore'].std(), size=y_test.shape), index=y_test.index)


# In[27]:


print(f"RMSE final sur Test : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R² final sur Test   : {r2_score(y_test, y_pred):.4f}")


# Régression linéaire

# In[81]:


from sklearn.linear_model import LinearRegression

# on scale pour la régression linéaire
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


linear_regressor = LinearRegression().fit(X_train_scaled, y_train)

y_pred_lr = linear_regressor.predict(X_test_scaled)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\n--- Métriques sur X_test ---")
print(f"RMSE : {rmse_lr:.4f}") 
print(f"MAE  : {mae_lr:.4f}")   
print(f"R²   : {r2_lr:.4f}")    


# ### CART

# In[ ]:


cart_regressor = DecisionTreeRegressor(random_state=1)

param_grid = {
    'max_depth': [3, 5, 8, 12],            
    'min_samples_split': [2, 10, 20],      # Limiter les divisions trop fines
    'min_samples_leaf': [1, 5, 10, 20],     # évite les valeurs extrêmes isolées
    'max_features': [None, 'sqrt', 'log2'],
}


# In[ ]:


cv = KFold(n_splits=5, shuffle=True, random_state=1)


grid_search_cart = GridSearchCV(
    estimator=cart_regressor,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search_cart.fit(X_train, y_train)

print(f"\nMeilleurs paramètres : {grid_search_cart.best_params_}")
print(f"Meilleur RMSE (CV) : {-grid_search_cart.best_score_:.4f}")


# In[83]:


best_cart = grid_search_cart.best_estimator_
y_pred_cart = best_cart.predict(X_test)

print(f"RMSE final sur Test : {np.sqrt(mean_squared_error(y_test, y_pred_cart)):.4f}")
print(f"R² final sur Test   : {r2_score(y_test, y_pred_cart):.4f}")


# In[40]:


from sklearn.tree import DecisionTreeRegressor, plot_tree

plt.figure(figsize=(24, 12))  
plot_tree(
    best_cart, 
    feature_names=X_train.columns, 
    filled=True,            
    rounded=True,          
    fontsize=10,
    precision=2
)
plt.title("Structure de l'Arbre de Décision (Règles d'octroi)", fontsize=16)
plt.show()



# In[42]:


feat_1 = 'CreditScore'
feat_2 = 'Free_Cash_Flow' 

if feat_1 in X_train.columns and feat_2 in X_train.columns:
    X_2d = X_train[[feat_1, feat_2]]


    cart_2d = DecisionTreeRegressor(max_depth=4, random_state=42)
    cart_2d.fit(X_2d, y_train)

    x_min, x_max = X_2d[feat_1].min() - 1, X_2d[feat_1].max() + 1
    y_min, y_max = X_2d[feat_2].min() - 1, X_2d[feat_2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/100),
                         np.arange(y_min, y_max, (y_max-y_min)/100))

    # Prédiction sur la grile
    Z = cart_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis') # Viridis : Jaune=Haut Risque, Violet=Bas Risque (ou inverse selon data)
    plt.colorbar(contour, label='Risk Score Prédit')
    plt.xlabel(feat_1)
    plt.ylabel(feat_2)
    plt.title(f'Surface de Décision CART ({feat_1} vs {feat_2})')
    plt.show()
else:
    print(f"Les colonnes {feat_1} ou {feat_2} ne sont pas trouvées pour le graphe 2D.")


# In[38]:


# Affichage des importances des variables pour le modèle CART déjà entraîné

importances = best_cart.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 15 if len(indices) > 15 else len(indices)

plt.figure(figsize=(10, 6))
plt.title("Importance des variables (Arbre de Décision CART)")
plt.bar(range(top_n), importances[indices][:top_n], align="center")
plt.xticks(range(top_n), np.array(X_train.columns)[indices][:top_n], rotation=60, ha='right')
plt.xlabel("Variable")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()




# ### Random Forest

# In[43]:


rf_regressor = RandomForestRegressor(random_state=1,n_jobs=-1)




# In[44]:


param_grid = {
    'max_depth': [3, 5, 8, 12],
    'min_samples_split': [2, 10, 20],   
    'min_samples_leaf': [1, 5, 10, 20],
    'max_features': [1/3, 'sqrt', 'log2']
}



# In[45]:


cv = KFold(n_splits=5, shuffle=True, random_state=1)

grid_search_rf = GridSearchCV(
    estimator=rf_regressor,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=cv,
    n_jobs=1,
    verbose=2
)

grid_search_rf.fit(X_train, y_train)

print(f"\nMeilleurs paramètres : {grid_search_rf.best_params_}")
print(f"Meilleur RMSE (CV) : {-grid_search_rf.best_score_:.4f}")


# In[84]:


best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print(f"RMSE final sur Test : {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}")
print(f"R² final sur Test   : {r2_score(y_test, y_pred_rf):.4f}")


# In[48]:


from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# ==========================================
# GRAPHE 2 : FEATURE IMPORTANCE (Spécifique RF)
# ==========================================
# On utilise le modèle final (celui avec 300 arbres issus de la boucle)
importances = best_rf.feature_importances_
indices = np.argsort(importances)[-15:] # Top 15

plt.figure(figsize=(10, 8))
plt.title('Importance des Variables (Random Forest)')
plt.barh(range(len(indices)), importances[indices], color='forestgreen', align='center')
plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
plt.xlabel('Importance relative (MDI)')
plt.show()


# XGBoost

# In[50]:


xgb.set_config(verbosity=2)

device = 'cuda'

xgb_regressor = xgb.XGBRegressor(tree_method='hist', device = device)


# In[51]:


param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 8, 12],
    'n_estimators': [500, 1000],
    "subsample": [0.7, 1.0],        
    "colsample_bytree": [0.7, 1.0],

    "min_child_weight": [1, 5, 10], 
    "reg_lambda": [1, 5, 10],    
    "reg_alpha": [0, 0.1, 1.0],     
}




# In[ ]:


random_search_xgb = RandomizedSearchCV(
    estimator=xgb_regressor,
    param_distributions=param_grid,
    n_iter=40,
    # n_iter=10,                       
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=1,         
    verbose=2,
    random_state=1
)
random_search_xgb.fit(X_train, y_train)

print(f"\nMeilleurs paramètres : {random_search_xgb.best_params_}")
print(f"Meilleur RMSE (CV) : {-random_search_xgb.best_score_:.4f}")


# In[85]:


best_xgb = random_search_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

print(f"RMSE final sur Test : {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.4f}")
print(f"R² final sur Test   : {r2_score(y_test, y_pred_xgb):.4f}")


# In[86]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# On récupère les résultats du RandomSearch dans un DataFrame
results_df = pd.DataFrame(random_search_xgb.cv_results_)

# On nettoie les noms de colonnes pour l'affichage (supprime le préfixe 'param_')
cols = [col for col in results_df.columns if col.startswith('param_')]
rename_map = {col: col.replace('param_', '') for col in cols}
results_df = results_df.rename(columns=rename_map)

# On convertit le score négatif en RMSE positif
results_df['RMSE'] = -results_df['mean_test_score']

# GRAPHE : Impact du Learning Rate et de la Profondeur
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=results_df, 
    x='learning_rate', 
    y='RMSE', 
    hue='max_depth', 
    palette='viridis', 
    s=100, # Taille des points
    style='n_estimators' # Forme des points selon le nombre d'arbres
)
plt.title('Impact des Hyperparamètres sur le RMSE')
plt.xscale('log') # Souvent mieux pour le learning rate
plt.ylabel('RMSE')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[87]:


from xgboost import plot_importance


plt.figure(figsize=(12, 10))
# Importance type 'gain' = apport moyen d'information de la variable
plot_importance(
    best_xgb, 
    max_num_features=15, 
    height=0.5, 
    title='Top 15 Variables',
    color='teal',
    grid=False
)
plt.show()


# In[88]:


print("--- Performance du Modèle Final ---")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.4, color='#1f77b4', label='Prédictions')

# Ligne de perfection
min_val = min(y_test.min(), y_pred_xgb.min())
max_val = max(y_test.max(), y_pred_xgb.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Parfait')

plt.xlabel('Vraies Valeurs (RiskScore)')
plt.ylabel('Valeurs Prédites')
plt.title(f'XGBoost: Prédictions vs Réalité (R² = {r2_score(y_test, y_pred_xgb):.2f})')
plt.legend()
plt.show()

# Distribution des erreurs
residuals = y_test - y_pred_xgb
plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Distribution des Résidus (Biais du modèle)')
plt.xlabel('Erreur (Réel - Prédit)')
plt.axvline(0, color='red', linestyle='--')
plt.show()


# In[89]:


import shap

# Initialisation (XGBoost est très rapide avec TreeExplainer)
# Note : Si erreur CUDA, essayez de passer le modèle en CPU avant, 
# ou utilisez X_test version CPU (pandas dataframe)
print("--- Analyse SHAP (Interprétabilité Globale) ---")

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.title("Impact des variables sur le RiskScore (SHAP Summary)")
plt.show()


# ### CAT Boost

# In[101]:


# on utilise pas le encoded ici car catboost gère les catégorielles nativement

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=non_train_col), df['RiskScore'], test_size=0.2, random_state=1)


# In[64]:


cat_vars = list(categorical_vars_count.keys())
print(cat_vars)
cat_model = CatBoostRegressor(
    task_type="GPU", devices="0",
    verbose=0,              
    cat_features=cat_vars,
    loss_function='RMSE',   
    thread_count=-1,
    early_stopping_rounds=50,
    random_state=1
)


# In[ ]:


param_grid = {
    'depth': [4, 8,12],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [500, 1000],
    'l2_leaf_reg': [1, 3, 5],
}
# param_grid = {
#     'depth': [4],
#     'learning_rate': [ 0.1],
#     'iterations': [500],
#     'l2_leaf_reg': [3, 5],
# }




# In[66]:


# 4. Configurer le GridSearch
grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error', 
    cv=3,
    n_jobs=1,
    verbose=2
)

grid_search.fit(X_train, y_train)
print(f"\nMeilleurs paramètres : {grid_search.best_params_}")
print(f"Meilleur RMSE (CV) : {-grid_search.best_score_:.4f}")


# In[ ]:


best_catboost = grid_search.best_estimator_

y_pred_catB = best_catboost.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_catB))
mae = mean_absolute_error(y_test, y_pred_catB)
r2 = r2_score(y_test, y_pred_catB)


print("\n--- Métriques sur X_test ---")
print(f"RMSE : {rmse:.4f}") 
print(f"MAE  : {mae:.4f}")   
print(f"R²   : {r2:.4f}")    


# In[92]:


# Affichage de l'importance des variables (features)
importances = best_catboost.feature_importances_ if hasattr(best_catboost, "feature_importances_") else None

if importances is not None:
    importances = np.array(importances)
    feature_names = X_train.columns
    # On sélectionne les 15 features les plus importantes
    indices = np.argsort(importances)[-15:]
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), np.array(feature_names)[indices])
    plt.xlabel("Importance")
    plt.title("Importance des variables (features)")
    plt.show()
else:
    print("Le modèle ne fournit pas d'attribut feature_importances_.")


# In[93]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Analyse des Hyperparamètres (CatBoost) ---")

# Récupération des résultats du GridSearch
# (Assurez-vous que 'grid_search' est bien le nom de votre objet GridSearchCV entraîné)
results_df = pd.DataFrame(grid_search.cv_results_)

# Nettoyage des noms de colonnes
cols = [col for col in results_df.columns if col.startswith('param_')]
rename_map = {col: col.replace('param_', '') for col in cols}
results_df = results_df.rename(columns=rename_map)
results_df['RMSE'] = -results_df['mean_test_score'] # On remet le score en positif

# Graphe : Heatmap ou Scatter selon le nombre de paramètres
# Ici on visualise l'impact de la Profondeur et du Learning Rate
if 'depth' in results_df.columns and 'l2_leaf_reg' in results_df.columns:
    plt.figure(figsize=(10, 8))

    # On pivote les données pour la heatmap
    pivot_table = results_df.pivot_table(values='RMSE', index='depth', columns='l2_leaf_reg')

    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis_r") # _r pour inverser (Bleu foncé = Meilleur RMSE)
    plt.title('Performance (RMSE) selon Profondeur et l2 leaf reg')
    plt.ylabel('Profondeur de l\'arbre')
    plt.xlabel('l2 leaf reg')
    plt.show()
else:
    print("Les colonnes 'depth' ou 'l2_leaf_reg' ne sont pas dans les résultats.")


# In[94]:


from sklearn.metrics import r2_score

print("--- Qualité des Prédictions ---")

plt.figure(figsize=(10, 6))
# Nuage de points
plt.scatter(y_test, y_pred_catB, alpha=0.3, color='#e74c3c', label='Prédictions')

# Diagonale parfaite
min_val = min(y_test.min(), y_pred_catB.min())
max_val = max(y_test.max(), y_pred_catB.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Parfait')

plt.xlabel('Vrai RiskScore')
plt.ylabel('RiskScore Prédit')
plt.title(f'CatBoost: Prédiction vs Réalité (R² = {r2_score(y_test, y_pred_catB):.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# In[95]:


import shap

print("--- Analyse SHAP (Interprétabilité) ---")

# CatBoost est compatible nativement avec SHAP
explainer = shap.TreeExplainer(best_catboost)
shap_values = explainer.shap_values(X_test)

plt.figure()
# Le summary plot montre l'impact positif/négatif
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.title("Impact des variables sur le RiskScore (SHAP)")
plt.show()


# # Comparaison des modèles entre eux
# 

# In[108]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("--- COMPARAISON FINALE DES MODÈLES ---")



# 1. Rassemblement des prédictions
# Assurez-vous que ces variables existent (sinon, commentez les lignes manquantes)
# Si vous n'avez pas gardé les variables, refaites un .predict() rapide
models_preds = {
    'Baseline (Linear)': y_pred_lr if 'y_pred_lr' in locals() else None,
    'CART': y_pred_cart if 'y_pred_cart' in locals() else None,
    'Random Forest': y_pred_rf if 'y_pred_rf' in locals() else None,
    'XGBoost': y_pred_xgb if 'y_pred_xgb' in locals() else None, # ou y_pred pour le dernier modèle
    'CatBoost': y_pred_catB if 'y_pred_catB' in locals() else None
}
print(models_preds)

# Filtrer les modèles non définis
models_preds = {k: v for k, v in models_preds.items() if v is not None}

# 2. Calcul des Métriques
results = []
for name, preds in models_preds.items():
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append({'Modèle': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2})

results_df = pd.DataFrame(results).set_index('Modèle')
print("\nTableau Récapitulatif des Performances :")
print(results_df.sort_values(by='RMSE'))
results_df_sorted = results_df.sort_values(by='RMSE')

# --- Graphe 1.1 : RMSE ---
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=results_df_sorted.index, y='RMSE', data=results_df_sorted, palette='Reds_r')
plt.title('Comparaison RMSE (Erreur Quadratique Moyenne)\nPlus bas = Mieux', fontsize=14)
plt.ylabel('RMSE')
plt.xlabel('Modèles')
plt.xticks(rotation=45)
# Ajout des étiquettes
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.show()

# --- Graphe 1.2 : MAE ---
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=results_df_sorted.index, y='MAE', data=results_df_sorted, palette='Oranges_r')
plt.title('Comparaison MAE (Erreur Absolue Moyenne)\nPlus bas = Mieux', fontsize=14)
plt.ylabel('MAE')
plt.xlabel('Modèles')
plt.xticks(rotation=45)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.show()

# --- Graphe 1.3 : R² ---
# On trie différemment pour le R² (du meilleur au moins bon)
results_df_r2 = results_df.sort_values(by='R²', ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=results_df_r2.index, y='R²', data=results_df_r2, palette='Greens_r')
plt.title('Comparaison R² (Coefficient de Détermination)\nPlus haut = Mieux', fontsize=14)
plt.ylabel('R²')
plt.xlabel('Modèles')
plt.xticks(rotation=45)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.show()

# ==========================================
# GRAPHE 2 : Distribution des Erreurs (Qui se trompe le moins ?)
# ==========================================
plt.figure(figsize=(12, 7))
for name, preds in models_preds.items():
    residuals = y_test - preds
    sns.kdeplot(residuals, label=f"{name}", fill=True, alpha=0.1)

plt.title("Distribution des Résidus (Erreurs de prédiction)")
plt.xlabel("Erreur (Réel - Prédit)")
plt.xlim(-20, 20) # Ajustez cette limite selon l'échelle de votre RiskScore
plt.axvline(0, color='black', linestyle='--')
plt.legend()
plt.show()

# ==========================================
# GRAPHE 3 : Nuage de Points Comparatif
# ==========================================
# On crée une grille de subplots selon le nombre de modèles
n_models = len(models_preds)
cols = 2
rows = (n_models + 1) // 2

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, (name, preds) in enumerate(models_preds.items()):
    ax = axes[i]
    # Scatter plot
    ax.scatter(y_test, preds, alpha=0.3, s=10, color='teal')
    # Ligne idéale
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

    ax.set_title(f'{name} (R² = {results_df.loc[name, "R²"]:.3f})')
    ax.set_xlabel('Réel')
    ax.set_ylabel('Prédit')

# Cacher les axes vides s'il y en a
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

