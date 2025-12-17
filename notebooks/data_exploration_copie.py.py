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


# In[2]:


df = pd.read_csv('../data/Loan_data.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# ### Preprocessing

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


numerical_vars = [col for col in df.columns if df[col].dtype != pd.CategoricalDtype]
categorical_vars_count = {col:df[col].nunique() for col in df.columns if df[col].dtype == pd.CategoricalDtype}


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

plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)              

plt.show()


# #### Matrice de corrélations des variables catégorielles (après encodage)

# In[13]:


plt.figure(figsize=(12, 10))

sns.heatmap(encoded_df[encoded_categorical_vars].corr(), 
            linewidths=0.5, 
            xticklabels=True, 
            yticklabels=True)

plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)              

plt.show()


# In[14]:


plt.figure(figsize=(12, 10))

sns.heatmap(encoded_df.corr(), 
            linewidths=0.5, 
            xticklabels=True, 
            yticklabels=True)

plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)              

plt.show()


# In[15]:


sns.regplot(data=encoded_df, x='CreditScore', y='RiskScore')


# In[16]:


sns.lmplot(data=encoded_df, x='NetWorth', y='RiskScore')


# In[17]:


sns.histplot(encoded_df['RiskScore'])


# Division train test

# In[18]:


target = ['LoanApproved', 'RiskScore']

X_train, X_test, y_train, y_test = train_test_split(encoded_df.drop(columns=target), encoded_df['RiskScore'], test_size=0.2, random_state=1)


# # Entraînement des modèles
# 

# #### Baseline: Modèle trivial et régression linéaire

#  "Modèle" aléatoire

# In[19]:


y_pred = pd.DataFrame(data=np.random.normal(df['RiskScore'].mean(), df['RiskScore'].std(), size=y_test.shape), index=y_test.index)


# In[20]:


print(f"RMSE final sur Test : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R² final sur Test   : {r2_score(y_test, y_pred):.4f}")


# Régression linéaire

# In[21]:


from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression().fit(X_train, y_train)

y_pred = linear_regressor.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Métriques sur X_test ---")
print(f"RMSE : {rmse:.4f}") 
print(f"MAE  : {mae:.4f}")   
print(f"R²   : {r2:.4f}")    


# CART

# In[22]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

cart_regressor = DecisionTreeRegressor(random_state=1)

param_grid = {
    'max_depth': [3, 5, 8, 12],            # Toujours contrôler la profondeur
    'min_samples_split': [2, 10, 20],      # Limiter les divisions trop fines
    'min_samples_leaf': [1, 5, 10, 20],     # Lisser les prédictions (évite les valeurs extrêmes isolées)
    'max_features': [None, 'sqrt', 'log2'],
}

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

best_cart = grid_search_cart.best_estimator_
y_pred = best_cart.predict(X_test)

print(f"RMSE final sur Test : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R² final sur Test   : {r2_score(y_test, y_pred):.4f}")


# Random Forest

# In[26]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

rf_regressor = RandomForestRegressor(random_state=1,n_jobs=-1)

param_grid = {
    'max_depth': [3, 5, 8, 12],
    'min_samples_split': [2, 10, 20],   
    'min_samples_leaf': [1, 5, 10, 20],
    'max_features': [1/3, 'sqrt', 'log2']
}


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

best_rf = grid_search_rf.best_estimator_
y_pred = best_rf.predict(X_test)

print(f"RMSE final sur Test : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R² final sur Test   : {r2_score(y_test, y_pred):.4f}")


# XGBoost

# In[ ]:


import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

xgb.set_config(verbosity=2)

device = 'cuda'

xgb_regressor = xgb.XGBRegressor(tree_method='hist', device = device)

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



random_search_xgb = RandomizedSearchCV(
    estimator=xgb_regressor,
    param_distributions=param_grid,
    n_iter=100,                        # 40 configs au lieu de 3888
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=1,                         # ou 1 si tu veux être sûr
    verbose=2,
    random_state=1
)
random_search_xgb.fit(X_train, y_train)

print(f"\nMeilleurs paramètres : {random_search_xgb.best_params_}")
print(f"Meilleur RMSE (CV) : {-random_search_xgb.best_score_:.4f}")

best_xgb = random_search_xgb.best_estimator_
y_pred = best_xgb.predict(X_test)

print(f"RMSE final sur Test : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R² final sur Test   : {r2_score(y_test, y_pred):.4f}")


# Modèles CAT Boost

# In[ ]:


TRAIN_CAT_BOOST = True
TRAIN_CART = True


# In[ ]:


# on utilise pas le encoded ici car catboost gère les catégorielles nativement
target = ['LoanApproved', 'RiskScore']

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=target), df['LoanApproved'], test_size=0.2, random_state=1)


# In[ ]:


cat_vars = list(categorical_vars_count.keys())
model = CatBoostRegressor(
    verbose=0,              
    cat_features=cat_vars,
    loss_function='RMSE',   
    thread_count=-1,
    early_stopping_rounds=50
)


# In[ ]:


# param_grid = {
#     'depth': [4, 8,12],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'iterations': [500, 1000],
#     'l2_leaf_reg': [1, 3, 5],
#     'subsample': [0.6, 0.8, 1.0]
# }
# pour le test
param_grid = {
    'depth': [4],
    'learning_rate': [0.01],
    'iterations': [500],
    'l2_leaf_reg': [1],
    'subsample': [0.6, 0.8, 1.0]
}



# In[ ]:


# 4. Configurer le GridSearch
if TRAIN_CAT_BOOST:
    grid_search = GridSearchCV(
        estimator=model,
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


if TRAIN_CAT_BOOST:
    best_model = grid_search.best_estimator_
else:

    best_model = model.load_model('../models/catboost_loan_approval_model.cbm')

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Métriques sur X_test ---")
print(f"RMSE : {rmse:.4f}") 
print(f"MAE  : {mae:.4f}")   
print(f"R²   : {r2:.4f}")    


# In[ ]:


best_model.save_model('../models/catboost_loan_approval_model.cbm')


# In[ ]:




