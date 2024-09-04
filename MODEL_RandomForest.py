#Pre-processing and split of the Data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

X = df.drop(columns=["EPS_Target", "Company", "Var"])
#X = df[["Cost of Goods Sold (COGS) incl. D&A", "Total Equity"]]
y = df.EPS_Target.astype('int')

# Normaliser les données
scaler = MinMaxScaler()

# Fit the scaler on your data and transform it
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=102)

#To launch n paramters Random forest
#Random forrest parameters : optimal n from 80 to 200
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# Definir les paramètres fixes
params = {
    'max_depth': 10,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
}

# creer le modèle avec les paramètres pré-définits
model_rf = RandomForestClassifier(**params)

# Definir l'intervalle testé pour n
param_grid = {
    'n_estimators': range(80, 200)  # de 80 à 200
}

# Définir la fonction score en utilisant le F1
scorer = make_scorer(f1_score)

# Appliquer le grid search en utilisant le F1 comme score
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=5, scoring=scorer)
grid_search.fit(X_train, y_train)

#Obtenir le meilleur paramètres
best_model_rf = grid_search.best_estimator_
best_params = grid_search.best_params_

# Tester
y_hat_test_rf = best_model_rf.predict(X_test)
y_hat_train_rf = best_model_rf.predict(X_train)

# Afficher le meilleur paramètre
print("Best Parameters:", best_params)
# To launch :version of random forest test (Print + optimal parameters + tested on 50 random_state)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# Definition des paramètres
params = {
    'max_depth': 10,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 136,
}

# Define the number of random states
num_random_states = 50

# Initialize lists to store results
f1_score_0_test_rf = []
f1_score_1_test_rf = []

for random_state in range(num_random_states):
    # Create the RandomForestClassifier with the specified parameters
    model_rf = RandomForestClassifier(random_state=random_state, **params)

    # Entrainement du modèle
    model_rf.fit(X_train, y_train)

    # Test de prédiction
    y_hat_test_rf = model_rf.predict(X_test)

    # Calcul du F1 pour chaque class
    f1_score_0 = f1_score(y_test, y_hat_test_rf, pos_label=0)
    f1_score_1 = f1_score(y_test, y_hat_test_rf, pos_label=1)

    # Ajout des scores dans la liste
    f1_score_0_test_rf.append(f1_score_0)
    f1_score_1_test_rf.append(f1_score_1)
    accuracy_test_rf.append(accuracy)

# Calcul de moyenne
mean_f1_score_0 = np.mean(f1_score_0_test_rf)
mean_f1_score_1 = np.mean(f1_score_1_test_rf)
mean_accuracy = np.mean(accuracy_test_rf)

# Afficher la moyenne
print("Mean F1 Score for class 0:", mean_f1_score_0)
print("Mean F1 Score for class 1:", mean_f1_score_1)
