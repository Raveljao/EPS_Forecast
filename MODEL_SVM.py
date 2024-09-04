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

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np

# Intervalle de c value testé
c_values = range(900, 2001, 10)

# Valeur gamma testé
gamma_values = [1, 0.1, 0.01, 0.0001]

# Initialisé les variables où seront stockés la meilleur combinaison de C value, gamma, ainsi que le score F1 correspondant
best_c = None
best_gamma = None
best_mean_f1 = -1
best_mean_f1_class_0 = -1
best_mean_f1_class_1 = -1

# Itérer les gammas sur chaque c value
for c_value in c_values:
    for gamma_value in gamma_values:
        # Initialiser la list de score F1
        f1_score_0_test_list = []
        f1_score_1_test_list = []

        # Itérer le tests sur 10 random states
        for random_state in range(10):

            X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=random_state)

            # Créer le modèle SVM avec la combinaison de c value et le gamma testé
            model_svm = SVC(kernel='rbf', C=c_value, gamma=gamma_value)
            model_svm.fit(X_train, y_train)
            y_hat_test_svm = model_svm.predict(X_test)

            # Générer le rapport de classification et extraire les données souhaités score F1 pour les class 0 et 1
            report_test = classification_report(y_test, y_hat_test_svm, output_dict=True, zero_division=1)

            f1_score_0_test = report_test['0']['f1-score']
            f1_score_1_test = report_test['1']['f1-score']

            # Rajouter les valeurs F1 dans les listes
            f1_score_0_test_list.append(f1_score_0_test)
            f1_score_1_test_list.append(f1_score_1_test)

        # Calculer la moyenne des scores des listes
        mean_f1_class_0_test = np.mean(f1_score_0_test_list)
        mean_f1_class_1_test = np.mean(f1_score_1_test_list)

        mean_f1_test = (mean_f1_class_0_test + mean_f1_class_1_test) / 2

        # Vérifier que si la moyenne est meilleur que la meilleur moyenne de score F1 actuelle
        if mean_f1_test > best_mean_f1:
            best_mean_f1 = mean_f1_test
            best_mean_f1_class_0 = mean_f1_class_0_test
            best_mean_f1_class_1 = mean_f1_class_1_test
            best_c = c_value
            best_gamma = gamma_value

# Afficher la meilleur combinaison de c value et gamma, ainsi que la moyenne de score F1 correspondante
print("Optimal C value:", best_c)
print("Optimal gamma value:", best_gamma)
print("Mean F1 Score for class 0:", best_mean_f1_class_0)
print("Mean F1 Score for class 1:", best_mean_f1_class_1)
print("Mean F1 Score for both classes:", best_mean_f1)

#To launch for SVM (print filtered + optimal c + optimal gamma + on 50 random state)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np

# Initialisation de la liste pour stocker les résultats

f1_score_0_test_list = []
f1_score_1_test_list = []

# Iteration sur 50 random state
for random_state in range(50):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=random_state)

    # Création du modèle SVM & définition des paramètres
    model_svm = SVC(kernel='rbf', C=1920, gamma=0.01)

    # Entrainement et prediction du modèle
    model_svm.fit(X_train, y_train)

    y_hat_test_svm = model_svm.predict(X_test)

    # Génération d'un rapport de classification et extraction des données F1 pour les classe 0 et 1
    report_test = classification_report(y_test, y_hat_test_svm, output_dict=True, zero_division=1)

    f1_score_0_test = report_test['0']['f1-score']
    f1_score_1_test = report_test['1']['f1-score']

    # Ajout des valeurs dans les listes correspondantes

    f1_score_0_test_list.append(f1_score_0_test)
    f1_score_1_test_list.append(f1_score_1_test)

# Calcul de la moyenne des score F1

mean_f1_score_0_test = np.mean(f1_score_0_test_list)
mean_f1_score_1_test = np.mean(f1_score_1_test_list)

# Afficher la moyenne des score F1

print("Mean F1 Score for class 0:", mean_f1_score_0_test)
print("Mean F1 Score for class 1:", mean_f1_score_1_test)
